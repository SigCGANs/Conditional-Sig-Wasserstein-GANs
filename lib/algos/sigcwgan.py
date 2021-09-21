from dataclasses import dataclass

import math
import torch
from torch import autograd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from torch import optim
from lib.arfnn import ResFNN

from torch.nn.utils import weight_norm
from lib.algos.base import BaseAlgo, BaseConfig
from lib.algos.gans import CGANTrainer
from lib.augmentations import SignatureConfig
from lib.augmentations import augment_path_and_compute_signatures
from lib.utils import sample_indices, to_numpy


def sigcwgan_loss(sig_pred: torch.Tensor, sig_fake_conditional_expectation: torch.Tensor):
    return torch.norm(sig_pred - sig_fake_conditional_expectation, p=2, dim=1).mean()
def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

@dataclass
class SigCWGANConfig:
    mc_size: int
    sig_config_future: SignatureConfig
    sig_config_past: SignatureConfig

    def compute_sig_past(self, x):
        return augment_path_and_compute_signatures(x, self.sig_config_past)

    def compute_sig_future(self, x):
        return augment_path_and_compute_signatures(x, self.sig_config_future)


def calibrate_sigw1_metric(config, x_future, x_past):
    sigs_past = config.compute_sig_past(x_past)
    sigs_future = config.compute_sig_future(x_future)
    assert sigs_past.size(0) == sigs_future.size(0)
    X, Y = to_numpy(sigs_past), to_numpy(sigs_future)
    lm = LinearRegression()
    lm.fit(X, Y)
    sigs_pred = torch.from_numpy(lm.predict(X)).float().to(x_future.device)
    return sigs_pred


def sample_sig_fake(G, q, sig_config, x_past):
    x_past_mc = x_past.repeat(sig_config.mc_size, 1, 1).requires_grad_()
    x_fake = G.sample(q, x_past_mc)
    sigs_fake_future = sig_config.compute_sig_future(x_fake)
    sigs_fake_ce = sigs_fake_future.reshape(sig_config.mc_size, x_past.size(0), -1).mean(0)
    return sigs_fake_ce, x_fake

def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg
  
def sample_x_fake(G, q, sig_config, x_past):
    x_past_mc = x_past.repeat(sig_config.mc_size, 1, 1).requires_grad_()
    x_fake_mc = G.sample(q, x_past_mc)

    return  x_fake_mc


class SigCWGAN(BaseAlgo):
    def __init__(
            self,
            base_config: BaseConfig,
            config: SigCWGANConfig,
            x_real: torch.Tensor,
    ):
        super(SigCWGAN, self).__init__(base_config, x_real)
        self.sig_config = config
        self.mc_size = config.mc_size

        self.x_past = x_real[:, :self.p]
        x_future = x_real[:, self.p:]
        self.sigs_pred = calibrate_sigw1_metric(config, x_future, self.x_past)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=1e-2)
        self.G_scheduler = optim.lr_scheduler.StepLR(self.G_optimizer, step_size=100, gamma=0.9)

    def sample_batch(self, ):
        random_indices = sample_indices(self.sigs_pred.shape[0], self.batch_size)  # sample indices
        # sample the least squares signature and the log-rtn condition
        sigs_pred = self.sigs_pred[random_indices].clone().to(self.device)
        x_past = self.x_past[random_indices].clone().to(self.device)
        return sigs_pred, x_past

    def step(self):
        self.G.train()
        self.G_optimizer.zero_grad()  # empty 'cache' of gradients
        sigs_pred, x_past = self.sample_batch()
        sigs_fake_ce, x_fake = sample_sig_fake(self.G, self.q, self.sig_config, x_past)
        loss = sigcwgan_loss(sigs_pred, sigs_fake_ce)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(), 10)
        self.training_loss['loss'].append(loss.item())
        self.training_loss['total_norm'].append(total_norm)
        self.G_optimizer.step()
        self.G_scheduler.step()  # decaying learning rate slowly.
        self.evaluate(x_fake)

class MCWGAN_1(BaseAlgo):
    def __init__(
            self,
            base_config: BaseConfig,
            config: SigCWGANConfig,
            x_real: torch.Tensor,
    ):
        super(MCWGAN_1, self).__init__(base_config, x_real)
        self.sig_config = config
        self.mc_size = config.mc_size
        #function D
        self.D = ResFNN(self.dim * (self.q), 1, self.hidden_dims, True).to(self.device)
        #self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=2e-4, betas=(0, 0.9))  # Using TTUR
        
        self.x_past = x_real[:, :self.p]
        self.x_future = x_real[:, self.p:]

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=1e-2)
        self.G_scheduler = optim.lr_scheduler.StepLR(self.G_optimizer, step_size=100, gamma=0.9)
        self.reg_param=10
        self.steps_per_phi=20
    def sample_batch(self, ):
        random_indices = sample_indices(self.x_past.shape[0], self.batch_size)  # sample indices
        # sample the x_future and x_past
        x_future=self.x_future[random_indices].clone().to(self.device)
        x_past = self.x_past[random_indices].clone().to(self.device)
        return x_past,x_future

    def mcwgan_loss(self,x_future: torch.Tensor, x_fake_mc: torch.Tensor,D):
        d_real=D(x_future)
        d_fake=D(x_fake_mc).reshape(self.sig_config.mc_size,x_future.size(0), -1).mean(0)
        return torch.norm(d_real-d_fake, p=2, dim=1).mean()
        
    def fit(self):
        if self.batch_size == -1:
            self.batch_size = self.x_real.shape[0]
        for i in tqdm(range(self.total_steps), ncols=80):  # sig_pred, x_past, x_real
            self.step(i)
            
    def step(self,i):
        if i%self.steps_per_phi==0:
            self.D.apply(weights_init_uniform)# resample the weights of model 
        x_past,x_future = self.sample_batch()
        x_fake_mc = sample_x_fake(self.G, self.q, self.sig_config, x_past)
        loss=self.G_step(x_future,x_fake_mc)
        self.training_loss['loss'].append(loss)
        self.training_loss['Reg'].append(0)
        with torch.no_grad():
          x_fake_future = self.G.sample(self.q, x_past)#get samples for evaluation
        self.evaluate(x_fake_future)
        
    #def D_step(self,x_future,x_fake):
        #x_fake.requires_grad_()
        #x_future.requires_grad_()
        ##toggle_grad(self.D, True)
        #self.D.train()
        #self.D_optimizer.zero_grad()  # empty 'cache' of gradients
        #reg = self.reg_param * self.wgan_gp_reg(x_future, x_fake)
        #reg.backward()
        #self.D_optimizer.step()
        #return reg.item()
        
    def G_step(self,x_future,x_fake_mc):
        x_future.requires_grad_()
        x_fake_mc.requires_grad_()
        toggle_grad(self.G, True)
        self.G.train() 
        self.G_optimizer.zero_grad()  # empty 'cache' of gradients
        self.D.eval()
        loss = self.mcwgan_loss(x_future, x_fake_mc,self.D)
        loss.backward()
        self.G_optimizer.step()
        self.G_scheduler.step()  # decaying learning rate slowly.
        return loss.item()

    #def wgan_gp_reg(self, x_real, x_fake, center=1.):
       # batch_size = x_real.size(0)
        #eps = torch.rand(batch_size, device=x_real.device).view(batch_size, 1, 1)
        #x_interp = (1 - eps) * x_real + eps * x_fake
        #x_interp = x_interp.detach()
        #x_interp.requires_grad_()
        #d_out= self.D(x_interp)
        #reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()
        #return reg
    

def weights_init_uniform(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        with torch.no_grad(): 
          if classname.find('Linear') != -1:
            stdv=1./math.sqrt(m.weight.size(1))
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
              m.bias.data.uniform_(-stdv, stdv)
            
            
#def weights_norm(m):
#        classname = m.__class__.__name__
#        # for every Linear layer in a model..
#        if classname.find('Linear') != -1:
#            # apply a uniform distribution to the weights and a bias=0
#            m.weight.data.uniform_(0.0, 1.0)
#            m.bias.data.fill_(0)

    #model_uniform = Net()
    #model_uniform.apply(weights_init_uniform)

#TORCH.NN.UTILS.WEIGHT_NORM