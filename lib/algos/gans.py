import functools

import torch
from torch import autograd

from lib.algos.base import BaseAlgo
from lib.arfnn import ResFNN
from lib.utils import sample_indices


class CGANTrainer(object):
    def __init__(
            self,
            G,
            D,
            G_optimizer,
            D_optimizer,
            p,
            q,
            gan_algo,
            base_config,
            reg_param: float = 10.
    ):
        self.G = G
        self.D = D
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer

        self.p = p
        self.q = q

        self.gan_algo = gan_algo
        self.reg_param = reg_param
        self.base_config = base_config

    def G_trainstep(self, x_fake, x_real):
        toggle_grad(self.G, True)
        self.G.train()
        self.G_optimizer.zero_grad()
        d_fake = self.D(x_fake)
        gloss = self.compute_loss(d_fake, 1)
        if self.gan_algo == 'TimeGAN':
            gloss = gloss + torch.mean((x_fake - x_real) ** 2)
        gloss.backward()
        self.G_optimizer.step()
        return gloss.item()

    def D_trainstep(self, x_fake, x_real):
        toggle_grad(self.D, True)
        self.D.train()
        self.D_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()
        d_real = self.D(x_real)
        dloss_real = self.compute_loss(d_real, 1)

        # On fake data
        x_fake.requires_grad_()
        d_fake = self.D(x_fake)
        dloss_fake = self.compute_loss(d_fake, 0)

        # Compute regularizer on fake/real
        dloss = dloss_fake + dloss_real
        dloss.backward()

        if self.gan_algo in ('RCWGAN', 'CWGAN'):
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake)
            reg.backward()
        else:
            reg = torch.ones(1)
        # Step discriminator params
        self.D_optimizer.step()

        # Toggle gradient to False
        toggle_grad(self.D, False)
        return dloss_real.item(), dloss_fake.item(), reg.item()

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        if self.gan_algo in ['RCGAN', 'TimeGAN']:
            return torch.nn.functional.binary_cross_entropy_with_logits(d_out, targets)
        elif self.gan_algo in ['RCWGAN', 'CWGAN']:
            return (2 * target - 1) * d_out.mean()


    def wgan_gp_reg(self, x_real, x_fake, center=1.):
        batch_size = x_real.size(0)
        if self.gan_algo == 'CWGAN':
            x_fake_ = x_fake.reshape(self.base_config.mc_samples, batch_size, x_real.shape[1], -1).mean(0)
        else:
            x_fake_ = x_fake
        eps = torch.rand(batch_size, device=x_real.device).view(batch_size, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake_
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.D(x_interp)
        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()
        return reg


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


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


class GAN(BaseAlgo):
    def __init__(self, base_config, gan_algo, x_real):
        super(GAN, self).__init__(base_config, x_real)
        self.D_steps_per_G_step = 2
        self.D = ResFNN(self.dim * (self.p + self.q), 1, self.hidden_dims, True).to(self.device)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=2e-4, betas=(0, 0.9))  # Using TTUR
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=1e-4, betas=(0, 0.9))

        self.gan_algo = gan_algo
        self.trainer = CGANTrainer(  # central object to tune the GAN
            G=self.G, D=self.D, G_optimizer=self.G_optimizer, D_optimizer=self.D_optimizer,
            gan_algo=gan_algo, p=self.p, q=self.q, base_config=base_config
        )

    def step(self):
        for i in range(self.D_steps_per_G_step):
            # generate x_fake
            indices = sample_indices(self.x_real.shape[0], self.batch_size)
            x_past = self.x_real[indices, :self.p].clone().to(self.device)
            with torch.no_grad():
                if self.gan_algo == 'CWGAN':
                    x_past = x_past.clone().repeat(self.base_config.mc_samples, 1, 1)
                    x_fake = self.G.sample(self.q, x_past)
                    #x_fake = x_fake.reshape(self.base_config.mc_samples, x_past.shape[0], self.q, -1).mean(0)
                else:
                    x_fake = self.G.sample(self.q, x_past.clone())
                x_fake = torch.cat([x_past, x_fake], dim=1)
            D_loss_real, D_loss_fake, reg = self.trainer.D_trainstep(x_fake, self.x_real[indices].to(self.device))
            if i == 0:
                self.training_loss['D_loss_fake'].append(D_loss_fake)
                self.training_loss['D_loss_real'].append(D_loss_real)
                if self.gan_algo in ['RCWGAN', 'CWGAN']:
                    self.training_loss['{}_reg'.format(self.gan_algo)].append(reg)
        # Generator step
        indices = sample_indices(self.x_real.shape[0], self.batch_size)
        x_past = self.x_real[indices, :self.p].clone().to(self.device)
        x_fake = self.G.sample(self.q, x_past)
        x_fake_past = torch.cat([x_past, x_fake], dim=1)
        G_loss = self.trainer.G_trainstep(x_fake_past, self.x_real[indices].clone().to(self.device))
        self.training_loss['D_loss'].append(D_loss_fake + D_loss_real)
        self.training_loss['G_loss'].append(G_loss)
        self.evaluate(x_fake)


class RCGAN(GAN,):
    def __init__(self, base_config, x_real):
        super(RCGAN, self).__init__(base_config, 'RCGAN', x_real)


class TimeGAN(GAN, ):
    def __init__(self, base_config, x_real):
        super(TimeGAN, self).__init__(base_config, 'TimeGAN', x_real)


class RCWGAN(GAN, ):
    def __init__(self, base_config, x_real):
        super(RCWGAN, self).__init__(base_config, 'RCWGAN', x_real)


class CWGAN(GAN, ):
    def __init__(self, base_config, x_real):
        super(CWGAN, self).__init__(base_config, 'CWGAN', x_real)
