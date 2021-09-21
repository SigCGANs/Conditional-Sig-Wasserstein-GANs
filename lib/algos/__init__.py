from lib.algos.gans import RCGAN, RCWGAN, TimeGAN
from lib.algos.gmmn import GMMN
from lib.algos.sigcwgan import SigCWGAN,MCWGAN_1

ALGOS = dict(SigCWGAN=SigCWGAN, TimeGAN=TimeGAN, RCGAN=RCGAN, GMMN=GMMN, RCWGAN=RCWGAN,MCWGAN_1=MCWGAN_1)
