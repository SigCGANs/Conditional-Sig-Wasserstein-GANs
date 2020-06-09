import numpy as np
import pandas as pd
import torch


class StandardScalerTS():
    """ Standard scales a given (indexed) input vector along the specified axis. """

    def __init__(self, axis=(1)):
        self.mean = None
        self.std = None
        self.axis = axis

    def transform(self, x):
        if self.mean is None:
            self.mean = torch.mean(x, dim=self.axis)
            self.std = torch.std(x, dim=self.axis)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse_transform(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)


class Pipeline:
    def __init__(self, steps):
        """ Pre- and postprocessing pipeline. """
        self.steps = steps

    def transform(self, x, until=None):
        x = x.clone()
        for n, step in self.steps:
            if n == until:
                break
            x = step.transform(x)
        return x

    def inverse_transform(self, x, until=None):
        for n, step in self.steps[::-1]:
            if n == until:
                break
            x = step.inverse_transform(x)
        return x


def get_equities_dataset(assets=('SPX', 'DJI'), with_vol=True):
    """
    Get different returns series.
    """
    oxford = pd.read_csv('./data/oxfordmanrealizedvolatilityindices.csv')

    start = '2011-08-11 00:00:00+01:00'
    end = '2019-10-07 00:00:00+01:00'

    if assets == ('SPX', ):
        df_asset = oxford[oxford['Symbol'] == '.SPX'].set_index(['Unnamed: 0'])[start:end]
        price = np.log(df_asset[['close_price']].values)
        rtn = (price[1:] - price[:-1]).reshape(1, -1, 1)
        vol = np.log(df_asset[['medrv']].values[-rtn.shape[1]:]).reshape(1, -1, 1)
        data_raw = np.concatenate([rtn, vol], axis=-1)
    elif assets == ('SPX', 'DJI'):
        df_spx = oxford[oxford['Symbol'] == '.SPX'].set_index(['Unnamed: 0'])[start:end]
        df_dji = oxford[oxford['Symbol'] == '.DJI'].set_index(['Unnamed: 0'])[start:end]
        index = df_dji.index.intersection(df_spx.index)
        df_dji = df_dji.loc[index]
        df_spx = df_spx.loc[index]
        price_spx = np.log(df_spx[['close_price']].values)
        rtn_spx = (price_spx[1:] - price_spx[:-1]).reshape(1, -1, 1)
        vol_spx = np.log(df_spx[['medrv']].values).reshape(1, -1, 1)
        price_dji = np.log(df_dji[['close_price']].values)
        rtn_dji = (price_dji[1:] - price_dji[:-1]).reshape(1, -1, 1)
        vol_dji = np.log(df_dji[['medrv']].values).reshape(1, -1, 1)
        data_raw = np.concatenate([rtn_spx, vol_spx[:, 1:], rtn_dji, vol_dji[:, 1:]], axis=-1)
    else:
        raise NotImplementedError()
    data_raw = torch.from_numpy(data_raw).float()
    pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
    data_preprocessed = pipeline.transform(data_raw)
    return pipeline, data_raw, data_preprocessed


def get_var_dataset(window_size, batch_size=5000, dim=3, phi=0.8, sigma=0.5):
    def multi_AR(window_size, dim=3, phi=0.8, sigma=0.5, burn_in=200):
        window_size = window_size + burn_in
        xt = np.zeros((window_size, dim))
        one = np.ones(dim)
        ide = np.identity(dim)
        MU = np.zeros(dim)
        COV = sigma * one + (1 - sigma) * ide
        W = np.random.multivariate_normal(MU, COV, window_size)
        for i in range(dim):
            xt[0, i] = 0
        for t in range(window_size - 1):
            xt[t + 1] = phi * xt[t] + W[t]
        return xt[burn_in:]

    var_samples = []
    for i in range(batch_size):
        tmp = multi_AR(window_size, dim, phi=phi, sigma=sigma)
        var_samples.append(tmp)
    data_raw = torch.from_numpy(np.array(var_samples)).float()

    def get_pipeline():
        transforms = list()
        transforms.append(('standard_scale', StandardScalerTS(axis=(0, 1))))  # standard scale
        pipeline = Pipeline(steps=transforms)
        return pipeline

    pipeline = get_pipeline()
    data_preprocessed = pipeline.transform(data_raw)
    return pipeline, data_raw, data_preprocessed

def get_arch_dataset(window_size,  lag=4, bt= 0.055, N=5000, dim=1):
    """
    Creates the dataset: loads data.

    :param data_path: :param t_lag: :param device: :return: """

    def get_raw_data(N=5000,lag=4, T=2000, omega=0.00001, bt= 0.055,burn_in=2000):
        beta = bt*np.ones(lag)
        eps = np.random.randn(N, T + burn_in)
        logrtn = np.zeros((N, T + burn_in))

        initial_arch = omega / (1 -  beta[0])

        arch = initial_arch + np.zeros((N, T + burn_in))

        logrtn[:, :lag] = np.sqrt(arch[:, :lag]) * eps[:, :lag]
        
        for t in range(lag-1,T + burn_in - 1):
            
            arch[:, t + 1] = omega +  np.matmul(beta.reshape(1,-1), np.square(logrtn[:, t-lag+1:t+1]).transpose()) #* (logrtn[:, t] < 0.)
            logrtn[:, t + 1] = np.sqrt(arch[:, t + 1]) * eps[:, t + 1]
        if False:
            import matplotlib.pyplot as plt
            plt.plot(arch[0, burn_in:])
            plt.show()
            plt.plot(logrtn[0, burn_in:])
            plt.show()
        return arch[:, burn_in:], logrtn[:, burn_in:]

    pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
    vol, logrtn = get_raw_data(T=window_size, N=N, bt=bt)
    if dim == 1:
        data_raw = torch.from_numpy(logrtn[..., None]).float()
        #data_raw = torch.from_numpy(vol[..., None]).float()
    else:
        data = np.concatenate([logrtn[:, :-1, None], np.log(vol[:, 1:, None])], axis=-1)
        data_raw = torch.from_numpy(data).float()
    data_pre = pipeline.transform(data_raw)
    return pipeline, data_raw, data_pre

def rolling_window(x, x_lag):
    x = x[None, ...]
    return torch.cat([x[:, t:t + x_lag] for t in range(x.shape[1] - x_lag)], dim=0)
