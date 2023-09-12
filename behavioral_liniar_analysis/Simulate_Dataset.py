
import torch
import torch.nn.functional as F
import numpy as np                 # import numpy
from torch.utils.data import Dataset
import utils
import scipy


class LoadDataset(Dataset):
    def __init__(self, path, device):
        data = scipy.io.loadmat(path)['imagingData']['samples'][0, 0]
        self.y = torch.zeros((len(data), data[0].shape[0] * data[0].shape[1]), device=device)
        for i in range(len(data)):
            self.y[i] = torch.tensor(data[i].flatten())
        self.y = torch.unsqueeze(self.y, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.y[idx]


class SimulatedDataset1D(Dataset):
    def __init__(self, hyp, x = None, one_kernel = False):

        self.J = hyp["J"]
        self.N = hyp["N"]
        self.K = hyp["K"]
        self.C = hyp["C"]
        self.s = hyp["s"]
        self.x_mean = hyp["x_mean"]
        self.x_std = hyp["x_std"]
        self.snr = hyp["SNR"]
        self.device = hyp["device"]

        self.m = self.N - self.K + 1

        # load filters
        self.H = utils.load_filters(self.device)

        # generate sparse code using the already built-in function generate_x()
        if x == None:
          self.x = self.generate_x()
        else:
          self.x = x

        # do y = Hx
        with torch.no_grad():
            # create y = Hx
            if one_kernel:
                self.y = F.conv_transpose1d(self.x, self.H[0].unsqueeze(0))
            else:
                self.y = F.conv_transpose1d(self.x, self.H)
            sig_s = torch.std(self.y)
            sig_n = sig_s*10**(-self.snr/20)
            # add noise with std of noise_std to the data (y)
            self.y_noise = self.y + sig_n * torch.randn(self.y.shape, device=self.device).float()

    def generate_x(self):
        with torch.no_grad():
            x = torch.zeros((self.J, self.C, self.m), device=self.device)
            # for loop over each example
            for j in range(self.J):
                # for loop over each filter
                for c in range(self.C):
                    ind = np.random.choice(self.m, self.s, replace=False)
                    x[j, c, ind] = (torch.ones(self.s, device=self.device)) * (
                            self.x_std * torch.randn((self.s), device=self.device).float() + self.x_mean
                    )
        return x

    def __len__(self):
        return self.J

    def __getitem__(self, idx):

        return self.y_noise[idx], self.x[idx], self.H, self.y[idx]


class SimulatedCNMFDataset1D(Dataset):
    def __init__(self, hyp):
        self.g = hyp["g"]
        self.T = hyp["T"]
        self.framerate = hyp["framerate"]
        self.firerate = hyp["firerate"]
        self.b = hyp["b"]
        self.N = hyp["N"]
        self.snr = hyp["SNR"]
        self.x_mean = hyp["x_mean"]
        self.device = hyp["device"]
        self.p = len(self.g)
        self.H = utils.load_filters(self.device)

        self.noise_std = np.sqrt(self.x_mean / (10 ** (self.snr / 10)))

        self.gam = torch.cat([torch.flip(self.g.clone().detach(), [0]), torch.tensor([1.0])]).reshape(-1, 1)
        with torch.no_grad():
            trueSpikes = np.random.rand(self.N, self.T) < self.firerate / self.framerate
            self.truth = self.x_mean * torch.from_numpy(trueSpikes.astype(float)).float()
            for t in range(self.p, self.T):
                self.truth[:, t] = torch.squeeze(torch.matmul(self.truth[:, t - self.p:t + 1], self.gam))
            self.Y = self.b + self.truth + self.noise_std * torch.randn(self.N, self.T)
            self.Y = torch.unsqueeze(self.Y, 1)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.Y[idx], self.truth[idx]
