"""
Copyright (c) 2020 Bahareh Tolooshams
crsae model
:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F

from crsae.utils import construct_toeplitz_matrix, TwoSidedReLU, has_nan_or_inf


class CRsAE1D(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(CRsAE1D, self).__init__()
        self.T = hyp["T"]  # number of encoder unfolding
        self.L = hyp.get("L", None)  # Optional L parameter in case not provided
        self.C = hyp["C"]  # number of filters
        self.K = hyp["K"]  # legnth of the filters
        self.lam = hyp["lam"]  # lambda (regularization parameter)
        self.device = hyp["device"]  # device (i.e., cpu, cuda0)
        self.twosided = hyp["twosided"]

        # initialize the filter H
        if H is None:
            # initialize with random normal
            H = torch.randn((self.C, 1, self.K), device=self.device)
            # normalize that each filter has norm 1
            H = F.normalize(H, p=2, dim=-1)
        # register the filters as weights of
        # the neural network so that to be trainable.
        self.register_parameter("H", torch.nn.Parameter(H))

        if self.L is None:
            self.L = self.calculate_L_from_filters()

        # create ReLU
        self.relu = (
            TwoSidedReLU(num_conv=self.C, lam=self.lam, L=self.L, device=self.device)
            if self.twosided
            else torch.nn.ReLU()
        )

    def get_param(self, name):
        # get parameters with name
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        # normalize the filters
        self.get_param("H").data = F.normalize(self.get_param("H").data, p=2, dim=-1)

    def calculate_L_from_filters(self):
        kernels = torch.split(self.get_param("H"), split_size_or_sections=self.C, dim=0)
        toeplitz_matrices = [construct_toeplitz_matrix(h, self.K) for h in kernels]
        H = torch.cat(toeplitz_matrices, dim=1)
        HTH = torch.matmul(H.t(), H)
        eigenvalues = torch.linalg.eigvals(HTH)
        L = torch.max(eigenvalues)
        return L

    def H_operator(self, x):
        h = self.get_param("H")
        res = F.conv_transpose1d(x, h)
        assert not has_nan_or_inf(res)
        return res

    def HT_operator(self, x):
        h = self.get_param("H")
        res = F.conv1d(x, h)
        assert not has_nan_or_inf(res)
        return res

    def encoder(self, y, H=None):
        if H is None:
            H = self.get_param("H")
        enc_dim = F.conv1d(y[0].view(1, -1), H).shape[-1]
        x_old = torch.zeros(y.shape[0], self.C, enc_dim, device=self.device)
        x_tmp = torch.zeros(y.shape[0], self.C, enc_dim, device=self.device)
        x_new = torch.zeros(y.shape[0], self.C, enc_dim, device=self.device)
        s_old = torch.tensor(1, device=self.device).float()

        # T recurrent steps
        for t in range(self.T):
            res = y - self.H_operator(x_tmp)
            x_new = x_tmp + self.HT_operator(res) / self.L

            assert not has_nan_or_inf(res)
            assert not has_nan_or_inf(self.HT_operator(res) / self.L)
            assert not has_nan_or_inf(x_new)

            x_new = self.relu(x_new - self.lam / self.L)
            s_new = (1 + torch.sqrt(1 + 4 * s_old * s_old)) / 2
            x_tmp = x_new + (s_old - 1) / s_new * (x_new - x_old)
            assert not has_nan_or_inf(x_tmp)
            x_old = x_new
            s_old = s_new
        return x_new

    def decoder(self, x, H=None):
        if H is None:
            H = self.get_param("H")
        res = F.conv_transpose1d(x, H)
        assert not has_nan_or_inf(res)
        return res

    def forward(self, y, H=None):
        # Use the encoder to get sparse representations x
        x = self.encoder(y, H)
        # Use the decoder to reconstruct the input y_hat
        y_hat = self.decoder(x, H)
        return y_hat, x

    def separate(self, y):
        with torch.no_grad():
            # encoder
            x = self.encoder(y)

            hx_separate = torch.zeros(
                (y.shape[0], self.C, y.shape[-1]), device=self.device
            )
            for c in range(self.C):
                xc = torch.unsqueeze(x[:, c, :], dim=1)
                hc = torch.unsqueeze(self.get_param("H")[c, :, :], dim=0)
                hx_separate[:, c, :] = torch.squeeze(F.conv_transpose1d(xc, hc), dim=1)

        return hx_separate
