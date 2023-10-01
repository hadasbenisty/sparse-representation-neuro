"""
Copyright (c) 2020 Bahareh Tolooshams
crsae model
:author: Bahareh Tolooshams
"""


import torch
import torch.nn.functional as F


class CRsAE1D(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(CRsAE1D, self).__init__()

        self.T = hyp["T"]  # number of encoder unfolding
        self.L = hyp["L"]  # 1/L is the step size in the encoder
        self.C = hyp["C"]  # number of filters
        self.K = hyp["K"]  # legnth of the filters
        self.lam = hyp["lam"]  # lambda (regularization parameter)
        self.device = hyp["device"]  # device (i.e., cpu, cuda0)

        # initialize the filter H
        if H is None:
            # initialize with random normal
            H = torch.randn((self.C, 1, self.K), device=self.device)
            # normalize that each filter has norm 1
            H = F.normalize(H, p=2, dim=-1)
        # register the filters as weights of
        # the neural network so that to be trainable.
        self.register_parameter("H", torch.nn.Parameter(H))
        # create ReLU
        self.relu = torch.nn.ReLU()

    def get_param(self, name):
        # get parameters with name
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        # normalize the filters
        self.get_param("H").data = F.normalize(self.get_param("H").data, p=2, dim=-1)

    def H_operator(self, x):
        return F.conv_transpose1d(x, self.get_param("H"))

    def HT_operator(self, x):
        return F.conv1d(x, self.get_param("H"))

    def encoder(self, y):
        enc_dim = F.conv1d(y, self.get_param("H")).shape[-1]

        x_old = torch.zeros(y.shape[0], self.C, enc_dim, device=self.device)
        x_tmp = torch.zeros(y.shape[0], self.C, enc_dim, device=self.device)
        x_new = torch.zeros(y.shape[0], self.C, enc_dim, device=self.device)
        s_old = torch.tensor(1, device=self.device).float()

        # T recurrent steps
        for t in range(self.T):
            res = y - self.H_operator(x_tmp)

            x_new = x_tmp + self.HT_operator(res) / self.L

            x_new = self.relu(x_new - self.lam / self.L)

            s_new = (1 + torch.sqrt(1 + 4 * s_old * s_old)) / 2
            x_tmp = x_new + (s_old - 1) / s_new * (x_new - x_old)

            x_old = x_new
            s_old = s_new
        return x_new

    def decoder(self, x):
        return F.conv_transpose1d(x, self.get_param("H"))

    def forward(self, y):
        # encoder
        x = self.encoder(y)
        # decoder
        y_hat = self.decoder(x)
        return y_hat, x

    def separate(self, y):
        with torch.no_grad():
            # encoder
            x = self.encoder(y)

            hx_separate = torch.zeros((y.shape[0], self.C, y.shape[-1]), device=self.device)
            for c in range(self.C):
                xc = torch.unsqueeze(x[:, c, :], dim=1)
                hc = torch.unsqueeze(self.get_param("H")[c, :, :], dim=0)
                hx_separate[:, c, :] = torch.squeeze(F.conv_transpose1d(xc, hc), dim=1)

        return hx_separate


class MultiDimBehavioralCRSAE(torch.nn.Module):
    def __init__(self,
                 y_channels,
                 trial_length,
                 trial_num,
                 e_opt_iterations,
                 m_opt_iterations,
                 smoothness_param,
                 channel_kernel_num,
                 kernel_size,
                 e_reg_coeff,
                 m_reg_coeffs,
                 device,
                 share_kernels=False,
                 ):
        super().__init__()
        self.y_channels = y_channels
        self.trial_length = trial_length
        self.trial_num = trial_num
        self.e_opt_iterations = e_opt_iterations
        self.m_opt_iterations = m_opt_iterations
        self.smoothness_param = smoothness_param
        self.channel_kernel_num = channel_kernel_num
        self.kernel_size = kernel_size
        self.e_reg_coeff = e_reg_coeff
        self.m_reg_coeff = m_reg_coeffs
        self.device = device
        self.share_kernels = share_kernels
        for c in range(self.y_channels):
            self.register_parameter(f"H_{c}", torch.nn.Parameter(H))


    def get_params(self):
        return dict(
            y_channels=self.y_channels,
            trial_length=self.trial_length,
            trial_num=self.trial_num,
            e_opt_iterations=self.e_opt_iterations,
            m_opt_iterations=self.m_opt_iterations,
            smoothness_param=self.smoothness_param,
            channel_kernel_num=self.channel_kernel_num,
            kernel_size=self.kernel_size,
            e_reg_coeff=self.e_reg_coeff,
            m_reg_coeffs=self.m_reg_coeff,
            device=self.device,
            share_kernels=self.share_kernels
        )

    def encoder(self, y):
        """

        :param y: y_channels X trial_length*trial_num
        :return:
        sparse representation of y that minimizes reconstruction and classification loss.
        """

        enc_dim = F.conv1d(y[0, :], self.get_param("H")).shape[-1]
        x_old = torch.zeros(y.shape[0], self.C, enc_dim, device=self.device)
        x_tmp = torch.zeros(y.shape[0], self.C, enc_dim, device=self.device)
        x_new = torch.zeros(y.shape[0], self.C, enc_dim, device=self.device)
        s_old = torch.tensor(1, device=self.device).float()

        # T recurrent steps
        for t in range(self.T):
            res = y - self.H_operator(x_tmp)

            x_new = x_tmp + self.HT_operator(res) / self.L

            x_new = self.relu(x_new - self.lam / self.L)

            s_new = (1 + torch.sqrt(1 + 4 * s_old * s_old)) / 2
            x_tmp = x_new + (s_old - 1) / s_new * (x_new - x_old)

            x_old = x_new
            s_old = s_new
        return x_new

    def h_operator(self, input_signal):
        """
        the input signal is
        :param input_signal:
        :return:
        """
        return input_signal





