"""
Copyright (c) 2020 Bahareh Tolooshams
crsae model
:author: Bahareh Tolooshams
"""

import numpy as np
import torch
import torch.nn.functional as F
from crsae.utils import (
    construct_toeplitz_matrix,
    TwoSidedReLU,
    has_nan_or_inf,
    max_eignval_by_power_iteration,
)


class CRsAE1D(torch.nn.Module):
    def __init__(
        self,
        num_unfolding_steps,
        data_signal_length,
        num_filters,
        filter_length,
        lambda_reg,
        device,
        use_twosided_relu,
        L=None,
        filters=None,
    ):
        """
        Initialize the CRsAE1D model.

        Parameters:
        num_unfolding_steps (int): Number of encoder unfolding steps.
        data_signal_length (int): Length of data signal vector.
        num_filters (int): Number of filters.
        filter_length (int): Length of the filters.
        lambda_reg (float): Regularization parameter.
        device (str): Device to use (e.g., 'cpu', 'cuda:0').
        use_twosided_relu (bool): Whether to use two-sided ReLU.
        L (float, optional): Scaling parameter. If None, it will be calculated. Default is None.
        filters (torch.Tensor, optional): Initial filters. If None, random filters will be initialized. Default is None.
        """
        super(CRsAE1D, self).__init__()
        self.num_unfolding_steps = num_unfolding_steps
        self.data_signal_length = data_signal_length
        self.L = L  # Optional L parameter in case not provided
        self.num_filters = num_filters
        self.filter_length = filter_length
        self.lambda_reg = lambda_reg
        self.device = device
        self.use_twosided_relu = use_twosided_relu
        self.encoding_dim = self.data_signal_length - self.filter_length + 1

        # Initialize the filters
        if filters is None:
            # Initialize with random normal values
            filters = torch.randn(
                (self.num_filters, 1, self.filter_length), device=self.device
            )
            # Normalize such that each filter has norm 1
            filters = F.normalize(filters, p=2, dim=-1)
        # Register the filters as trainable parameters
        self.register_parameter("filters", torch.nn.Parameter(filters))

        if self.L is None:
            self.L = self.calculate_l_from_filters()

        # Create ReLU activation
        self.relu = (
            TwoSidedReLU(
                num_conv=self.num_filters,
                lam=self.lambda_reg,
                L=self.L,
                device=self.device,
            )
            if self.use_twosided_relu
            else torch.nn.ReLU()
        )

    def get_param(self, name):
        """
        Get parameters by name.
        """
        return self.state_dict(keep_vars=True)[name]

    def normalize_filters(self):
        """
        Normalize the filters to have a norm of 1.
        """
        self.get_param("filters").data = F.normalize(
            self.get_param("filters").data, p=2, dim=-1
        )

    def calculate_l_from_filters(self):
        """
        Calculate the scaling parameter L from the filters.
        Convergence guarantees exist for L being the Lipschitz constant of the Least-Squares element of the loss.
        That is equivalent to the largest eigenvalue of H.T @ H where H is the matrix multiplying all x's.
        """
        L = max_eignval_by_power_iteration(
            self.get_param("filters"), self.encoding_dim, self.device
        )
        return L

    def h_operator(self, x):
        """
        Apply the H operator (transpose convolution) to the input tensor.
        """
        h = self.get_param("filters")
        res = F.conv_transpose1d(x, h)
        assert not has_nan_or_inf(res)
        return res

    def ht_operator(self, x):
        """
        Apply the H^T operator (convolution) to the input tensor.
        """
        h = self.get_param("filters")
        res = F.conv1d(x, h)
        assert not has_nan_or_inf(res)
        return res

    def encoder(self, y, filters=None):
        """
        Encoder function to get sparse representations.

        Parameters:
        y (torch.Tensor): Input tensor.
        filters (torch.Tensor, optional): Filters to use. If None, use the model's filters. Default is None.

        Returns:
        torch.Tensor: Encoded sparse representations.
        """
        if filters is None:
            filters = self.get_param("filters")
        x_old = torch.zeros(
            y.shape[0], self.num_filters, self.encoding_dim, device=self.device
        )
        x_tmp = torch.zeros(
            y.shape[0], self.num_filters, self.encoding_dim, device=self.device
        )
        x_new = torch.zeros(
            y.shape[0], self.num_filters, self.encoding_dim, device=self.device
        )
        s_old = torch.tensor(1, device=self.device).float()

        # Recurrent steps
        for t in range(self.num_unfolding_steps):
            res = y - self.h_operator(x_tmp)
            x_new = x_tmp + self.ht_operator(res) / self.L

            assert not has_nan_or_inf(res)
            assert not has_nan_or_inf(self.ht_operator(res) / self.L)
            assert not has_nan_or_inf(x_new)

            x_new = self.relu(x_new - self.lambda_reg / self.L)
            s_new = (1 + torch.sqrt(1 + 4 * s_old * s_old)) / 2
            x_tmp = x_new + (s_old - 1) / s_new * (x_new - x_old)
            assert not has_nan_or_inf(x_tmp)
            x_old = x_new
            s_old = s_new
        return x_new

    def decoder(self, x, filters=None):
        """
        Decoder function to construct the denoised estimations y_hat.

        Parameters:
        x (torch.Tensor): Encoded tensor.
        filters (torch.Tensor, optional): Filters to use. If None, use the model's filters. Default is None.

        Returns:
        torch.Tensor: Denoised estimations y_hat.
        """
        if filters is None:
            filters = self.get_param("filters")
        res = F.conv_transpose1d(x, filters)
        assert not has_nan_or_inf(res)
        return res

    def forward(self, y, filters=None):
        """
        Forward pass of the CRsAE1D model.

        Parameters:
        y (torch.Tensor): Input tensor.
        filters (torch.Tensor, optional): Filters to use. If None, use the model's filters. Default is None.

        Returns:
        tuple: Denoised estimations and sparse representations.
        """
        x = self.encoder(y, filters)
        y_hat = self.decoder(x, filters)
        return y_hat, x

    def separate(self, y):
        """
        Separate the input signal into its components.

        Parameters:
        y (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Separated signal components.
        """
        with torch.no_grad():
            x = self.encoder(y)

            separated_signals = torch.zeros(
                (y.shape[0], self.num_filters, y.shape[-1]), device=self.device
            )
            for c in range(self.num_filters):
                xc = torch.unsqueeze(x[:, c, :], dim=1)
                hc = torch.unsqueeze(self.get_param("filters")[c, :, :], dim=0)
                separated_signals[:, c, :] = torch.squeeze(
                    F.conv_transpose1d(xc, hc), dim=1
                )

        return separated_signals
