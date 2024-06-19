import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz


def construct_toeplitz_matrix(h, signal_length):
    """
    Construct a Toeplitz matrix for a given filter h and signal length.

    Parameters:
    h (torch.Tensor): 1D filter tensor.
    signal_length (int): Length of the signal for which the Toeplitz matrix is constructed.

    Returns:
    np.ndarray: A Toeplitz matrix.
    """
    # Reshape h and convert to numpy array
    h = h.view(-1).cpu().detach().numpy()

    # Create zero arrays for the first column and row of the Toeplitz matrix
    col = np.zeros(signal_length)
    row = np.zeros(signal_length)

    # Assign values to the first column and first row
    col[: len(h)] = h
    row[0] = h[0]

    # Construct and return the Toeplitz matrix
    return toeplitz(col, row)


def max_eignval_by_power_iteration(filters, encoding_dim, device, num_simulations=100):
    """
    Use the power method to find the largest eigenvalue of H^T @ H.

    Parameters:
    filters (torch.Tensor): Convolutional filters (num_filters, 1, filter_length).
    encoding_dim (int): Dimension of the encoded signal.
    device (torch.device): Device to run the computation on.
    num_simulations (int): Number of iterations for the power method.

    Returns:
    float: The largest eigenvalue.
    """

    def ht_h_operator(inp, filters_h):
        """
        Apply the H^T H operator to the input tensor.
        """
        h_b_k = F.conv_transpose1d(inp, filters_h)
        hth_b_k = F.conv1d(h_b_k, filters_h)
        return hth_b_k

    # Initialize a random vector b_k
    b_k = torch.rand(encoding_dim, device=device)

    # Power iteration loop
    for _ in range(num_simulations):
        b_k = ht_h_operator(b_k.view(1, 1, -1), filters).view(-1)  # H^T H @ b_k
        b_k1_norm = torch.norm(b_k)
        b_k = b_k / b_k1_norm

    # Rayleigh quotient for the largest eigenvalue approximation
    h_b_k = ht_h_operator(b_k.view(1, 1, -1), filters).view(-1)
    eigenvalue = torch.dot(b_k, h_b_k) / torch.dot(b_k, b_k)

    return eigenvalue.item()


class TwoSidedReLU(torch.nn.Module):
    """
    A custom two-sided ReLU activation function.

    Parameters:
    num_conv (int): Number of convolutional filters.
    lam (float, optional): Regularization parameter. Default is 1e-3.
    L (int, optional): Scaling parameter. Default is 100.
    sigma (float, optional): Standard deviation parameter. Default is 1.
    device (torch.device, optional): Device on which to place the tensor. Default is None.
    """

    def __init__(self, num_conv, lam=1e-3, L=100, sigma=1, device=None):
        super(TwoSidedReLU, self).__init__()
        self.L = L
        self.lam = torch.nn.Parameter(
            lam * torch.ones(1, num_conv, 1, 1, device=device)
        )
        self.sigma = sigma
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the two-sided ReLU activation function.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying the two-sided ReLU.
        """
        # Compute the threshold value la
        la = self.lam * (self.sigma**2)

        # Apply two-sided ReLU activation
        out = self.relu(torch.abs(x) - la / self.L) * torch.sign(x)

        return out


def has_nan_or_inf(tensor):
    """
    Check if a tensor contains any NaN or infinite values.

    Parameters:
    tensor (torch.Tensor): Input tensor.

    Returns:
    torch.Tensor: A boolean tensor indicating the presence of NaN or infinite values.
    """
    # Check for NaN values
    has_nan = torch.isnan(tensor).any()

    # Check for infinite values
    has_inf = torch.isinf(tensor).any()

    # Return logical OR of has_nan and has_inf
    return torch.logical_or(has_nan, has_inf)


# Test and plot the behavior of the two-sided ReLU activation function
def plot_two_sided_relu():
    x = (
        torch.linspace(-1, 1, 100).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    )  # Create a sample input in a smaller range

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    params = [
        (1e-3, 0.1),  # Non-linearity at ±0.01
        (1e-2, 0.1),  # Non-linearity at ±0.1
        (1e-3, 0.01),  # Non-linearity at ±0.1
        (1e-2, 0.01),  # Non-linearity at ±1.0
    ]

    for i, (lam, L) in enumerate(params):
        ax = axs[i // 2, i % 2]
        relu_two_sided = TwoSidedReLU(num_conv=1, lam=lam, L=L)
        y = relu_two_sided(x).squeeze().detach().numpy()

        threshold = lam * (1**2) / L
        ax.plot(x.squeeze().numpy(), y, label=f"λ={lam}, L={L}")
        ax.axvline(
            x=threshold,
            color="r",
            linestyle="--",
            label=f"Threshold = ±{threshold:.2f}",
        )
        ax.axvline(x=-threshold, color="r", linestyle="--")
        ax.legend()
        ax.set_title(f"Two-Sided ReLU with λ={lam}, L={L}")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Call the plot function to visualize the results
    plot_two_sided_relu()
