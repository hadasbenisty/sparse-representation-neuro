import torch
import torch.nn.functional as F
import numpy as np                 # import numpy
import matplotlib.pyplot as plt    # import matplotlib
from torch.utils.data import DataLoader
from tqdm import tqdm


def vis_code(x):
    x = x.clone().detach().cpu().numpy()
    color = ["b", "r"]
    plt.figure(figsize=(20, 3))
    for c in range(x.shape[0]):
        xc = x[c, :]
        nz_x = np.where(xc > 0)[0]
        plt.plot(np.zeros(xc.shape), "black")
        plt.stem(nz_x, xc[nz_x], label="x{}".format(c + 1), linefmt="{}".format(color[c]),
                 markerfmt="o{}".format(color[c]), basefmt="black", use_line_collection=True)
        plt.legend()
        plt.xlabel("Time [ms]")
    return


def vis_code_est(x, xhat):
    x = x.clone().detach().cpu().numpy()
    xhat = xhat.clone().detach().cpu().numpy()
    color = ["b", "r"]
    plt.figure(figsize=(20, 3))

    for c in range(x.shape[0]):
        xc = x[c, :]
        nz_x = np.where(xc > 0)[0]
        plt.plot(np.zeros(xc.shape), "black")
        plt.stem(nz_x, xc[nz_x], label="x{}".format(c + 1), linefmt="black", markerfmt="o{}".format(color[c]),
                 basefmt="black", use_line_collection=True)
        xchat = xhat[c, :]
        nz_xhat = np.where(xchat > 0)[0]
        plt.stem(nz_xhat, xchat[nz_xhat], label="xhat{}".format(c + 1), linefmt="-.{}".format(color[c]),
                 markerfmt="*{}".format(color[c]), basefmt="black", use_line_collection=True)
        plt.legend()
        plt.title("x")
        plt.legend()
        plt.xlabel("Time [ms]")
    return


def vis_filters(h):
    h = h.clone().detach().cpu().numpy()
    color = ["b", "r"]
    plt.figure(figsize=(5, 2))
    for c in range(h.shape[0]):
        plt.plot(h[c, 0, :], label="true", color=color[c])
    plt.title("h")
    plt.xlabel("Time [ms]")
    return


def vis_filter_est(h, h_init, h_hat):
    if h is not None:
        h = h.clone().detach().cpu().numpy()
    h_init = h_init.clone().detach().cpu().numpy()
    h_hat = h_hat.clone().detach().cpu().numpy()
    color = ["b", "r"]
    plt.figure(figsize=(15, 3))
    for c in range(h_init.shape[0]):
        plt.subplot(1, h_init.shape[0], c + 1)
        if h is not None:
            plt.plot(h[c, 0, :], label="true", color="black")
        plt.plot(h_init[c, 0, :], "--", label="init", color="gray")
        plt.plot(h_hat[c, 0, :], label="est", color=color[c])
        plt.title("h{}".format(c + 1))
        plt.xlabel("Time [ms]")
        plt.legend()
    return


def vis_data(x):
    x = x.clone().detach().cpu().numpy()
    plt.figure(figsize=(20, 3))
    plt.plot(x, label="raw", color="black")
    plt.title("x")
    plt.xlabel("Time [ms]")
    plt.legend()
    return


def vis_data_est(y, y_hat):
    y = y.clone().detach().cpu().numpy()
    y_hat = y_hat.clone().detach().cpu().numpy()
    plt.figure(figsize=(20,3))
    plt.plot(y, label="raw", color="black")
    plt.plot(y_hat, label="denoised", color="orange")
    plt.title("y")
    plt.xlabel("Time [ms]")
    plt.legend()
    return


def vis_data_separated_est(hx1, hx2):
    hx1 = hx1.clone().detach().cpu().numpy()
    hx2 = hx2.clone().detach().cpu().numpy()
    color = ["b", "r"]
    plt.figure(figsize=(20, 3))
    plt.plot(hx1, label="1", color=color[0])
    plt.plot(hx2, label="2", color=color[1])
    plt.title("y")
    plt.xlabel("Time [ms]")
    plt.legend()


def vis_miss_false(missed_per_list, false_per_list):
    plt.figure(figsize=(10, 5))
    plt.plot(missed_per_list, false_per_list, color="black", label="CRsAE")
    plt.xlabel("True Miss [%]")
    plt.ylabel("False Alarm [%]")
    plt.ylim(0, 1.1 * np.max(false_per_list))
    plt.xlim(0, 1.1 * np.max(missed_per_list))
    plt.legend()
    return


def load_filters(device):
    return torch.load("data/h_sim_2.pt", map_location=torch.device('cpu')).to(device)


def load_h_init_harris(device):
    return torch.load("data/h_init_harris.pt", map_location=torch.device('cpu')).to(device)


def load_y_harris(device):
    return torch.load("data/y_harris.pt", map_location=torch.device('cpu')).to(device)


def load_y_series_harris(device):
    return torch.load("data/y_series_harris.pt", map_location=torch.device('cpu')).to(device)


# create distance measure for dictionary
def compute_err_h(h, h_hat):
    h = h.clone().detach()
    h_hat = h_hat.clone().detach()

    err = torch.zeros(h.shape[0])
    for c in range(h.shape[0]):
        corr = torch.sum(h[c, 0, :] * h_hat[c, 0, :])
        err[c] = torch.sqrt(torch.abs(1 - corr) ** 2)
    return err


def initialize_filter(H, device):
    flag = 1
    while flag:
        H_init = H + 0.4 * torch.randn(H.shape, device=device)
        H_init = F.normalize(H_init, p=2, dim=-1)
        if torch.max(compute_err_h(H, H_init)) < 0.5:
            if torch.min(compute_err_h(H, H_init)) > 0.4:
                flag = 0

    return H_init


def evaluate_model(dataset,net,device,criterion):

    test_loader = DataLoader(dataset, shuffle=False, batch_size=len(dataset))
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():  # "don't keep track of the gradients" -> faster, can also use .detach()
        loss_all = 0
        for idx, (y, _, _, y_true) in tqdm(enumerate(test_loader), disable=False):
            y = y.to(device)
            y_hat, _ = net(y)
            loss = criterion(y_true, y_hat) / np.var(np.array(y_true.cpu()))
            loss_all += float(loss.item())

        print(f'Test Accuracy is: {loss_all}')
        return loss_all


def evaluate_model_CNMF(dataset,net,device,criterion):

    test_loader = DataLoader(dataset, shuffle=False, batch_size=len(dataset))
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():  # "don't keep track of the gradients" -> faster, can also use .detach()
        loss_all = 0
        for idx, (y, y_true) in tqdm(enumerate(test_loader), disable=False):
            y = y.to(device)
            y_true = y_true
            y_hat, _ = net(y)
            loss = np.mean((np.array(y_true) - np.array(y_hat.cpu())) ** 2) / np.var(np.array(y_true.cpu()))
            loss_all += loss

        print(f'Test Loss is: {loss_all}')
        return loss_all
