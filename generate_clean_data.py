# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 15:44:11 2023

@author: hadas
Generate clean data (SNR = 100)
"""

# -*- coding: utf-8 -*-


import torch

import Simulate_Dataset as simdata
import scipy.io as spio
use_cuda = False
# Simulate data with a general kernel
device = torch.device("cuda:0" if use_cuda else "cpu")
# dataset parameters
fs = 10 * 1000                     # sampling frequency in Hz
data_dur =  4 * 60                 # data duration in s
actpot_dur = 1.8                   # action potential length in ms
window_dur = 0.05                   # window dutation in s
C = 2                              # number of neurons
s = 3                              # number of appearance of each neuron in each window
x_mean = 2                         # code amplitude mean
x_std = 0.1                         # code amplitude std
noise_std = 0.1
snr = 20

K = int(actpot_dur * fs / 1000) # filter legnth
N = int(window_dur * fs)        # example duration
J = int(data_dur / window_dur)  # number of examples

# data parameters
data_hyp ={"J": J, "N": N, "K": K, "C": C, "s": s, "x_mean": x_mean,
           "x_std": x_std, "SNR": snr, "device": device}


cur_snr = 100
data_hyp ={"J": J, "N": N, "K": K, "C": C, "s": s, "x_mean": x_mean,
         "x_std": x_std, "SNR": cur_snr, "device": device}
dataset = simdata.SimulatedDataset1D(data_hyp)
x = dataset.x.clone().detach().cpu().numpy()
y = dataset.y.clone().detach().cpu().numpy()
data_mat = {'y': y, 'x': x}
spio.savemat(f"Simulated_Data/CRASE/CRASE_dataset_SNR_{cur_snr}.mat", data_mat)


# generate CNMF

g = torch.tensor([1.6, -0.712])
x_mean = 1
T = 500
framerate = 70
firerate = 0.3
b = 0
N = 2000

cur_snr = 100
data_hyp ={"g": g, "T": T, "framerate": framerate, "firerate": firerate, "b": b,
         "N": N,"x_mean":x_mean,"SNR": cur_snr, "device": device}
dataset = simdata.SimulatedCNMFDataset1D(data_hyp)
y = dataset.Y.clone().detach().cpu().numpy()
x = dataset.truth.clone().detach().cpu().numpy()
data_mat = {'y': y, 'x': x}
spio.savemat(f"Simulated_Data/CNMF/CNMF_SNR_{cur_snr}.mat", data_mat)