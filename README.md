# sparse-representation-neuro
Sparse representation for joint analysis of neuronal imaging and behavior

# 1. Spike Detection
We consider two approaches: CRsAE1D and CNMF.
## Data:
a. Simulated data
* Generate_clean_data.py generates spikes data using 2 kernels: an exponential kernel (as assumed by CNMF) and a Gaussian kernel. Clean signals are saved with SNR = 100.
* add_noise_to_simulated_data.m - adds noise to clean signals

b. Ca imaging data - 2 photon imaging 

# 2. Spike Detection with Behavior Modeling
