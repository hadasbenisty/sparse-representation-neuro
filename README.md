# sparse-representation-neuro
Sparse representation for joint analysis of neuronal imaging and behavior

# 1. Deconvolution 
We consider two approaches: CRsAE (1D) and CNMF.
## Simulated Data:
a. Simulated clean data
Generate_clean_data.py generates 2 types of spikes data: 
* Using an exponential kernel (as assumed by CNMF)
* Using 2 other non-exponential kernels
Clean signals are saved with SNR = 100.
  
b. Add noise to simulated data - add_noise_to_simulated_data.m 

c. Deconvolution by CNMF main_cnmf_simulate_data.m 

d. Deconvolution by CRsAE - spike_detection_by_CRsAE_on_simulated.py, collect results - collect_simulated_data_CRsAE_results.m

## Ca imaging data - 2 photon imaging 

# 2. Deconvolution with Behavior Modeling
