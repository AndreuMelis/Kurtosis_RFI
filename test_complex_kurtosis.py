from scipy.stats import norm, kurtosis
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from read_data_from_file import *
from scipy.constants import k as kb
from scipy import signal
from functions import *

#Initialize variable
fs = 2e6
Nrad = 100e-3
Nint = 1e-3
buffer_size = 4096
k_vect_noise = []
computed_noise_variance = []
sub_vect_length = 10
N_buffer = 1000
k = 0
i = 0
n_bins = 10#freq bins to remove from remove_dc

lower_thr, upper_thr = test_thr(buffer_size*sub_vect_length)
noise = np.random.normal(0,np.sqrt(kb*300*2e6*10**(6)*4*50),2*buffer_size*N_buffer).view(np.complex128)

#Compute kurtosis from generated noise
while(i<N_buffer*(buffer_size)):
    k_noise, noise_variance = complex_kurtosis(noise[i:(buffer_size+i)], buffer_size)
    k_vect_noise.append(k_noise)
    computed_noise_variance.append(noise_variance)
    i += buffer_size

mean_kurtosis_noise = np.mean(k_vect_noise)
variance_kurtosis_noise = np.var(k_vect_noise)
mean_compute_noise_variance = np.mean(computed_noise_variance)

#Mean sub_vector_length kurtosis samples and generate a new array
#k_vect_noise = sub_vector_mean(k_vect_noise, sub_vect_length)

print("########################\n########################\n")
print("Upper threshold = "+str(upper_thr))
print("Lower threshold = "+str(lower_thr))
print("Noise variance of generated noise = "+str(mean_compute_noise_variance))
print("Mean noise kurtosis = "+str(mean_kurtosis_noise)+'\n')
print("########################\n########################\n")

t = np.arange(0, len(k_vect_noise))
plt.figure(1)
plt.plot(t, k_vect_noise)
plt.plot(t,np.full(t.shape, lower_thr))
plt.plot(t,np.full(t.shape, upper_thr))
plt.plot(t,np.full(t.shape, mean_kurtosis_noise))
plt.legend(["Kurtosis Noise", "Upper thr", "Lower thr", "Mean Noise Kurtosis"])
plt.title("Complex Kurtosis")
plt.ylabel("Kurtosis")
plt.xlabel("Sample per integrated buffer")

plt.show()
