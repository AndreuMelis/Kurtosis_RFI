from scipy.stats import norm, kurtosis
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from read_data_from_file import *
from scipy.constants import k as kb
from scipy import signal
from scipy.optimize import curve_fit
from functions import *

#Initialize variable
fs = 2e6
Trad = 100e-3
buffer_size = 4096
k_vect_noise = []
computed_noise_variance = []
time_log = []
sub_vect_length = 10
N_buffer = 1200
k = 0
i = 0
n_bins = 10#freq bins to remove from remove_dc
N_kurt_Trad = 4 #number of kurtosis values per Trad
M = int(((Trad*fs)/4096)/N_kurt_Trad) #number of buffers integrated per kurtosis value
noise_mask = np.ones(2*buffer_size*N_buffer).view(np.complex128)

noise = np.random.normal(0,np.sqrt(kb*300*2e6*10**(6)*4*50),2*buffer_size*N_buffer).view(np.complex128)

pri = 4e-1#[s]
pulsewidth = 2e-5
fc = 20e3
pri_samples = pri*fs
N_pulses = int(buffer_size*N_buffer/pri_samples)
rfi = pulsed_signal(fs, fc, buffer_size*N_buffer, N_pulses, pulsewidth)

sign = noise+rfi

mean_kurtosis = 2
var_kurtosis = 4/(M*buffer_size) 
gaus_thr = gaussian_thr(0, np.sqrt(var_kurtosis), 10e-4)
upper_thr = mean_kurtosis+gaus_thr
lower_thr = mean_kurtosis-gaus_thr

#Compute time kurtosis from generated noise
while(i<N_buffer*(buffer_size)):

    k_noise, noise_variance = complex_kurtosis(sign[i:(M*buffer_size+i)], M*buffer_size)
    k_vect_noise.append(k_noise)

    if(k_noise > upper_thr or k_noise < lower_thr):
        noise_mask[i:(M*buffer_size+i)] = 0j
        time_log.append(i/(fs))#logs time of each buffer with rfi

    computed_noise_variance.append(noise_variance)
    i += M*buffer_size

#blanking the signal with the blanking mask
clear_signal = np.multiply(sign, noise_mask)

mean_kurtosis_noise = np.mean(k_vect_noise)
variance_kurtosis_noise = np.var(k_vect_noise)
mean_compute_noise_variance = np.mean(computed_noise_variance)

print(time_log)
print("########################\n########################\n")
print("Upper threshold = "+str(upper_thr))
print("Lower threshold = "+str(lower_thr))
print("Kurtosis variance = "+str(variance_kurtosis_noise))
print("Mean noise kurtosis = "+str(mean_kurtosis_noise)+'\n')
print("########################\n########################\n")

t_k = np.arange(0, len(k_vect_noise))
plt.figure(1)
plt.plot(t_k, k_vect_noise)
plt.plot(t_k,np.full(t_k.shape, lower_thr))
plt.plot(t_k,np.full(t_k.shape, upper_thr))
plt.plot(t_k,np.full(t_k.shape, mean_kurtosis_noise))
plt.legend(["Kurtosis Noise", "Upper thr", "Lower thr", "Mean Noise Kurtosis"])
plt.title("Complex Kurtosis")
plt.ylabel("Kurtosis")
plt.xlabel("Sample per integrated buffer")

fig = plt.figure(2)
t = np.arange(0,len(sign))
major_ticks = np.arange(0,N_buffer*buffer_size, buffer_size*M)
ax = fig.add_subplot(1,1,1)
ax.set_xticks(major_ticks)
plt.plot(t, noise_mask)
plt.plot(t, sign)
plt.plot(t, clear_signal)
ax.grid(which='minor', alpha = 0.2)
ax.grid(which='major', alpha = 0.5)
plt.legend(["Mask", "Rx Signal", "Masked Signal"])
"""
def gaussian(x, A, mu, sigma):
    return A*np.exp((-x-mu)**2/(2*sigma**2))

p0 = [0.5,2,np.sqrt(var_kurtosis)]
k_hist, bin_edges = np.histogram(k_vect_noise)
params, cov = curve_fit(gaussian, bin_edges[:-1], k_hist, p0=p0)

plt.figure(2)
x = np.linspace(bin_edges[0], bin_edges[-1], 100)
plt.hist(k_vect_noise)
plt.plot(x, gaussian(x, *params), 'r-', linewidth=2)
"""
plt.show()
