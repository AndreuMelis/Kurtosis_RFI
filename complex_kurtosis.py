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
samples = []
buffer_size = 4096
k_vect = []
k_vect_rm_dc = []
noise_variance_vect = []
sub_vect_length = 10
k = 0
i = 0
n_bins = 10#freq bins to remove from remove_dc

file_name = "data1.bin"
samples = read_binary_file(file_name, buffer_size)
N_buffer = int(len(samples)/buffer_size)

lower_thr, upper_thr = test_thr(buffer_size)

#samples_rm_dc, buffer_size_rm_dc = remove_dc(samples, buffer_size, fs, n_bins)
Sxx, f_s, t_s = spectrogram(fs, buffer_size, samples, buffer_size)

#Compute kurtosis from captured signal
i = 0
while(i<N_buffer*(buffer_size)):
    k, noise_variance = complex_kurtosis(samples[i:(buffer_size+i)], buffer_size)
    k_vect.append(k)
    noise_variance_vect.append(noise_variance)
    i += buffer_size

mean_kurtosis = np.mean(k_vect)
variance_kurtosis = np.var(k_vect)
mean_noise_variance = np.mean(noise_variance)
#k_vect = sub_vector_mean(k_vect, sub_vect_length)


print("########################\n########################\n")
print("Upper threshold = "+str(upper_thr))
print("Lower threshold = "+str(lower_thr))
print("Noise variance = "+str(mean_noise_variance))
print("Mean kurtosis = "+str(mean_kurtosis))
print("Variance Kurtosis = "+str(variance_kurtosis)+"\n")
print("########################\n########################\n")

t = np.arange(0, len(k_vect))
plt.figure(1)
plt.plot(t,k_vect)
#plt.plot(t, k_vect_rm_dc)
plt.plot(t,np.full(t.shape, lower_thr))
plt.plot(t,np.full(t.shape, upper_thr))
plt.plot(t,np.full(t.shape, mean_kurtosis))
plt.legend(["Kurtosis Captured", "Upper thr", "Lower thr", "Mean Captured Kurtosis"])
plt.title("Complex Kurtosis")
plt.ylabel("Kurtosis")
plt.xlabel("Sample per integrated buffer")
plt.figure(2)
plt.hist(samples.real, bins=100, color='red')
plt.hist(samples.imag, bins=100, color='blue')
plt.legend(['Real','Imag'])
fig = plt.figure(3)
ax = fig.gca(projection='3d')
t_mesh, f_mesh = np.meshgrid(t_s, f_s)
ax.plot_surface(t_mesh, f_mesh, Sxx, cmap='jet')
ax.set_xlabel('Time [sec]')
ax.set_ylabel('Frequency [Hz]')
ax.set_zlabel('Magnitude')

"""
fig = plt.figure(4)
Sxx_rm_dc, f_s_rm_dc, t_s_rm_dc = spectrogram(fs, buffer_size_rm_dc, samples_rm_dc)
i = 0
while(i<N_buffer*(buffer_size_rm_dc)):
    k_rm_dc, noise_variance_rm_dc = complex_kurtosis(samples_rm_dc[i:(buffer_size_rm_dc+i)], buffer_size_rm_dc)
    k_vect_rm_dc.append(k_rm_dc)
    i += buffer_size_rm_dc
mean_kurtosis_rm_dc = np.mean(k_vect_rm_dc)
ax = fig.gca(projection='3d')
t_mesh, f_mesh = np.meshgrid(t_s_rm_dc, f_s_rm_dc)
ax.plot_surface(t_mesh, f_mesh, Sxx_rm_dc, cmap='jet')
ax.set_xlabel('Time [sec]')
ax.set_ylabel('Frequency [Hz]')
ax.set_zlabel('Magnitude')
"""
plt.show()
