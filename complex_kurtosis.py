from scipy.stats import norm, kurtosis
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from read_data_from_file import *
from scipy.constants import k as kb
from scipy import signal
from scipy.fft import fft, fftfreq
from functions import *
from tqdm import tqdm
from classes import *
import sys

# Initialize variable
fs = 4.096e6
Trad = 100e-3
buffer_size = 4096
N_kurt_Trad = 4 # number of kurtosis values per Trad
M = ((Trad*fs)/buffer_size)/N_kurt_Trad # number of buffers integrated per kurtosis value
fftsize = 128
f_low = 0.5e6
f_high = 1.5e6
N_kurtosis_chunk = 12 # must be multiples of 4
chunk_size = int(M*buffer_size*N_kurtosis_chunk)  
file_name = "time_log_rfi"
time_log_file = open(file_name, "w")


#Read output from rita-mwr.cpp
if len(sys.argv) > 1:
    file_name = sys.argv[1]
    fir_enable = int(sys.argv[2])
    print("\nStarting RFI detection\n")
else: 
    print("Usage: python3 /home/andreu/TFG/RFI/Kurtosis/complex_kurtosis_if_v2.py data/data1.bin")
    file_name = "data1.bin"
    fir_enable = 1
    exit

#file_name = "fir_manual_17_19.bin"

with open(file_name, "rb") as f:
    # Memory map the file
    mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
    
    # Calculate the number of samples in the file
    num_samples = len(mm)//(2*4)
    
    # Determine the number of chunks based on chunk_size
    num_chunks = int((num_samples-1)//chunk_size)
    num_samples_out = int((num_samples-1)%chunk_size)

    # Allocate the output arrays
    k_total = np.empty(int(num_chunks*N_kurtosis_chunk), dtype = np.float64)
    k_mean_no_rfi = np.empty_like(k_total)
    upper_thr_total = np.empty_like(k_total)
    lower_thr_total = np.empty_like(k_total)
    var_total = np.empty_like(k_total)
    var_real_total = np.empty_like(k_total)
    var_imag_total = np.empty_like(k_total)
    circularity_total = np.empty_like(k_total)
    mask_total = np.empty(num_samples, dtype=np.complex64)

    #Initial threshold
    mean_kurt_no_rfi = 2
    var_kurtosis = 4/(M*buffer_size) 
    gaus_thr = gaussian_thr(0, np.sqrt(var_kurtosis), 10e-4)
    
    cont_no_rfi = No_Rfi_Counter(0,0,mean_kurt_no_rfi)
    i_sample = 0

    # Loop over the chunks
    for i in tqdm(range(num_chunks), colour='green'):
        
        # Calculate the start and end indices for this chunk
        start_index = i*chunk_size
        end_index = min((i+1)*chunk_size, num_samples)
        chunk_size = end_index - start_index
        
        # Extract the chunk of data from the memory map
        byte_string = mm[start_index*8:end_index*8]
        complex_arr = np.frombuffer(byte_string, dtype=np.float32, count=2*chunk_size)
        complex_arr = complex_arr.reshape((-1,2))
        samples = complex_arr[:, 0] + 1j * complex_arr[:, 1]

        #Highpass filtering to avoid DC component
        samples_filtered = butter_bandpass_filter(samples, f_low, f_high, fs, order = 32)
        N_buffer = int(len(samples)/buffer_size)#number of buffers in samples

        #Compute kurtosis from filtered signal
        k, mask, upper_thr, lower_thr, noise_variance, circ, var, var_real, var_imag, cont_no_rfi, i_sample = time_kurtosis(samples_filtered, M, buffer_size, fs, gaus_thr, cont_no_rfi, time_log_file, i_sample)
        
        #Save the results of the entire dataset
        k_total[i*N_kurtosis_chunk:(i+1)*N_kurtosis_chunk] = k
        k_mean_no_rfi[i*N_kurtosis_chunk:(i+1)*N_kurtosis_chunk] = cont_no_rfi.mean*np.ones((N_kurtosis_chunk))
        mask_total[i*chunk_size:(i+1)*chunk_size] = mask
        upper_thr_total[i*N_kurtosis_chunk:(i+1)*N_kurtosis_chunk] = upper_thr
        lower_thr_total[i*N_kurtosis_chunk:(i+1)*N_kurtosis_chunk] = lower_thr
        var_total[i*N_kurtosis_chunk:(i+1)*N_kurtosis_chunk] = var
        var_real_total[i*N_kurtosis_chunk:(i+1)*N_kurtosis_chunk] = var_real
        var_imag_total[i*N_kurtosis_chunk:(i+1)*N_kurtosis_chunk] = var_imag
        circularity_total[i*N_kurtosis_chunk:(i+1)*N_kurtosis_chunk] = circ
        clear_signal = np.multiply(samples_filtered, mask)
            
time_log_file.close()

mean_kurtosis_with_rfi = np.mean(k_total)
mean_kurt_no_rfi = np.mean(k_mean_no_rfi)
variance_kurtosis = np.var(k_total)
mean_noise_variance = np.mean(noise_variance)

'''
print("########################\n########################\n")
print("Upper threshold = "+str(upper_thr))
print("Lower threshold = "+str(lower_thr))
print("Noise variance = "+str(mean_noise_variance))
print("Mean kurtosis = "+str(mean_kurtosis))
print("Variance Kurtosis = "+str(variance_kurtosis)+"\n")
print("########################\n########################\n")
'''
print("########################\n########################\n")
print("Noise variance = "+str(mean_noise_variance))
print("Mean kurtosis without rfi = "+str(mean_kurt_no_rfi))
print("Mean kurtosis with rfi = "+str(mean_kurtosis_with_rfi))
print("Variance Kurtosis = "+str(variance_kurtosis)+"\n")
print("########################\n########################\n")

#PLOT KURTOSIS AND THRESHOLDS
t_k = np.arange(0, len(k_total))
plt.figure(1)
"""
plt.plot(t,k_vect)
plt.plot(t,np.full(t.shape, lower_thr))
plt.plot(t,np.full(t.shape, upper_thr))
plt.plot(t,np.full(t.shape, mean_kurtosis))
plt.legend(["Kurtosis Captured", "Lower thr", "Upper thr","Mean Kurtosis", "Lower thr filt", "Upper thr filt", "Mean Kurtosis Filt"])
"""
plt.plot(t_k, k_total)
plt.plot(t_k,np.full(t_k.shape, lower_thr_total))
plt.plot(t_k,np.full(t_k.shape, upper_thr_total))
plt.plot(t_k,np.full(t_k.shape, k_mean_no_rfi))
plt.legend(["Kurtosis Filtered","Lower thr filt", "Upper thr filt", "Mean Kurtosis Filt"])
plt.title("Complex Kurtosis")
plt.ylabel("Kurtosis")
plt.xlabel("Sample per integrated buffer")
'''
if(fir_enable == 1):
    with open("kurtosis_fir.txt", "a") as f:
        np.savetxt(f, k_total, fmt = "%f")
else:
    with open("kurtosis.txt", "a") as f:
        np.savetxt(f, k_total, fmt = "%f")    
'''
'''
#PLOTTING WAVEFORM
fig = plt.figure(3)
t = np.arange(0,num_samples/fs, 1/fs)
#major_ticks = np.arange(0,N_buffer*buffer_size, buffer_size*M)
#ax = fig.add_subplot(1,1,1)
#ax.set_xticks(major_ticks)
plt.plot(t, np.abs(mask_total))
#plt.plot(t, samples)
#plt.plot(t, clear_signal)
'''
"""
ax.grid(which='minor', alpha = 0.2)
ax.grid(which='major', alpha = 0.5)
plt.legend(["Mask", "Rx Signal", "Masked Signal"])
"""
'''
#PLOTTING CIRCULARITY
plt.figure(5)
plt.plot(t_k, circularity_total)
plt.title('Circularity indicator')

#PLOTTING VARIANCE
plt.figure(6)
plt.plot(t_k, var_total)
plt.plot(t_k, var_real_total)
plt.plot(t_k, var_imag_total)
plt.title('Variance of captured samples')
plt.legend(['Variance Modulus', 'Variance Real', 'Variance Imaginary'])
'''
plt.show()
exit