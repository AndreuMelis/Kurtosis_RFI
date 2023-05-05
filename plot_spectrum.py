from scipy.stats import norm, kurtosis
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from read_data_from_file import *
from functions import *
from tqdm import tqdm
from PIL import Image
import imageio
import sys
from scipy.fft import fft, fftfreq


#Initialize variable
fs = 4.096e6
Trad = 100e-3
Nint = 1e-3
samples = []
buffer_size = 4096
k_vect = []
k_vect_filtered = []
time_log = []
k = 0
N_kurt_Trad = 4 #number of kurtosis values per Trad
N_buffer = int(len(samples)/buffer_size)
M = int(((Trad*fs)/4096)/N_kurt_Trad) #number of buffers integrated per kurtosis value
fftsize = 1024
f_low = 0.5e6
f_high = 1.5e6
N_kurtosis_chunk = 1 # must be multiples of 4
chunk_size = Trad*fs*4

frames = []

#Read output from rita-mwr.cpp
if len(sys.argv) > 1:
    file_name = sys.argv[1]
    print("\nStarting spectrum computation\n")
else: 
    file_name = "data_gps.bin"

#file_name = "nofilter.bin"

with open(file_name, "rb") as f:
    mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
    num_samples = len(mm)//(2*4)
    num_chunks = int((num_samples-1)//chunk_size)
    t_increment = chunk_size/fs
    samples_total = np.empty(num_samples, dtype=np.complex64)
    fig, ax = plt.subplots()

    for i in tqdm(range(num_chunks), colour='green'):
        start_index = int(i*chunk_size)
        end_index = int(min((i+1)*chunk_size, num_samples))
        chunk_size = end_index - start_index
        byte_string = mm[start_index*8:end_index*8]
        complex_arr = np.frombuffer(byte_string, dtype=np.float32, count=2*chunk_size)
        complex_arr = complex_arr.reshape((-1,2))
        samples = complex_arr[:, 0] + 1j * complex_arr[:, 1]
        samples_filtered = butter_bandpass_filter(samples, f_low, f_high, fs, order=32)
        X = fft(samples_filtered, norm="ortho")
        freq = fftfreq(len(samples_filtered), 1/fs)
        freq = freq[:len(freq)//2]
        X_mag = np.abs(X[:len(X)//2])
        samples_total[i*chunk_size:(i+1)*chunk_size] = samples_filtered
        # Add title and labels
        ax.clear()
        ax.plot(freq, X_mag)
        ax.set_xlim(0, fs/2)
        ax.set_title('Spectrum')
        ax.set_ylabel('Magnitude')
        ax.set_xlabel('Frequency (Hz)')
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)

plt.close(fig)

#filename = 'spectrum_fir_matlab.gif'
#imageio.mimsave(filename, frames, duration=0.1)
samples_total = samples_total[0:4095]
X_total = fft(samples_total)
freq_total = fftfreq(len(samples_total), 1/fs)
freq_total = freq_total[:len(freq_total)//2]
X_mag_total = 10*np.log10((np.abs(X_total[:len(X_total)//2])/num_samples)**2)
#ax = plt.gca()
#ax.set_ylim([-15, 0])
plt.plot(freq_total, X_mag_total)
plt.title("Captured Spectrum")
plt.ylabel("Magnitude [dB]")
plt.xlabel("Frequency [Hz]")
plt.grid(which='both', axis='both')
plt.show()