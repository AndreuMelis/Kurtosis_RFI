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
f_low = 0.25e6
f_high = 1.25e6
N_kurtosis_chunk = 1 # must be multiples of 4
chunk_size = Trad*fs*4

frames = []
#Read output from rita-mwr.cpp
if len(sys.argv) > 1:
    file_name = sys.argv[1]
    print("\nStarting RFI detection\n")
else: 
    print("Usage: python3 /home/andreu/TFG/RFI/Kurtosis/complex_kurtosis_if_v2.py data/data1.bin")

file_name = "data_gps.bin"

with open(file_name, "rb") as f:
    mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
    num_samples = len(mm)//(2*4)
    num_chunks = int((num_samples-1)//chunk_size)
    t_increment = chunk_size/fs
    samples_total = np.empty(num_samples, dtype=np.complex64)
    fig, ax = plt.subplots()
    im = ax.imshow([[0, 0], [0, 0]], cmap='viridis')

    for i in tqdm(range(num_chunks), colour='green'):
        start_index = int(i*chunk_size)
        end_index = int(min((i+1)*chunk_size, num_samples))
        chunk_size = end_index - start_index
        byte_string = mm[start_index*8:end_index*8]
        complex_arr = np.frombuffer(byte_string, dtype=np.float32, count=2*chunk_size)
        complex_arr = complex_arr.reshape((-1,2))
        samples = complex_arr[:, 0] + 1j * complex_arr[:, 1]
        samples_filtered = butter_bandpass_filter(samples, f_low, f_high, fs, order = 32)
        Sxx_filtered, f_s, t = spectrogram(fs, fftsize, samples_filtered, 0.5)

        samples_total[i*chunk_size:(i+1)*chunk_size] = samples_filtered
        im = ax.imshow(10*np.log10(np.abs(Sxx_filtered)**2), aspect='auto', origin='lower', extent=[t[0]+t_increment*i, t[-1]+t_increment*i, 0, fs])

        # Add title and labels
        ax.set_xlim(t[0]+t_increment*i, t[-1]+t_increment*i)
        ax.set_ylim(0, fs)
        ax.set_title('Spectrogram')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        # Update the plot
        plt.draw()
        plt.pause(0.001)
        # Add the current figure to the list of frames
        fig.canvas.draw()
        frame = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(frame)
        # Remove the old spectrogram before plotting the next one
        im.remove()

plt.close(fig)

filename = 'bar.gif'
imageio.mimsave(filename, frames, duration=0.1)
'''
fig, ax = plt.subplots()
plt.pcolormesh(t_s, f_s[:len(f_s)//2], np.abs(np.abs(Sxx_filtered[:(fftsize//2), :])), cmap='viridis')
plt.colorbar()
"""
ax.set_yticks(np.arange(0, fs/2, fs/fftsize))
ax.grid(True, axis='y')
ax.grid(True, axis='x')
"""
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')


plt.specgram(samples_total, NFFT=fftsize, Fs=fs, noverlap=0.25*fftsize, cmap="jet", window=np.hamming(fftsize), sides='onesided')
plt.colorbar()
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
'''
plt.show()