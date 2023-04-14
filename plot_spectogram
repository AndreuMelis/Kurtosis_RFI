from scipy.stats import norm, kurtosis
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from read_data_from_file import *
from scipy.constants import k as kb
from scipy import signal
from scipy.fft import fft, fftfreq
from functions import *
from numpy import matlib


fs = 2e6
buffer_size = 128
t_end = 1 #seconds of simulation
N_buffer = int((t_end*fs)/128)
fftsize =128 #with 0.75 overlap is a 32 fft in time domain
fft_overlap = 0.75
Trad = 100e-3
t = np.arange(0.0, N_buffer*buffer_size/fs, 1/fs)

noise = np.random.normal(0,np.sqrt(kb*300*2e6*10**(6)*4*50),2*buffer_size*N_buffer).view(np.complex128)

#select type of rfi to be detected
rfi_type = "chirp"

if(rfi_type == "sin"):
    #signal with frequency varying as a sinusoid
    rfi = sin_freq_singal(400e3, 5e3, 2, t)

if(rfi_type == "pulsed"):
    #burst of pulses
    pri = 4e-2
    pulsewidth = 2e-3
    f_pulses = 200e3
    pri_samples = pri*fs
    N_pulses = int(buffer_size*N_buffer/pri_samples)
    rfi = pulsed_signal(fs, f_pulses, buffer_size*N_buffer, N_pulses, pulsewidth)

if(rfi_type == "chirp"):
    #chirp signal across the band
    rfi = chirp_signal(100e3, 700e3, t_end, t)

#select type of interference
sign = noise + rfi

#compute spectogram
Zxx, f_s, t_s = spectrogram(fs, fftsize, sign, fft_overlap)

#plot spectogram


fig, ax = plt.subplots()
plt.pcolormesh(t_s, f_s[:len(f_s)//2], np.abs(Zxx[:(fftsize//2), :]), cmap='viridis')
plt.colorbar()
"""
ax.set_yticks(np.arange(0, fs/2, fs/fftsize))
ax.grid(True, axis='y')
ax.grid(True, axis='x')
"""
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

"""
plt.specgram(sign, NFFT=fftsize, Fs=fs, noverlap=fft_overlap*fftsize, cmap="jet", window=np.hamming(fftsize))
plt.colorbar()
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
"""
plt.show()