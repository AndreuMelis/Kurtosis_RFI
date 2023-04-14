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
fc = 100e3
buffer_size = 128
t_end = 1 #seconds of simulation
N_buffer = int((t_end*fs)/128)
fftsize =128 #with 0.75 overlap is a 32 fft in time domain
fft_overlap = 0.75
Trad = 100e-3

noise = np.random.normal(0,np.sqrt(kb*300*2e6*10**(6)*4*50),2*buffer_size*N_buffer).view(np.complex128)
t = np.arange(0.0, N_buffer*buffer_size/fs, 1/fs)


#select type of rfi to be detected
rfi_type = "sin"

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

sign = noise+rfi
Y = fft(sign[0:fftsize])
X = fftfreq(fftsize, 1/fs)[:fftsize//2]
X_s = fftfreq(fftsize, 1/fs)[:fftsize]
Zxx, f_s, t_s = spectrogram(fs, fftsize, sign, fft_overlap)
f_samples_Trad = int((Trad*fs)/(fftsize*(1-fft_overlap))) #number of non-overlapping f_samples per f_bin per T_rad
spectral_k, mask, t_sk = spectral_kurtosis(Zxx, f_samples_Trad)
""""
fig = plt.figure(1)
ax = fig.gca(projection='3d')
t_mesh, f_mesh = np.meshgrid(t_s, f_s)
ax.plot_surface(t_mesh, f_mesh, Zxx, cmap='jet')
ax.set_xlabel('Time [sec]')
ax.set_ylabel('Frequency [Hz]')
ax.set_zlabel('Magnitude')
"""
"""
plt.figure(3)
plt.plot(X, 2.0/fftsize * np.abs(Y[0:fftsize//2]))
plt.grid()
"""
fig, ax = plt.subplots()
plt.imshow(spectral_k, cmap='jet', aspect='auto', extent=[t[0], t[-1], 0, fs])
#decomment follwoing lines to add vertical axis for each f_bin
"""
ax.set_yticks(np.arange(0, fs, fs/fftsize))
ax.grid(True, axis='y')
"""
plt.xlabel('Time (sub-bins)')
plt.ylabel('Frequency bin')
plt.colorbar()

fig, ax = plt.subplots()
plt.imshow(mask, cmap='binary', aspect='auto', extent=[t[0], t[-1], 0, fs])
#decomment follwoing lines to add vertical axis for each f_bin
"""
ax.set_yticks(np.arange(0, fs, fs/fftsize))
ax.grid(True, axis='y')
"""
plt.xlabel('Time (sub-bins)')
plt.ylabel('Frequency bin')
plt.colorbar()

"""
fig = plt.figure(4)
ax = fig.gca(projection='3d')
t_mesh, f_mesh = np.meshgrid(t_sk, f_s)
ax.plot_surface(t_sk, f_mesh, spectral_k, cmap='jet')
ax.set_xlabel('Time [sec]')
ax.set_ylabel('Frequency [Hz]')
ax.set_zlabel('Magnitude')
"""
"""
plt.figure(2)
plt.pcolormesh(t_s, f_s[:len(f_s)//2], np.abs(Zxx[:(fftsize//2), :])**2, cmap='viridis')
plt.colorbar()
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
"""
plt.show()