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
from tqdm import tqdm

fs = 4e6
fc = 100e3
buffer_size = 128
t_end = 0.3 #seconds of simulation
N_buffer = int((t_end*fs)/128)
fftsize =128 #with 0.75 overlap is a 32 fft in time domain
fft_overlap = 0.75
Trad = 0.1

#varince is divided by two as it's the variance of the complex and real part
noise = np.random.normal(0,np.sqrt(kb*300*2e6*10**(6)*4*50/2),2*buffer_size*N_buffer).view(np.complex128)
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
    rfi = chirp_signal(100e3, 2e6, t_end, t)

N_sim = 1
for i in tqdm(range(N_sim), colour='green'):
    std_th = np.sqrt(kb*300*1e6*10**(6)*50)
    noise = np.random.normal(0,std_th,2*buffer_size*N_buffer).view(np.complex128)
    t = np.arange(0.0, N_buffer*buffer_size/fs, 1/fs)
    sign = noise + rfi
    Zxx, f_s, t_s = spectrogram(fs, fftsize, sign, fft_overlap)
    #Zxx_rfi_mitigated = np.empty_like(Zxx)
    f_samples_Trad = int((Trad*fs)/(fftsize*(1-fft_overlap))) #number of non-overlapping f_samples per f_bin per T_rad
    spectral_k, mask, t_sk, Zxx_mitigated = spectral_kurtosis(Zxx, f_samples_Trad)
    
    #PLOT KURTOSIS RESULT
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    im1 = ax1.imshow(spectral_k[spectral_k.shape[0]//2:spectral_k.shape[0], :], cmap='jet', aspect='auto', extent=[t[0], t[-1], 0, fs/2], vmin = 0, vmax = 4)
    ax1.set_title('Spectral Kurtosis')
    ax1.set_xticks(np.arange(0, t_end, Trad))
    ax1.grid(True, axis='x')
    fig.colorbar(im1, ax = ax1)

    #PLOT RFI SPECTROGRA
    
    im2 = ax2.specgram(sign, NFFT=fftsize, Fs=fs, noverlap=fft_overlap*fftsize, cmap="jet", window=np.hamming(fftsize), sides='onesided', mode='magnitude')[3]
    ax2.set_title('RFI Spectrogram')
    ax2.set_xticks(np.arange(0, t_end, Trad))
    ax2.grid(True, axis='x')
    fig.colorbar(im2, ax = ax2)

    #PLOT RECOVERED RFI AFTER DETECTION
    
    t, rfi_mitigated = ispectrogram(fs, Zxx_mitigated, fftsize, fft_overlap)
    '''
    im3 = ax3.specgram(rfi_mitigated, NFFT=fftsize, Fs=fs, noverlap=fft_overlap*fftsize, cmap="jet", window=np.hamming(fftsize), sides='onesided')[3]
    ax3.set_title('Signal after detecion and mitigation')
    ax3.set_xticks(np.arange(0, t_end, Trad))
    ax3.grid(True, axis='x')
    fig.colorbar(im3, ax = ax3)
    '''
    Sxx_mitigated = 10*np.log10(np.abs(Zxx_mitigated)**2)
    im3 = ax3.imshow(Sxx_mitigated[Sxx_mitigated.shape[0]//2:Sxx_mitigated.shape[0], :], cmap='jet', aspect='auto', extent=[t[0], t[-1], 0, fs/2])
    ax3.set_title('Signal after detecion and mitigation')
    ax3.set_xticks(np.arange(0, t_end, Trad))
    ax3.grid(True, axis='x')
    fig.colorbar(im3, ax = ax3)
    
    plt.show()
    
    # MP ASSESMENT
    # compute power of rfi
    poweri = compute_power(rfi, fftsize)
    # compute power of residual rfi after mitigation
    poweri_prima = compute_power(rfi_mitigated, fftsize)

    # MP assesment
    MP = np.empty_like(poweri)
    MP = poweri_prima/poweri
    plt.figure()
    plt.plot(np.arange(0, len(MP)), MP)
    plt.show()