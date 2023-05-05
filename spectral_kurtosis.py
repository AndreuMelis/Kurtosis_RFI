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

fs = 4.096e6
fc = 100e3
buffer_size = 128
t_end = 0.5 #seconds of simulation
N_buffer = int((t_end*fs)/128)
fftsize =128 #with 0.75 overlap is a 32 fft in time domain
fft_overlap = 0.75
Trad = 100e-3

#varince is divided by two as it's the variance of the complex and real part
noise = np.random.normal(0,np.sqrt(kb*300*2e6*10**(6)*4*50/2),2*buffer_size*N_buffer).view(np.complex128)
t = np.arange(0.0, N_buffer*buffer_size/fs, 1/fs)


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

std_vect = []
std_expected_vect = []
mean_vect = []

mean_noise_vect = []
std_noise_vect = []
std_th_vect = []

i = 0
N_sim = 5
for i in tqdm(range(N_sim), colour='green'):
    std_th = np.sqrt(kb*300*1e6*10**(6)*50)
    noise = np.random.normal(0,std_th,2*buffer_size*N_buffer).view(np.complex128)
    t = np.arange(0.0, N_buffer*buffer_size/fs, 1/fs)
    sign = noise
    Zxx, f_s, t_s = spectrogram(fs, fftsize, sign, fft_overlap)
    f_samples_Trad = int((Trad*fs)/(fftsize*(1-fft_overlap))) #number of non-overlapping f_samples per f_bin per T_rad
    spectral_k, mask, t_sk = spectral_kurtosis(Zxx, f_samples_Trad)
    '''
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
    plt.imshow(spectral_k[spectral_k.shape[0]//2:spectral_k.shape[0], :], cmap='jet', aspect='auto', extent=[t[0], t[-1], 0, fs/2], vmin = 0, vmax = 4)
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
    '''

    #STATS AND PLOTS OF KURTOSIS
    plt.subplot(2,1,1)
    mean = np.round(np.mean(spectral_k.flatten()), 6)
    std = np.round(np.std(spectral_k.flatten()), 6)
    std_expected = np.round((np.sqrt(4/12500)),6)

    print('KURTOSIS STATISTICS')
    print('Mean = '+str(mean))
    print('Expected deviation = ', str(np.round(2-2*((12500-1)/12500),6)))
    plt.hist(spectral_k.flatten(), bins=100, stacked = True)
    plt.axvline(mean, color='r')
    plt.axvline((std+2), color='g')
    plt.axvline((std_expected+2), color='y')
    plt.legend(['Kurtosis Mean', 'Kurtosis std', 'Expected Kurtosis std', 'Kurtosis Hist'])
    plt.title('Kurtosis Histogram')
    bbox = dict(boxstyle ="round", fc ="0.8")
    plt.annotate('Mean deviation = '+str(np.round(np.abs(mean-2),6)), xy=(1.94,17.5), bbox = bbox)
    plt.annotate('Expected mean deviation= '+str(np.round(2-2*((12500-1)/12500),6)), xy=(1.94,15), bbox = bbox)
    plt.annotate('Std deviation = '+str(std), xy=(1.94,12.5), bbox = bbox)
    plt.annotate('Expected std = '+str(std_expected), xy=(1.94,10), bbox = bbox)

    #STATS AND PLOTS OF SIGNAL
    plt.subplot(2,1,2)
    mean_noise = np.round(np.abs(np.mean(noise)),6)
    std_noise = np.round(np.std(noise), 6)
    
    print('NOISE STATISTICS')
    print('Mean = '+str(mean))
    plt.hist(noise, bins=100)
    plt.axvline(mean_noise, color='r')
    plt.axvline(std_noise, color='g')
    plt.axvline(std_th, color='y')
    plt.legend(['Noise Mean', 'Noise std', 'Expected Noise std', 'Noise Hist'])
    plt.title('Noise Histogram')
    plt.annotate('Deviation from mean = '+str(np.abs(mean_noise)), xy=(-0.002,40000), bbox = bbox)
    plt.annotate('Std deviation = '+str(std_noise), xy=(-0.002,40000-6000), bbox = bbox)
    plt.annotate('Expected std = '+str(np.round(std_th,6)), xy=(-0.002,40000-12000), bbox = bbox)
    plt.show()
    
    mean_vect.append(mean)
    std_vect.append(std)
    std_expected_vect.append(std_expected)

    mean_noise_vect.append(mean_noise)
    std_noise_vect.append(std_noise)
    std_th_vect.append(std_th)

    i+=1

x = np.arange(0, len(mean_vect), 1)
plt.figure()
plt.subplot(2,1,1)
plt.plot(x, mean, color = 'r')
plt.axhline(2*((12500-1)/12500))
plt.axhline(np.mean(mean))
plt.title('Kurtosis Mean')
plt.legend(['Actual', 'Expected = '+str(np.round(2*((12500-1)/12500),6)), 'Mean of actual mean = '+str(np.mean(mean))])
plt.subplot(2,1,2)
plt.plot(x, std, color='g')
plt.plot(x, std_expected, color='y')
plt.axhline(np.mean(std))
plt.title('Kurtosis Std deviation')
plt.legend(['Expected std', 'Actual std', 'Mean of std = '+str(np.mean(std))])

plt.figure()
plt.subplot(2,1,1)
plt.plot(x, mean_noise_vect, color = 'r')
plt.title('Mean Noise')
plt.subplot(2,1,2)
plt.plot(x, std_noise_vect,color = 'g')
plt.plot(x, std_th_vect, color = 'y')
plt.title('Noise Std deviation')
plt.legend(['Expected std', 'Actual std'])
plt.show()