from scipy.stats import norm, kurtosis
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from read_data_from_file import *
from scipy.constants import k as kb
from scipy import signal
from scipy.special import erfinv

def variance(x, n, mean_r, mean_i):
    var = 0
    for x in x:
        var += ((x.real-mean_r)*(x.real-mean_r))+((x.imag-mean_i)*(x.imag-mean_i))
    return var/n

#Time Complex Kurtosis
def complex_kurtosis(x, n):
    real_mean = 1/n*(np.sum(x.real))
    imag_mean = 1/n*(np.sum(x.imag))
    var = variance(x, n, real_mean, imag_mean)

    fourth_moment = 0
    for xr, xi in zip(x.real, x.imag):
        fourth_moment += (((xr-real_mean)*(xr-real_mean))+((xi-imag_mean)*(xi-imag_mean)))*(((xr-real_mean)*(xr-real_mean))+((xi-imag_mean)*(xi-imag_mean)))
    fourth_moment = fourth_moment/n
    return (n/(n-1))*fourth_moment/(var*var), var

def spectral_kurtosis(x, n):

    mean_kurtosis = 2
    var_kurtosis = 4/(6250) 
    gaus_thr = gaussian_thr(0, np.sqrt(var_kurtosis), 10e-4)
    upper_thr = mean_kurtosis+gaus_thr
    lower_thr = mean_kurtosis-gaus_thr

    rows, columns = np.shape(x)
    k_matrix_f = np.empty((rows, int(columns/n)))
    mask_matrix = np.empty((rows, int(columns/n)))
    fourth_moment_f = 0
    for j in range(rows):
        #select frequency bin j
        f_bin = x[j,:]
        k = 0
        i = 0
        while(i<len(f_bin)-1):
            #from frequency bin j select frequency samples corresponding to Trad
            f_sub_bin = f_bin[i:(i+n)]
            real_mean_f = np.mean(f_sub_bin.real)
            imag_mean_f = np.mean(f_sub_bin.imag)
            var_f = variance(f_sub_bin, n, real_mean_f, imag_mean_f)
            for fr, fi in zip(f_sub_bin.real, f_sub_bin.imag):
                    fourth_moment_f += (((fr-real_mean_f)*(fr-real_mean_f))+((fi-imag_mean_f)*(fi-imag_mean_f)))*(((fr-real_mean_f)*(fr-real_mean_f))+((fi-imag_mean_f)*(fi-imag_mean_f)))
            fourth_moment_f = fourth_moment_f/n
            k_f = (n/(n-1))*fourth_moment_f/(var_f*var_f)
            if(k_f > 2.5 or k_f < 1.5):
                mask = 0
            else:
                mask = 1
            k_matrix_f[j,k]=k_f
            mask_matrix[j,k]=mask
            k+=1
            i+=n
    t = []
    t = list(range(int(columns/n)))
    return k_matrix_f,mask_matrix,t

def gaussian_thr(mean, std, pfa):
    return (mean + std*np.sqrt(2)*erfinv(2*(1-pfa)-1))

def abs_spectrogram(fs, win_length, x, buffer_size):
    f,t,Zxx = signal.stft(x, fs, window='hamming', nperseg=win_length/4)
    Sxx = np.abs(Zxx)**2
    return Sxx, f, t

def spectrogram(fs, win_length, x, overlap):
    f,t,Zxx = signal.stft(x, fs, window='hamming', nperseg=win_length, nfft=win_length, noverlap=win_length*overlap)
    return Zxx, f, t

def sub_vector_mean(input_vect, sub_vect_length):
    result_vect=[]
    for i in range(0, len(input_vect), 10):
        sub_vect = input_vect[i:i+sub_vect_length]
        sub_vect_mean = np.mean(sub_vect)
        result_vect.append(sub_vect_mean)
    return result_vect

def pulsed_signal(fs, fc, length, num_pulses, pulsewidth):
    t_end = round((length/fs)/num_pulses, 4)
    t = np.arange(0, t_end-1/fs,1/fs)
    x = 0.02*np.sin(2*np.pi*fc*t)
    pulsewidth_samples = fs*pulsewidth
    N_samples = fs*t_end
    x[0:int(N_samples-pulsewidth_samples)] = 0
    i = 0
    y = x
    while(i<(num_pulses-1)):
        y = np.concatenate((y, x))
        i+=1
    if(len(y)<length):
        dif = length-len(y)
        aux = np.zeros(dif)
        y = np.concatenate((y, aux))
    if(len(y)>length):
        dif = abs(length-len(y))
        y = y[0:len(y)-dif]

    return y

def chirp_signal(f0, f1, t1, t):
    #chirp signal across the band
    rfi_chirp = 10*np.sin(2*np.pi*t*(f0 + (f1-f0)*np.power(t,2)/(3*t1**2)))
    return rfi_chirp

def sin_freq_singal(fc, fdev, fsin, t):
    #signal with frequency varying as a sinusoid
    fc = 2*np.pi*fc*t
    fm = 2*np.pi*fdev*np.sin(fsin * 2*np.pi*t)
    rfi_sin  = 0.5*np.sin(fc+fm)
    return rfi_sin

def filter_dc(x, fs):
    b, a = signal.butter(1, 0.05, 'highpass', fs=fs)
    y = signal.filtfilt(b, a, x)
    return y

