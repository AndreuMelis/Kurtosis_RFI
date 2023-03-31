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
    columns, rows = np.shape(x)
    k_matrix_f = np.empty((columns, int(rows/n)))
    fourth_moment_f = 0
    for j in range(columns):
        f_bin = x[j,:]
        k = 0
        i = 0
        while(i<len(f_bin)-1):
            f_sub_bin = f_bin[i:(i+n)]
            real_mean_f = np.mean(f_sub_bin.real)
            imag_mean_f = np.mean(f_sub_bin.imag)
            var_f = variance(f_sub_bin, n, real_mean_f, imag_mean_f)
            for fr, fi in zip(f_sub_bin.real, f_sub_bin.imag):
                    fourth_moment_f += (((fr-real_mean_f)*(fr-real_mean_f))+((fi-imag_mean_f)*(fi-imag_mean_f)))*(((fr-real_mean_f)*(fr-real_mean_f))+((fi-imag_mean_f)*(fi-imag_mean_f)))
            fourth_moment_f = fourth_moment_f/n
            k_f = (n/(n-1))*fourth_moment_f/(var_f*var_f)
            k_matrix_f[j,k]=k_f
            k+=1
            i+=n
    t = []
    t = list(range(int(rows/n)))
    return k_matrix_f,t

def test_thr(n):
    Pfa = 10e-4
    T = 300
    noise_variance_thr = 2*kb*T*2e6*(10**(6))*4*50
    print("Theoretical noise variance = "+str(noise_variance_thr)+"\n")
    thr = np.sqrt((noise_variance_thr)*(np.log(1/Pfa)))
    lower_thr = 2-thr
    upper_thr = 2+thr
    return lower_thr, upper_thr

def exp_thr(var, pfa):
    exp_th_coef = -2*np.log(pfa)
    thr = list(map(lambda x: exp_th_coef*x, (var)))
    thr = np.sqrt(thr)
    return thr

def gaussian_thr(mean, std, pfa):
    return (mean + std*np.sqrt(2)*erfinv(2*(1-pfa)-1))

def abs_spectrogram(fs, win_length, x, buffer_size):
    f,t,Zxx = signal.stft(x, fs, window='hamming', nperseg=win_length/4)
    Sxx = np.abs(Zxx)**2
    return Sxx, f, t

def spectrogram(fs, win_length, x):
    f,t,Zxx = signal.stft(x, fs, window='hamming', nperseg=win_length, nfft=win_length, noverlap=win_length*0.5)
    return Zxx, f, t

def remove_dc(x, n, fs, n_bins):
    
    X = np.empty((int((len(x)/n)*(n-n_bins))), dtype=np.complex128)
    y = np.empty((int((len(x)/n)*(n-n_bins))), dtype=np.complex128)

    i=0
    j=0
    while(i<len(x)):
        X_fft = np.fft.fft(x[i:(i+n)])
        X_fft = X_fft[n_bins:]
        for xfft in X_fft:
            X[j] = xfft
            j+=1
        i += n

    i=0
    j=0
    while(i<len(X)):
        y_ifft = np.fft.ifft(X[i:(i+n-n_bins)])
        for yfft in y_ifft:
            y[j] = yfft
            j+=1
        i+=(n-n_bins)
    print(str(y[60000]))
    return y, (n-n_bins)

def sub_vector_mean(input_vect, sub_vect_length):
    result_vect=[]
    for i in range(0, len(input_vect), 10):
        sub_vect = input_vect[i:i+sub_vect_length]
        sub_vect_mean = np.mean(sub_vect)
        result_vect.append(sub_vect_mean)
    return result_vect

def interference_generator(fs, fc, length, num_pulses):
    t_end = (length/fs)/num_pulses
    t = np.arange(0, t_end,1/fs)
    x = 0.02*np.sin(2*np.pi*fc*t)
    x[0:int((fs*t_end/1.001))] = 0
    i = 0
    y = x
    while(i<(num_pulses-1)):
        y = np.concatenate((y, x))
        i+=1
    
    return y