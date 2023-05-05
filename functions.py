from scipy.stats import norm, kurtosis
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from read_data_from_file import *
from scipy.constants import k as kb
from scipy import signal
from scipy.special import erfinv
from scipy.signal import butter, lfilter
from memory_profiler import profile
from classes import *

def variance(x, n, mean_r, mean_i):
    var = ((x.real-mean_r)**2).sum() + ((x.imag-mean_i)**2).sum()
    return var/n

def circularity(x, covariance):
    var_real = np.var(x.real)
    var_imag = np.var(x.imag)
    circ = ((var_imag-var_real)**2+4*covariance**2)/((var_imag+var_real)**2)
    return circ, var_real, var_imag

def covariance(x):
    covariance = np.cov(x.real, x.imag, bias=True)[0][1]
    return covariance

#Time Complex Kurtosis
def complex_kurtosis(x, n):
    real_mean = x.real.mean()
    imag_mean = x.imag.mean()
    var = variance(x, n, real_mean, imag_mean)
    cov = covariance(x)
    circ, var_real, var_imag = circularity(x,cov)
    fourth_moment = (((x.real-real_mean)**2)+((x.imag-imag_mean)**2))**2
    fourth_moment = fourth_moment.mean()
    return (n/(n-1))*fourth_moment/(var*var), var, circ, var_real, var_imag

def time_kurtosis(samples, M, buffer_size, fs, gaus_thr, cont_no_rfi, time_file, i_sample):
    #Compute kurtosis from captured signal
    noise_variance = 0
    k_rfi = 0
    N_samples = int(M*buffer_size)#Number of samples per Kurtosis value
    N = int(len(samples) // N_samples)#Number of Kurtosis obtained
    k_vect = np.zeros(N, dtype=np.float64)
    circularity = np.zeros_like(k_vect)
    var = np.zeros_like(k_vect)
    var_real = np.zeros_like(k_vect)
    var_imag = np.zeros_like(k_vect)

    mask = np.ones(len(samples), dtype=np.complex64)    
    upper_thr = np.zeros(N, dtype=np.float64)
    lower_thr = np.zeros(N, dtype=np.float64)
    upper_thr[0] = cont_no_rfi.mean+gaus_thr
    lower_thr[0] = cont_no_rfi.mean-gaus_thr
  
    i = 0
    for i in range(0, len(samples), N_samples):

        k = i//N_samples
        kurt, noise_variance_, circ, var_real_, var_imag_ = complex_kurtosis(samples[i:(N_samples+i)], N_samples)
        k_vect[k] = kurt
        circularity[k] = circ
        var[k] = noise_variance_
        var_real[k] = var_real_
        var_imag[k] = var_imag_
        noise_variance += noise_variance_

        if(k>0):
            upper_thr[k] = cont_no_rfi.mean+gaus_thr
            lower_thr[k] = cont_no_rfi.mean-gaus_thr

        if(kurt > upper_thr[k] or kurt < lower_thr[k]):
            mask[i:(N_samples+i)] = 0j
            k_rfi += kurt
            time_file.write(str(i_sample/(fs))+'\n')

        else:
            # Every n = n_kurt kurtosis samples in a chunck which do not contain RFI, kurtosis mean is actualized
            cont_no_rfi.counter += 1
            cont_no_rfi.kurt += kurt
            n_kurt = 200
            if(cont_no_rfi.counter%n_kurt == 0):
                cont_no_rfi.mean = cont_no_rfi.kurt/cont_no_rfi.counter
                cont_no_rfi.kurt = 0
                cont_no_rfi.counter = 0
        i_sample = i_sample + N_samples
    noise_variance = noise_variance/(len(k_vect))
    return k_vect, mask, upper_thr, lower_thr, noise_variance, circularity, var, var_real, var_imag, cont_no_rfi, i_sample

def spectral_kurtosis(x, n):

    mean_kurtosis = 2
    var_kurtosis = 4/(2*6250) 
    gaus_thr = gaussian_thr(0, np.sqrt(var_kurtosis), 10e-4)
    upper_thr = mean_kurtosis+gaus_thr
    lower_thr = mean_kurtosis-gaus_thr

    rows, columns = np.shape(x)
    k_matrix_f = np.empty((rows, int(columns/n)))
    mask_matrix = np.empty((rows, int(columns/n)))
    rfi = np.empty((rows, columns), dtype = np.complex64)
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
            k_f = fourth_moment_f/(var_f*var_f)

            if(k_f > 2.5 or k_f < 1.5):
                mask = 0
                rfi[j,i:(i+n)] = 0

            else:
                mask = 1
                rfi[j,i:(i+n)] = f_sub_bin

            k_matrix_f[j,k]=k_f
            mask_matrix[j,k]=mask
            k+=1
            i+=n
    t = []
    t = list(range(int(columns/n)))
    return k_matrix_f,mask_matrix,t, rfi

def gaussian_thr(mean, std, pfa):
    return (mean + std*np.sqrt(2)*erfinv(2*(1-pfa)-1))

def abs_spectrogram(fs, win_length, x, overlap):
    f,t,Zxx = signal.stft(x, fs, window='hamming', nperseg=win_length, nfft=win_length, noverlap=win_length*overlap)
    Sxx = np.abs(Zxx)**2
    return Sxx, f, t

def spectrogram(fs, win_length, x, overlap):
    f,t,Zxx = signal.stft(x, fs, window='hamming', nperseg=win_length, nfft=win_length, noverlap=win_length*overlap)
    return Zxx, f, t

def ispectrogram(fs, Zxx, win_length, overlap):
    t, z = signal.istft(Zxx, fs, window='hamming', nperseg=win_length, nfft=win_length, noverlap=win_length*overlap )
    return t, z

def compute_power(signal, fftsize):
    
    power = np.empty(int(len(signal)/fftsize))
    power_i = 0
    i = 0
    k = 0
    while(i<len(signal)):
        aux = signal[i:(i+fftsize)]
        power_i = np.sum(aux*np.conj(aux))
        power[k] = power_i/fftsize
        k += 1
        i += fftsize
    return power

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
    w, h = signal.freqs(b, a)
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(100, color='green') # cutoff frequency
    plt.show()
    return y

def butter_highpass(f_cut, fs, order=5):
    nyq = 0.5 * fs
    cutoff = f_cut / nyq
    b, a = butter(order, cutoff, btype='high')
    return b, a

def butter_highpass_filter(data, f_cut, fs, order=5):
    b, a = butter_highpass(f_cut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(f_low, f_high, fs, order):
    nyq = 0.5 * fs
    low = f_low / nyq
    high = f_high / nyq
    b, a = butter(order, (low, high), btype='bandpass')
    return b, a

def butter_bandpass_filter(data, f_low, f_high, fs, order):
    b, a = butter_bandpass(f_low, f_high, fs, order)
    y = lfilter(b, a, data)
    return y