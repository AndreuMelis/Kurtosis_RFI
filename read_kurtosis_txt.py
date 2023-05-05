import numpy as np
import matplotlib.pyplot as plt
from functions import *

k_fir = np.loadtxt('kurtosis_fir_mix.txt', dtype=np.float64)
k_nofir = np.loadtxt('kurtosis_mix.txt', dtype=np.float64)
buffer_size = 4096
fs = 4.096e6
Trad = 100e-3
buffer_size = 4096
N_kurt_Trad = 4 # number of kurtosis values per Trad
M = ((Trad*fs)/buffer_size)/N_kurt_Trad # number of buffers integrated per kurtosis value
mean_kurt_no_rfi = 2
var_kurtosis = 4/(M*buffer_size) 
gaus_thr = gaussian_thr(0, np.sqrt(var_kurtosis), 10e-4)

if(len(k_fir) > len(k_nofir)):
    k_fir = k_fir[0:len(k_nofir)]
else:
    k_nofir = k_nofir[0:len(k_fir)]

k_fir = k_fir[0:int(len(k_fir)/1.35)]
k_nofir = k_nofir[0:int(len(k_nofir)/1.35)]

t = np.arange(0, len(k_fir))
plt.plot(t, k_fir)
plt.plot(t, np.full(t.shape, np.mean(k_fir)))
plt.plot(t, k_nofir)
plt.plot(t, np.full(t.shape, np.mean(k_nofir)))
plt.plot(t,np.full(t.shape, 2-gaus_thr))
plt.plot(t,np.full(t.shape, 2+gaus_thr))
plt.legend(['Kurtosis FIR enabled','Mean Kurtosis FIR enabled', 'Kurtosis FIR disabled', 'Mean Kurtosis FIR disabled', 'Lower Threshold', 'Upper Threshold'])
plt.title("Complex Kurtosis")
plt.ylabel("Kurtosis")
plt.xlabel("Sample per integrated buffer")
plt.savefig("/home/andreu/Escriptori/Kurtosis Time/FIR_manual_matlab/kurtosis_mix.png")
plt.show()