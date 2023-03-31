from scipy.stats import norm, kurtosis
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from read_data_from_file import *
from scipy.constants import k as kb
from scipy import signal
from functions import *

fs = 2e6
buffer_size = 4096
N_buffer = 1000
fftsize = 32

noise = np.random.normal(0,np.sqrt(kb*300*2e6*10**(6)*4*50),2*buffer_size*N_buffer).view(np.complex128)

Zxx, f_s, t_s = spectrogram(fs, fftsize, noise)
#spectral_k, t_sk = spectral_kurtosis(Zxx, 4*10

fig = plt.figure(2)
ax = fig.gca(projection='3d')
t_mesh, f_mesh = np.meshgrid(t_s, f_s)
ax.plot_surface(t_mesh, f_mesh, Zxx, cmap='jet')
ax.set_xlabel('Time [sec]')
ax.set_ylabel('Frequency [Hz]')
ax.set_zlabel('Magnitude')

"""
plt.figure(3)
plt.imshow(spectral_k)
plt.colorbar()

fig = plt.figure(4)
ax = fig.gca(projection='3d')
t_mesh, f_mesh = np.meshgrid(t_sk, f_s)
ax.plot_surface(t_sk, f_mesh, spectral_k, cmap='jet')
ax.set_xlabel('Time [sec]')
ax.set_ylabel('Frequency [Hz]')
ax.set_zlabel('Magnitude')
"""
plt.show()