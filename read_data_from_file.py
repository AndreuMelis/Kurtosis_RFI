import numpy as np
import itertools as IT
import struct
import os
import mmap

def read_data_from_file(file):

    samplesTotal = []
    # Per la RTL
    #signal = np.fromfile(file, dtype='int8', count=2*2046)
    # Per la Pluto
    signal = np.fromfile(file, dtype='int16', count=-1)
    real = signal[0::2]#agafa index 0 i corre 2
    imag = signal[1::2]
    for r, i in zip(real, imag):
        complx = r+1j*i
        samplesTotal.append(complx)

    return samplesTotal

def read_binary_file(filename, buffer_size):
    with open(filename, "rb") as f:
        
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
        byte_string = mm[:]
        num_samples = len(byte_string)//(2*4)
        num_samples = num_samples-num_samples%buffer_size
        samples = np.empty(num_samples, dtype=np.complex64)

        for i in range(num_samples):
            real, imag = struct.unpack("ff", byte_string[i*8:(i+1)*8])
            samples[i] = np.complex64(real + 1j * imag)
    mm.close()
    return samples