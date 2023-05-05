from scipy.signal import butter, lfilter
from scipy.fft import fft, fftfreq

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


def run():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz
    from scipy import signal
    from scipy.constants import k as kb

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 4.096e6
    f_low = 0.5e6
    f_high = 1.5e6

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9, 16, 32]:
        b, a = butter_bandpass(f_low, f_high, fs, order)
        w, h = freqz(b, a, worN=2000)
        mag = 20*np.log10(abs(h)) # convert magnitude to dB
        plt.plot((fs * 0.5 / np.pi) * w, mag, label="order = %d" % order)
        ax = plt.gca()
        ax.set_ylim([-150, 10])

    plt.plot([0, 0.5 * fs], [-3, -3], '--', label='-3 dB') # plot reference line at -3 dB

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.title('Butterworth Bandpass Filter Frequency Response')
    plt.legend()
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.show()
run()