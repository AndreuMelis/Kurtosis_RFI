class No_Rfi_Counter:
    def __init__(self, counter, kurt, mean):
        self.counter = counter
        self.kurt = kurt
        self.mean = mean

class Kurtosis:
    def __init__(self, kurtosis, upper_thr, lower_thr):
        self.kurtosis = kurtosis
        self.upper_thr = upper_thr
        self.lower_thr = lower_thr

class Signal_Stats:
    def __init__(self, circularity, variance, var_real, var_imag):
        self.circularity = circularity
        self.variance = variance
        self.var_real = var_real
        self.var_imag = var_imag