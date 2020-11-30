# from scipy.fft import fftshift
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy import signal

import scipy
from scipy.io import wavfile
import functiondata

# Audio signal (normalized) and sample rate in Hz
fs, x = wavfile.read('audio_file.wav')
x = x / (2.0**(x.itemsize*8-1))

# Estimate the background signal and infer the foreground signal
background_signal = functiondata.algorithm(x, fs)
foreground_signal = x-background_signal

# Write the background and foreground signals (un-normalized)
scipy.io.wavfile.write('background_signaler.wav', fs, background_signal)
scipy.io.wavfile.write('foreground_signaler.wav', fs, foreground_signal)