# from scipy.io import wavfile
# import matplotlib.pyplot as plt

import scipy
from scipy import signal
import numpy as np


def algorithm(x, fs):
    x = x / (2.0 ** (x.itemsize * 8 - 1))
    number_samples, number_channels = np.shape(x)

    # Window length, window function, and step length for the STFT
    window_length = 2 ** int(np.ceil(np.log2(0.04 * fs)))
    window_function = scipy.signal.windows.hamming(window_length, False)
    step_length = round(window_length / 2)

    cutoff_frequency = 100
    period_range = np.array([1, 10])

    # Number of time frames
    number_times = int(np.ceil((window_length - step_length + number_samples) / step_length))

    # Initialize the STFT
    audio_stft = np.zeros((window_length, number_times, number_channels), dtype=complex)

    # Loop over the channels
    for channel_index in range(0, number_channels):

        # STFT of the current channel
        audio_stft[:, :, channel_index] = stft_special(x[:, channel_index], window_function, step_length)

    # Magnitude spectrogram
    audio_spectrogram = abs(audio_stft[0:int(window_length / 2) + 1, :, :])

    # Beat spectrum of the spectrograms averaged over the channels (squared to emphasize peaks of periodicitiy)
    beat_spectrum = beatspectrum_special(np.power(np.mean(audio_spectrogram, axis=2), 2))

    # Period range in time frames for the beat spectrum
    period_range2 = np.round(period_range * fs / step_length).astype(int)

    # Repeating period in time frames given the period range
    repeating_period = periods_special(beat_spectrum, period_range2)

    # Cutoff frequency in frequency channels for the dual high-pass filter of the foreground
    cutoff_frequency2 = int(np.ceil(cutoff_frequency * (window_length - 1) / fs)) - 1

    # Initialize the background signal
    background_signal = np.zeros((number_samples, number_channels))

    # Loop over the channels
    for channel_index in range(0, number_channels):
        # Repeating mask for the current channel
        repeating_mask = mask_special(audio_spectrogram[:, :, channel_index], repeating_period)

        # High-pass filtering of the dual foreground
        repeating_mask[1:cutoff_frequency2 + 2, :] = 1

        # Mirror the frequency channels
        repeating_mask = np.concatenate((repeating_mask, repeating_mask[-2:0:-1, :]))

        # Estimated repeating background for the current channel
        background_signal1 = istft_special(repeating_mask * audio_stft[:, :, channel_index], window_function,
                                           step_length)

        # Truncate to the original number of samples
        background_signal[:, channel_index] = background_signal1[0:number_samples]

    return background_signal


def stft_special(x, window_function, step_length):
    """Short-time Fourier transform (STFT) (with zero-padding at the edges)"""

    # Number of samples and window length
    number_samples = len(x)
    window_length = len(window_function)

    # Number of time frames
    number_times = int(np.ceil((window_length - step_length + number_samples) / step_length))

    # Zero-padding at the start and end to center the windows
    x = np.pad(x, (window_length - step_length, number_times * step_length - number_samples),
               'constant', constant_values=0)

    # Initialize the STFT
    audio_stft = np.zeros((window_length, number_times))

    # Loop over the time frames
    for time_index in range(0, number_times):
        # Window the signal
        sample_index = step_length * time_index
        audio_stft[:, time_index] = x[sample_index:window_length + sample_index] * window_function

    # Fourier transform of the frames
    audio_stft = np.fft.fft(audio_stft, axis=0)

    return audio_stft


def acorr_special(data_matrix):
    """Autocorrelation using the Wiener–Khinchin theorem"""

    # Number of points in each column
    number_points = data_matrix.shape[0]

    # Power Spectral Density (PSD): PSD(X) = np.multiply(fft(X), conj(fft(X))) (after zero-padding for proper
    # autocorrelation)
    data_matrix = np.power(np.abs(np.fft.fft(data_matrix, n=2 * number_points, axis=0)), 2)

    # Wiener–Khinchin theorem: PSD(X) = np.fft.fft(repet._acorr(X))
    autocorrelation_matrix = np.real(np.fft.ifft(data_matrix, axis=0))

    # Discard the symmetric part
    autocorrelation_matrix = autocorrelation_matrix[0:number_points, :]

    # Unbiased autocorrelation (lag 0 to number_points-1)
    autocorrelation_matrix = (autocorrelation_matrix.T / np.arange(number_points, 0, -1)).T
    return autocorrelation_matrix


def beatspectrum_special(audio_spectrogram):
    """Beat spectrum using the autocorrelation"""

    # Autocorrelation of the frequency channels
    beat_spectrum = acorr_special(audio_spectrogram.T)

    # Mean over the frequency channels
    beat_spectrum = np.mean(beat_spectrum, axis=1)

    return beat_spectrum


def beatspectrogram_special(audio_spectrogram, segment_length, segment_step):
    """Beat spectrogram using the the beat spectrum"""

    # Number of frequency channels and time frames
    number_frequencies, number_times = np.shape(audio_spectrogram)

    # Zero-padding the audio spectrogram to center the segments
    audio_spectrogram = np.pad(audio_spectrogram,
                               ((0, 0),
                                (int(np.ceil((segment_length - 1) / 2)), int(np.floor((segment_length - 1) / 2)))),
                               'constant', constant_values=0)

    # Initialize beat spectrogram
    beat_spectrogram = np.zeros((segment_length, number_times))

    # Loop over the time frames (including the last one)
    for time_index in range(0, number_times, segment_step):
        # Beat spectrum of the centered audio spectrogram segment
        beat_spectrogram[:, time_index] = beatspectrum_special(audio_spectrogram[:,
                                                               time_index:time_index + segment_length])

        # Copy values in-between
        beat_spectrogram[:, time_index:min(time_index + segment_step - 1, number_times)] \
            = beat_spectrogram[:, time_index:time_index + 1]

    return beat_spectrogram


def periods_special(beat_spectra, period_range):
    """Repeating periods from the beat spectra (spectrum or spectrogram)"""

    # The repeating periods are the indices of the maxima in the beat spectra for the period range (they do not account
    # for lag 0 and should be shorter than a third of the length as at least three segments are needed for the median)
    if beat_spectra.ndim == 1:
        repeating_periods = np.argmax(
            beat_spectra[period_range[0]:min(period_range[1], int(np.floor(beat_spectra.shape[0] / 3)))]) + 1
    else:
        repeating_periods = np.argmax(
            beat_spectra[period_range[0]:min(period_range[1], int(np.floor(beat_spectra.shape[0] / 3))), :], axis=0) + 1

    # Re-adjust the index or indices
    repeating_periods = repeating_periods + period_range[0]

    return repeating_periods


def istft_special(audio_stft, window_function, step_length):
    """Inverse short-time Fourier transform (STFT)"""

    # Window length and number of time frames
    window_length, number_times = np.shape(audio_stft)

    # Number of samples for the signal
    number_samples = (number_times - 1) * step_length + window_length

    # Initialize the signal
    x = np.zeros(number_samples)

    # Inverse Fourier transform of the frames and real part to ensure real values
    audio_stft = np.real(np.fft.ifft(audio_stft, axis=0))

    # Loop over the time frames
    for time_index in range(0, number_times):
        # Constant overlap-add (if proper window and step)
        sample_index = step_length * time_index
        x[sample_index:window_length + sample_index] \
            = x[sample_index:window_length + sample_index] + audio_stft[:, time_index]

    # Remove the zero-padding at the start and end
    x = x[window_length - step_length:number_samples - (window_length - step_length)]

    # Un-apply window (just in case)
    x = x / sum(window_function[0:window_length:step_length])

    return x


def mask_special(audio_spectrogram, repeating_period):
    # Number of frequency channels and time frames
    number_frequencies, number_times = np.shape(audio_spectrogram)

    # Number of repeating segments, including the last partial one
    number_segments = int(np.ceil(number_times / repeating_period))

    # Pad the audio spectrogram to have an integer number of segments and reshape it to a tensor
    audio_spectrogram = np.pad(audio_spectrogram, ((0, 0), (0, number_segments * repeating_period - number_times)),
                               'constant', constant_values=np.inf)
    audio_spectrogram = np.reshape(audio_spectrogram,
                                   (number_frequencies, repeating_period, number_segments), order='F')

    # Derive the repeating segment by taking the median over the segments, ignoring the nan parts
    repeating_segment = np.concatenate((
        np.median(audio_spectrogram[:, 0:number_times - (number_segments - 1) * repeating_period, :], 2),
        np.median(audio_spectrogram[:, number_times - (number_segments - 1) * repeating_period:repeating_period,
                  0:number_segments - 1], 2)), 1)

    # Derive the repeating spectrogram by making sure it has less energy than the audio spectrogram
    repeating_spectrogram = np.minimum(audio_spectrogram, repeating_segment[:, :, np.newaxis])

    # Derive the repeating mask by normalizing the repeating spectrogram by the audio spectrogram
    repeating_mask = (repeating_spectrogram + np.finfo(float).eps) / (audio_spectrogram + np.finfo(float).eps)

    # Reshape the repeating mask and truncate to the original number of time frames
    repeating_mask = np.reshape(repeating_mask, (number_frequencies, number_segments * repeating_period), order='F')
    repeating_mask = repeating_mask[:, 0:number_times]

    return repeating_mask
