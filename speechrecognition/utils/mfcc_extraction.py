import numpy as np
import scipy.signal
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plot
from utils.audio_utils import get_signal


"""
Based on http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
"""


def framing(signal, sample_rate, frame_size=0.025, frame_stride=0.01):
    """
    Splits wav data into time frames. if frame_size is 0.025, that means that it will trim by 25 milliseconds
    frame_stride of 0.01 is a 10 ms stride with 15 ms overlap.
    The indices are the time indices at every point.
    It also applies a hamming window.
    :param signal:
    :param sample_rate:
    :param frame_size:
    :param frame_stride:
    :return: (frames, indices)
    """
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal,
                           z)  # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    frames *= np.hamming(frame_length)  # Hamming Window
    return frames, indices


def fourier_transform(frames, nfft=512):
    """
    FFT transformation
    P=|FFT(x_i)|^2/N
    N is usually 256 or 512
    :param frames:
    :param nfft:
    :return:
    """
    return np.absolute(np.fft.rfft(frames, nfft))  # Magnitude of the FFT


def power_spectrum(fft_frames, nfft=512):
    """
    Periodograms?
    :param fft_frames:
    :param nfft:
    :return:
    """
    return (1.0 / nfft) * (fft_frames ** 2)  # Power Spectrum


def filter_banks(power_spec_frames, sample_rate, nfft=512, nfilters=40, low_freq_mel=0):
    """
    We can convert between Hertz (f) and Mel (m) using the following equations:
    m=2595log_10(1+(f/700))
    f=700(10^(m/2595)-1)
    H_m(K) = {
        if k < f(m-1)
            0
        elif f(m-1)<=k<f(m)
            (k-f(m-1)) / (f(m)-f(m-1))
        elif k == f(m)
            1
        elif f(m) < k <= f(m+1)
            (f(m+1)-k) / (f(m+1)-f(m))
        elif k > f(m-1)
            0
    }
    :param power_spec_frames:
    :param sample_rate:
    :param nfft:
    :param nfilters:
    :param low_freq_mel:
    :return:
    """
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilters + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((nfft + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilters, int(np.floor(nfft / 2 + 1))))
    for m in range(1, nfilters + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    fb = np.dot(power_spec_frames, fbank.T)
    fb = np.where(fb == 0, np.finfo(float).eps, fb)  # Numerical Stability
    return 20 * np.log10(fb)  # dB


def mfcc(filter_banks, num_ceps=12, cep_lifter=22, lift=True):
    """
    Returns the mfcc for hte filter banks.
    lift does sinusoidal liftering which reduces noise

    :param filter_banks:
    :param num_ceps:
    :param cep_lifter:
    :param lift:
    :return:
    """
    mfccs = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13
    if lift:
        (nframes, ncoeff) = mfccs.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfccs *= lift  # *
    return mfccs


def mean_normalize(frames):
    """
    TODO Look at exactly how this is normalizing!!!!
    Normalizes by subtracting the mean from each value.
    :param frames:
    :return:
    """
    frames -= (np.mean(frames, axis=0) + 1e-8)
    return frames


def mfcc_indices(indices):
    """
    Returns adjusted indices for the filter banks or mfccs. Takes the first index of each row.
    :param indices:
    :return:
    """
    return np.array([i[0] for i in indices])


def filter_bank_features(signal, sample_rate, emphasize=True, normalize=True, frame_size=0.025):
    """
    Returns the filter banks from an audio file
    :param wavpath:
    :param emphasize:
    :param normalize:
    :param frame_size:
    :return:
    """
    frames, indices = framing(signal, sample_rate, frame_size=frame_size)
    frames = fourier_transform(frames)
    frames = power_spectrum(frames)
    frames = filter_banks(frames, sample_rate)
    if normalize:
        frames = mean_normalize(frames)
    return frames.astype(np.float32), mfcc_indices(indices)


def mfcc_features(signal, sample_rate, emphasize=True, lift=True, normalize=True, frame_size=0.025):
    """
    Returns the mfcc features from an audio file.
    :param wavpath:
    :param emphasize:
    :param lift:
    :param normalize:
    :param frame_size:
    :return:
    """
    frames, indices = filter_bank_features(signal, sample_rate, emphasize=emphasize, normalize=False, frame_size=frame_size)
    frames = mfcc(frames, lift=lift)
    if normalize:
        frames = mean_normalize(frames)
    return frames.astype(np.float32), indices






# if __name__ == '__main__':
#     wavpath = 'C:/Users/ryanc/Documents/kiel_corpus/DLME001.wav'
#
#     # signal, sample_rate = get_signal(wavpath, emphasize=True)
#
#     fb, _ = filter_bank_features(wavpath)
#
#     specs = spectrogram(wavpath)
#
#     mfccs, _ = mfcc_features(wavpath)
#
#     plot.title('Spectrogram and mfcc of a wav file')
#     p1 = plot.subplot(311)
#     p1.set_title('Spectrogram')
#     p1.imshow(specs.T, interpolation='nearest', origin='lower', extent=None, aspect='auto')
#
#     p2 = plot.subplot(312)
#     p2.set_title('Filter Banks')
#     p2.imshow(fb.T, interpolation='nearest', origin='lower', extent=None, aspect='auto')
#
#     p4 = plot.subplot(313)
#     p4.set_title('MFCC')
#     p4.imshow(mfccs.T, interpolation='nearest', origin='lower', extent=None, aspect='auto')
#
#     plot.tight_layout()
#
#     plot.show()



