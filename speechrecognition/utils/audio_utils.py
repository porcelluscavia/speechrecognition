import numpy as np
import scipy.signal
import scipy.io.wavfile
import librosa

"""
Based on http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
"""


def get_signal(wavpath, emphasize=True, pre_emphasis=0.97):
    """
    Gets signal and sample rate from wav file.
    If emphasize is True https://www.quora.com/Why-is-pre-emphasis-i-e-passing-the-speech-signal-through-a-first-order-high-pass-filter-required-in-speech-processing-and-how-does-it-work/answer/Nickolay-Shmyrev?srid=e4nz&share=71ca3e28
    :param wavpath:
    :param emphasize:
    :param pre_emphasis:
    :return: (signal, sample_rate)
    """
    sample_rate, signal = scipy.io.wavfile.read(wavpath)  # File assumed to be in the same directory
    if emphasize:
        return np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1]), sample_rate
    return signal, sample_rate



def get_signal_librosa(wavpath):
    """
    Gets signal and sample rate from wav file.
    If emphasize is True https://www.quora.com/Why-is-pre-emphasis-i-e-passing-the-speech-signal-through-a-first-order-high-pass-filter-required-in-speech-processing-and-how-does-it-work/answer/Nickolay-Shmyrev?srid=e4nz&share=71ca3e28
    :param wavpath:
    :param emphasize:
    :param pre_emphasis:
    :return: (signal, sample_rate)
    """
    signal, sample_rate = librosa.load(wavpath, sr=None)
    return signal, sample_rate




