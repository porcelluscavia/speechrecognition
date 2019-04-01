import numpy as np
import scipy.signal
from utils.audio_utils import get_signal
import matplotlib.pyplot as plot


def spectrogram(wav_data, sample_rate, window_size=25, step_size=10, eps=1e-10):
    """
    TODO add indices
    :param wav_data:
    :param sample_rate:
    :param window_size:
    :param step_size:
    :param eps:
    :return:
    """
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))

    sample_freq, segment_times, spec = scipy.signal.spectrogram(wav_data, fs=sample_rate, window='hann',
                                                                nperseg=nperseg, noverlap=noverlap, detrend=False)
    return np.log(spec.astype(np.float32) + eps), segment_times, sample_freq


def pad_matrix(matrix, img_size):
    new_matrix = np.zeros(img_size).astype('float32')
    if matrix.shape[0] == 0:
        return new_matrix
    elif len(matrix.shape) < 2:
        x_size = matrix.shape[0]
        y_size = 1
    else:
        x_size, y_size = matrix.shape

    if x_size >= img_size[0]:
        x_size = img_size[0]
    if y_size >= img_size[1]:
        y_size = img_size[1]
    new_matrix[:x_size, :y_size] = matrix[:x_size, :y_size]
    return new_matrix

#
# wav_path = 'C:/Users/ryanc/Documents/kiel_corpus/DLME001.wav'
#
# wav_data, sample_rate = get_signal(wavpath=wav_path)
#
# s, t, f = spectrogram(wav_data, sample_rate)
# x = 100
# y = 100
# plot.title('Spectrogram and mfcc of a wav file')
# p1 = plot.subplot(111)
# p1.set_title('Spectrogram')
# p1.imshow(pad_matrix(s, (x,y)), interpolation='nearest', origin='lower', extent=None, aspect='auto')
# plot.show()
