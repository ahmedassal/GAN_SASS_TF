import os
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import tensorflow as tf

# TODO merge these with hparams.py file
TFREC_FNAME = 'timit.tfrecords'
FFT_SIZE = 256
EPS = 1e-7


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def spectrum_to_feature(freqs):
    '''
    Convert STFT spectrum to feature vector

    The goal is to keep feature vector length as FFT_SIZE,
        this would be typically power of 2, yielding improved
        performance on GPU.

    This does it by:

    1. change magnitude M -> log(1+M)

    2. since first and last value will always be real
        we drop useless ones to make signal FFT_SIZE

    Args:
        freqs: [FFT_SIZE/2+1, LEN] complex valued matrix

    Returns:
        [LEN, FFT_SIZE] real valued matrix
    '''
    mags = np.abs(freqs)
    scale = np.log1p(mags) / (mags + EPS)
    log_spec = mags * scale
    log_spec[0].imag = log_spec[-1].real
    log_spec = log_spec[:-1]
    return np.concatenate([log_spec.real, log_spec.imag]).T


def feature_to_spectrum(features):
    features = features.T
    features = np.complex(features[:FFT_SIZE//2], features[FFT_SIZE//2:])
    features = np.pad(features, [(0, 1), (0, 0)], mode='constant')
    features[-1].real = features[0].imag
    features[0].imag = 0.
    mags = np.abs(features)
    features = np.expm1(mags) / (mags + EPS) * features
    return features


train_files = os.listdir('./train')
test_files = os.listdir('./test')

writer = tf.python_io.TFRecordWriter(TFREC_FNAME)

for fname in train_files:
    if not fname.endswith('.wav'):
        continue
    if fname.startswith('sx'):
        continue
    fm, waveform = wavfile.read(fname)
    if fm != 16000:
        raise ValueError('Sampling rate must be 16k')
    _, __, Zxx = signal.stft(waveform, nperseg=FFT_SIZE)
    feature = spectrum_to_feature(Zxx)
    with open(fname.upper().replace('.WAV', '.TXT')) as f:
        text = f.read()
