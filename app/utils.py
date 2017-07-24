import nltk
import numpy as np
from itertools import product

import app.hparams as hparams


def spectrum_to_feature(freqs):
    '''
    Convert STFT spectrum to feature vector

    The goal is to keep real valued feature vector length
        same as FFT_SIZE. This would be typically power
        of 2, yielding improved performance on GPU.

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
    scale = np.log1p(mags) / (mags + hparams.EPS)
    log_spec = scale * freqs
    log_spec[0].imag = log_spec[-1].real
    log_spec = log_spec[:-1]
    return np.concatenate(
        [log_spec.real, log_spec.imag]).T.astype(hparams.FLOATX)


def feature_to_spectrum(features):
    '''reverse of spectrum_to_feature'''
    fft_size = hparams.FFT_SIZE
    features = features.T
    features = features[:fft_size//2] + features[fft_size//2:]*1.j
    features = np.pad(features, [(0, 1), (0, 0)], mode='constant')
    features[-1].real = features[0].imag
    features[0].imag = 0.
    mags = np.abs(features)
    features = (np.expm1(mags) / (mags + hparams.EPS)) * features
    return features


def batch_levenshtein(x, y):
    '''
    Batched version of nltk.edit_distance, over character
    This performs edit distance over last axis, trimming trailing zeros.

    Args:
        x: int array, hypothesis
        y: int array, target

    Returns: int32 array
    '''
    x_shp = x.shape
    y_shp = y.shape
    assert x_shp[:-1] == y_shp[:-1]
    idx_iter = product(*map(range, x_shp[:-1]))

    z = np.empty(x_shp[:-1], dtype='int32')
    for idx in idx_iter:
        u, v = x[idx], y[idx]
        u = np.trim_zeros(u, 'b')
        v = np.trim_zeros(v, 'b')
        z[idx] = nltk.edit_distance(u, v)
    return z


def batch_wer(x, y, fn_decoder):
    '''
    Batched version of nltk.edit_distance, over words
    This performs edit distance over last axis, trimming trailing zeros.

    Args:
        x: int array, hypothesis
        y: int array, target
        fn_decoder: function to convert int vector into string
    Returns: int32 array
    '''
    x_shp = x.shape
    y_shp = y.shape
    assert x_shp[:-1] == y_shp[:-1]
    idx_iter = product(*map(range, x_shp[:-1]))

    z = np.empty(x_shp[-1], dtype='int32')
    for idx in idx_iter:
        x_str = fn_decoder(x[idx]).strip(' $').split(' ')
        y_str = fn_decoder(y[idx]).strip(' $').split(' ')
        z[idx] = nltk.edit_distance(x_str, y_str)
    return z


