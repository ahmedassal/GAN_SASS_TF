import os
import string
import gzip
from sys import stdout
from six.moves import cPickle as pickle

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal

# TODO add license from tensorpack/examples/CTC-TIMIT

# TODO merge these with hparams.py file
FFT_SIZE = 256
EPS = 1e-7

# "$" means end of text/phoneme
CHARSET = set(string.ascii_lowercase + ' $')
PHONEME_LIST = (
    '$_aa_ae_ah_ao_aw_ax_ax-h_axr_ay_b_bcl_ch_d_dcl_dh_'
    'dx_eh_el_em_en_eng_epi_er_ey_f_g_gcl_h#_hh_hv_ih_'
    'ix_iy_jh_k_kcl_l_m_n_ng_nx_ow_oy_p_pau_pcl_q_r_'
    's_sh_t_tcl_th_uh_uw_ux_v_w_y_z_zh').split('_')

PHONEME_DIC = {v: k for k, v in enumerate(PHONEME_LIST)}
WORD_DIC = {v: k for k, v in enumerate(CHARSET)}

INTX = 'int32'
FLOATX = 'float32'


def read_timit_txt(f):
    line = f.readlines()[0].strip().split(' ')[2:]
    line = ' '.join(line).replace('.', '').lower()
    line += '$'
    return np.asarray([WORD_DIC[c] for c in line if c in CHARSET], dtype=INTX)


def read_timit_phoneme(f):
    pho = []
    for line in f:
        line = line.strip().split(' ')[-1]
        pho.append(PHONEME_DIC[line])
    pho.append(PHONEME_LIST[0])
    return np.asarray(pho)


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
    scale = np.log1p(mags) / (mags + EPS)
    log_spec = scale * freqs
    log_spec[0].imag = log_spec[-1].real
    log_spec = log_spec[:-1]
    return np.concatenate([log_spec.real, log_spec.imag]).T.astype(FLOATX)


def feature_to_spectrum(features):
    '''reverse of spectrum_to_feature'''
    features = features.T
    features = np.complex(features[:FFT_SIZE//2], features[FFT_SIZE//2:])
    features = np.pad(features, [(0, 1), (0, 0)], mode='constant')
    features[-1].real = features[0].imag
    features[0].imag = 0.
    mags = np.abs(features)
    features = (np.expm1(mags) / (mags + EPS)) * features
    return features


TRAIN_DIR = './train'
TEST_DIR = './test'
train_files = os.listdir('./train')
test_files = os.listdir('./test')

train_signals = []
train_phonemes = []
train_texts = []
os.chdir(TRAIN_DIR)
for fname in train_files:
    if not fname.endswith('.wav'):
        continue
    if fname.startswith('sa'):
        continue
    fm, waveform = wavfile.read(fname)
    if fm != 16000:
        raise ValueError('Sampling rate must be 16k')
    Zxx = signal.stft(waveform, nperseg=FFT_SIZE)[2]
    stft_feature = spectrum_to_feature(Zxx)
    with open(fname.upper().replace('.WAV', '.TXT'), 'r') as f:
        text = read_timit_txt(f)

    with open(fname.upper().replace('.WAV', '.PHN'), 'r') as f:
        phoneme = read_timit_phoneme(f)

    train_signals.append(stft_feature)
    train_texts.append(text)
    train_phonemes.append(phoneme)

    stdout.write('.')
    stdout.flush()
os.chdir('../')

test_signals = []
test_phonemes = []
test_texts = []
os.chdir(TEST_DIR)
for fname in test_files:
    if not fname.endswith('.wav'):
        continue
    if fname.startswith('sa'):
        continue
    fm, waveform = wavfile.read(fname)
    if fm != 16000:
        raise ValueError('Sampling rate must be 16k')
    _, __, Zxx = signal.stft(waveform, nperseg=FFT_SIZE)
    stft_feature = spectrum_to_feature(Zxx)
    with open(fname.upper().replace('.WAV', '.TXT')) as f:
        text = read_timit_txt(f)

    with open(fname.upper().replace('.WAV', '.PHN')) as f:
        phoneme = read_timit_phoneme(f)

    test_signals.append(stft_feature)
    test_texts.append(text)
    test_phonemes.append(phoneme)

    stdout.write('.')
    stdout.flush()
os.chdir('../')

# sort the whole batch by length, so minibatches
# need less zero padding -> higher performance
train_order = np.argsort(list(map(len, train_signals)))
test_order = np.argsort(list(map(len, test_signals)))

train_signals = [train_signals[i] for i in train_order]
test_signals = [test_signals[i] for i in test_order]
train_phonemes = [train_phonemes[i] for i in train_order]
test_phonemes = [test_phonemes[i] for i in test_order]
train_texts = [train_texts[i] for i in train_order]
test_texts = [test_texts[i] for i in test_order]

with open('train_set.pkl', 'wb') as f:
    pickle.dump(train_signals, f)
    pickle.dump(train_phonemes, f)
    pickle.dump(train_texts, f)

with open('test_set.pkl', 'wb') as f:
    pickle.dump(test_signals, f)
    pickle.dump(test_phonemes, f)
    pickle.dump(test_texts, f)

stdout.write('\nFinished preprocessing\n')
