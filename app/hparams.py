'''
hyperparameters
'''

# Hyperparameters are in CAPS
# TODO use tf.app.flags to parse hyperparam from input
#      or consider use json file to store hyperparams
BATCH_SIZE = 8  # minibatch size
MAX_N_SIGNAL = 3

# width of spectrogram
# redo dataset preprocessing if you change this
FFT_SIZE = 256

# size of character set, excluding "blank" character
# redo dataset preprocessing if you change this
CHARSET_SIZE = 27

FLOATX = 'float32'  # default type for float
INTX = 'int32'  # defualt type for int

RELU_LEAKAGE = 0.3  # how leaky relu is, 0 -> relu, 1 -> linear
EPS = 1e-7  # to prevent sqrt() log() etc cause NaN
DROPOUT_KEEP_PROB = 0.8  # probability to keep in dropout layer
REG_SCALE = 1e-2  # regularization loss scale
REG_TYPE = 'L2'  # regularization type

# check "modules.py" to see available sub-modules
USE_ASR = False  # whether to integrate speech recognizer into GAN training
SEPARATOR_TYPE = 'bilstm-v1'
RECOGNIZER_TYPE = 'bilstm-ctc-v1'
DISCRIMINATOR_TYPE = 'bilstm-3way-v1'
OPTIMIZER_TYPE = 'adam'  # "sgd" or "adam"
LR = 1e-5  # learn rate
LR_DECAY = None

DATASET_TYPE = 'timit'  # "toy" or "timit"

# "greedy" or "beam", beam is slower but gives better result
CTC_DECODER_TYPE = 'greedy'

SUMMARY_DIR = './logs'
ASR_SUMMARY_DIR = './asr_logs'


# ==========================================================================
# normally you don't need touch anything below if you just want to tweak
# some hyperparameters
CLS_REAL_SIGNAL = 0
CLS_REAL_NOISE = 1
CLS_FAKE_SIGNAL = 2

assert isinstance(DROPOUT_KEEP_PROB, float)
assert 0. < DROPOUT_KEEP_PROB <= 1.
assert isinstance(LR, float) and LR >= 0.

import tensorflow as tf

# registry
separator_registry = {}
recognizer_registry = {}
discriminator_registry = {}
ozer_registry = {}
dataset_registry = {}


# decorators & getters
def register_separator(name):
    def wrapper(cls):
        separator_registry[name] = cls
        return cls
    return wrapper


def get_separator():
    return separator_registry[SEPARATOR_TYPE]


def register_recognizer(name):
    def wrapper(cls):
        recognizer_registry[name] = cls
        return cls
    return wrapper


def get_recognizer():
    return recognizer_registry[RECOGNIZER_TYPE]

def register_discriminator(name):
    def wrapper(cls):
        discriminator_registry[name] = cls
        return cls
    return wrapper


def get_discriminator():
    return discriminator_registry[DISCRIMINATOR_TYPE]


def register_optimizer(name):
    def wrapper(fn):
        ozer_registry[name] = fn
        return fn
    return wrapper


def get_optimizer():
    return ozer_registry[OPTIMIZER_TYPE]


def register_dataset(name):
    def wrapper(fn):
        dataset_registry[name] = fn
        return fn
    return wrapper


def get_dataset():
    return dataset_registry[DATASET_TYPE]


def get_regularizer():
    reger = dict(
        L1=tf.contrib.layers.l1_regularizer,
        L2=tf.contrib.layers.l2_regularizer)[REG_TYPE](REG_SCALE)
    return reger
