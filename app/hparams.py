'''
hyperparameters
'''

# Hyperparameters are in CAPS
# TODO use tf.app.flags to parse hyperparam from input
#      or consider use json file to store hyperparams
BATCH_SIZE = 2  # minibatch size
MAX_N_SIGNAL = 3
FFT_SIZE = 256  # width of spectrogram
CHARSET_SIZE = 32  # size of character set, including "end" character
EPS = 1e-7  # to prevent sqrt() log() etc cause NaN
RELU_LEAKAGE = 0.3  # how leaky relu is, 0 -> relu, 1 -> linear
USE_TEXT = False  # whether to integrate speech recognizer into GAN training
FLOATX = 'float32'  # default type for float
INTX = 'int32'  # defualt type for int

EXTRACTOR_TYPE = 'toy'
SEPARATOR_TYPE = 'bilstm-v1'
RECOGNIZER_TYPE = 'toy'
DISCRIMINATOR_TYPE = 'toy'

# TODO add ADAM and other ozers
OPTIMIZER_TYPE = 'sgd'
LR = 1e-4  # learn rate
LR_DECAY = None

DATASET_TYPE = 'toy'

SUMMARY_DIR = './logs'

CLS_REAL_SIGNAL = 0
CLS_REAL_NOISE = 1
CLS_FAKE_SIGNAL = 2


# normally you don't need touch anything below if you just want to tweak
# some hyperparameters

# registry
extractor_registry = {}
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
