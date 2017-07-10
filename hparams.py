'''
hyperparameters
'''

# Hyperparameters are in CAPS
# TODO use tf.app.flags to parse hyperparam from input
#      or consider use json file to store hyperparams
BATCH_SIZE = 16
MAX_N_SIGNAL = 3
SIGNAL_LENGTH = 512  # length of spectrogram
MAX_TEXT_LENGTH = 32  # max length of text
FFT_SIZE = 256  # width of spectrogram
CHARSET_SIZE = 32  # size of character set, including "end" character
LR = 1e-4  # learn rate
EPS = 1e-7  # to prevent sqrt() log() etc cause NaN
DISIM_EMBED_DIMS = 32  # embedding dimensions for dissimilarity function
RELU_LEAKAGE = 0.3  # how leaky relu is, 0 -> relu, 1 -> linear
USE_TEXT = False  # whether to integrate speech recognizer into GAN training
FLOATX = 'float32'  # default type for float
INTX = 'int32'  # defualt type for int
EXTRACTOR_TYPE = 'toy'
SEPARATOR_TYPE = 'toy'
RECOGNIZER_TYPE = 'toy'
DISCRIMINATOR_TYPE = 'toy'


# normally you don't need touch anything below if you just want to tweak
# some hyperparameters

# registry
extractor_registry = {}
separator_registry = {}
recognizer_registry = {}
discriminator_registry = {}

# decorators & getters
def register_extractor(name):
    def wrapper(cls):
        extractor_registry[name] = cls
        return cls
    return wrapper


def get_extractor():
    return extractor_registry[EXTRACTOR_TYPE]


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
