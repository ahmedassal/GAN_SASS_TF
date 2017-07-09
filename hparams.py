'''
hyperparameters
'''

# Hyperparameters are in CAPS
# TODO use tf.app.flags to parse hyperparam from input
#      or consider use json file to store hyperparams
BATCH_SIZE = 16
FFT_SIZE = 1024
MAX_N_SIGNAL = 3
LR = 1e-4  # learn rate
EPS = 1e-7  # to prevent sqrt() log() etc cause NaN
DISIM_EMBED_DIMS = 32  # embedding dimensions for dissimilarity function
RELU_LEAKAGE = 0.3  # how leaky relu is, 0 -> relu, 1 -> linear
FLOATX = 'float32'  # default type for float


# automatically derived constants
# DO NOT CHANGE unless you know what you are doing
# <empty for now>

# assert regarding hyperparameters
assert FFT_SIZE&1 == 0
