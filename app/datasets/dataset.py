from itertools import product

import numpy as np

import app.hparams as hparams


class Dataset(object):
    def __init__(self):
        self.is_loaded = False

    def epoch(self, subset, batch_size):
        '''
        Iterator, yields batches of numpy array
        Args:
            subset: string
            batch_size: int

        Yields:
            (signals, text_indices, text_values, text_shape)

            signals is rank-3 float32 array: [batch_size, time, features]
            text_values is sparse rank-2 int32 array: [batch_size, time]
            text_shape is 2D vector
        '''
        raise NotImplementedError()

    def install_and_load(self):
        '''
        Download and preprocess dataset and store it on local disk.
        This method should check whether data is available in ./datasets,
        or already downloaded in ./downloads

        Raises:
            raise RuntimeError is something fails
        '''
        raise NotImplementedError()

    def encode_from_str(arr):
        raise NotImplementedError()

    def decode_to_str(arr):
        raise NotImplementedError()

@hparams.register_dataset('toy')
class WhiteNoiseData(object):
    '''
    this always generates uniform noise

    for signal, it's of length 128
    for text, it's of length 64
    '''
    # make it more general, support more shape
    def __init__(self):
        self.is_loaded = False

    def epoch(self, subset, batch_size):
        if not self.is_loaded:
            raise RuntimeError('Dataset is not loaded.')
        for _ in range(10):
            signal = np.random.rand(
                batch_size, 128, hparams.FFT_SIZE)
            text_indices = np.asarray(
                list(product(range(batch_size), range(64))),
                dtype=hparams.INTX)
            text_values = np.asarray(np.random.randint(
                0, hparams.CHARSET_SIZE-1,
                (batch_size, 64), dtype=hparams.INTX).flat)
            text_shape = (batch_size, 64)
            yield signal, (text_indices, text_values, text_shape)

    def install_and_load(self):
        self.is_loaded = True
        return
