'''
TensorFlow Implementation of "GAN for Single Source Audio Separation"

TODO docs
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os

import numpy as np
import tensorflow as tf

# Hyperparameters are in CAPS
# TODO use tf.app.flags to parse hyperparam from input
#      or consider use json  file to store hyperparams
BATCH_SIZE = 16
FFT_SIZE = 1024
LR = 1e-4  # learn rate

class ModelModule(object):
    '''
    abstract class for a sub-module of model

    Args:
        model: Model instance
    '''
    def __init__(self, model):
        pass

    def __call__(self):
        pass

class Separator(ModelModule):
    '''
    voice separator

    feed mixed signal, output separated signals and maybe noise

    Args:
        n_signal: integer
            number of signals to output
    '''
    def __init__(self, model, n_signal):
        pass

    def __call__(self, s_input):
        '''
        Args:
            s_input: symbolic 3d tensor with shape [batch_size, lenght, fft_size]

        Returns
        '''
        pass

class Recognizer(ModelModule):
    '''
    speech recognizer, need clean voice as input
    '''
    def __init__(self, model):
        pass

    def __call__(self):
        pass

class Discriminator(ModelModule):
    '''
    feed audio signal, output scalar within [0., 1.]
    '''
    def __init__(self, model):
        pass

    def __call__(self):
        pass

class ToySeparator(Separator):
    '''
    minimal separator
    '''
    def __init__(self, model):
        pass

    def __call__(self):
        pass

class Model(object):
    '''
    base class for a full trainable model
    '''
    def __init__(self, name='BaseModel'):
        self.name = name

    def save(self, save_path, step):
        model_name = self.name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver()
        self.saver.save(self.sess,
                        os.path.join(save_path, model_name),
                        global_step=step)

    def load(self, save_path, model_file=None):
        if not os.path.exists(save_path):
            print('[!] Checkpoints path does not exist...')
            return False
        print('[*] Reading checkpoints...')
        if model_file is None:
            ckpt = tf.train.get_checkpoint_state(save_path)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                return False
        else:
            ckpt_name = model_file
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(save_path, ckpt_name))
        print('[*] Read {}'.format(ckpt_name))
        return True

    def build():
        # TODO
        pass

    def train():
        # TODO
        pass

if __name__ == '__main__':
    # TODO parse cmd args
    # TODO manage device
    # TODO build model
    # TODO train or inference
    # TODO write summary file
    pass
