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

import hparams, ops


# Global vars
g_sess = tf.Session()


class ModelModule(object):
    '''
    abstract class for a sub-module of model

    Args:
        model: Model instance
    '''
    def __init__(self, model, name):
        pass

    def __call__(self):
        raise NotImplementedError()


class Extractor(ModelModule):
    '''
    voice extractor

    Feed mixed signal, output extracted signals and maybe noise

    Args:
        n_signal: integer
            Number of signals to separate out.
    '''
    def __init__(self, model, name, n_signal):
        self.name = name
        self.n_signal = n_signal

    def __call__(self, s_input):
        '''
        Args:
            s_input: symbolic 3d tensor with shape
            [batch_size, length, fft_size]

        Returns:
            symbolic 4d tensor with shape
            [batch_size, n_signal, length, fft_size]
        '''
        raise NotImplementedError()


class Separator(ModelModule):
    '''
    This module should implement dissimilarity function.
    '''
    def __init__(self, model, name):
        self.name = name

    @classmethod
    def _get_embedding(cls, s_signals):
        '''
        Args:
            s_signals: tensor variable
                [batch_size, n_signal, length, fft_size]

        Returns:
            tensor variable of shape:
                [batch_size, n_signal, embedding_dims]
        '''
        raise NotImplementedError()

    @staticmethod
    def _embedding_to_dismat(s_embeds):
        '''
        converts embeddings to dissimilarity matrix

        Args:
            s_embeds: tensor variable
                [batch_size, n_signal, embedding_dims]

        Returns:
            batched dissimilarity matrix
        '''
        # we compute outer product of norms
        # because n_signal is small, so this would be faster

        # TODO exploit tf.rsqrt for speedup ?
        s_norms = tf.norm(s_embeds, axis=-1)  # [batch_size, n_signal]
        s_norms_outer = tf.matmul(
            tf.expand_dims(s_norms, axis=2),
            tf.expand_dims(s_norms, axis=1))
        s_cosines = tf.matmul(s_embeds, tf.transpose(s_embeds, [0, 2, 1]))
        return - s_cosines / (s_norms_outer + hparams.EPS)

    def __call__(self, s_signals):
        '''
        Args:
            s_signals: tensor variable
                4d tensor of shape [batch_size, n_signal, length, fft_size]
                This tensor is expected to be input from extractor

        Returns:
            batched dissimilarity matrices [batch_size, n_signal, n_signal]
        '''
        with tf.name_scope(self.name, reuse=True):
            s_embeds = self._get_embedding(s_signals)
            s_dismat = Separator._embedding_to_dismat(s_embeds)
        return s_dismat


class Recognizer(ModelModule):
    '''
    speech recognizer, need clean voice as input
    '''
    def __init__(self, model, name):
        pass

    def __call__(self):
        pass


class Discriminator(ModelModule):
    '''
    feed audio signal, output scalar within [0., 1.]
    '''
    use_text = True  # whether this discriminator takes text as input
    def __init__(self, model):
        pass

    def __call__(self, s_signals, s_texts):
        '''TODO docs'''
        pass


class ToyExtractor(Extractor):
    '''
    Minimal separator for debugging purposes

    This simply pass each FFT spectrum as a data point
    into a 3 layer MLP
    '''
    def __init__(self, model, name, n_signal):
        self.n_signal = n_signal
        self.name = name

    def __call__(self, s_input):
        inp_shape = s_input.get_shape().as_list()
        out_shape = inp_shape.copy()
        out_shape.insert(self.n_signal, 1)
        fft_size = inp_shape[-1]
        with tf.variable_scope(self.name):
            s_mid = ops.lyr_linear(
                'linear0', s_input, fft_size*2, axis=-1)
            s_mid = ops.relu(s_mid, hparams.RELU_LEAKAGE)
            s_output = ops.lyr_linear(
                'linear1', s_mid, fft_size*self.n_signal, axis=-1)
            s_output = ops.relu(s_output, hparams.RELU_LEAKAGE)
            s_output = tf.reshape(s_output, out_shape)
        return s_output


class ToySeparator(Separator):
    '''
    This separator is a 3 layer MLP for debugging purposes

    First we take the mean of spectrum across time to get
    3d tensor [batch_size, n_signal, fft_size]

    Then we pass that over an 3 layer MLP

    '''
    def __init__(self, model, name):
        self.name = name

    @classmethod
    def _get_embedding(cls, s_signals):
        edims = hparams.DISIM_EMBED_DIMS
        inp_shape = s_signals.get_shape().as_list()
        fft_size = inp_shape[-1]
        s_signals_sum = tf.reduce_mean(s_signals, axis=-2)
        s_mid = ops.lyr_linear('linear0', s_signals_sum, fft_size*2, axis=-1)
        s_mid = ops.relu(s_mid, hparams.RELU_LEAKAGE)
        s_output = ops.lyr_linear(
            'linear1', s_mid, edims, axis=-1)
        return s_output


class ToyDiscriminator(Discriminator):
    '''
    This Discriminator is for debugging purposes.
    '''
    use_text = False
    def __init__(self, model, name):
        self.name = name

    def __call__(self, s_signals, s_texts=False):
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
