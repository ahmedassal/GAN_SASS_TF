'''
TensorFlow Implementation of "GAN for Single Source Audio Separation"

TODO docs
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import pdb

import numpy as np
import tensorflow as tf

import hparams, ozers, ops


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
        return - tf.square(s_cosines / (s_norms_outer + hparams.EPS))

    def __call__(self, s_signals):
        '''
        Args:
            s_signals: tensor variable
                4d tensor of shape [batch_size, n_signal, length, fft_size]
                This tensor is expected to be input from extractor

        Returns:
            batched dissimilarity matrices [batch_size, n_signal, n_signal]
        '''
        with tf.name_scope(self.name):
            s_embeds = self._get_embedding(s_signals)
            s_dismat = Separator._embedding_to_dismat(s_embeds)
        return s_dismat


class Recognizer(ModelModule):
    '''
    speech recognizer, need clean voice as input
    '''
    def __init__(self, model, name):
        pass

    def __call__(self, s_signals):
        '''
        Args:
            s_signals: tensor variable
                [batch_size, n_signal, length, fft_size]

        Returns:
            logits, tensor variable of shape:
            [batch_size, n_signal, out_length, charset_size]

            perform argmax on last axis to obtain symbol idx
        '''
        pass


class Discriminator(ModelModule):
    '''
    Feed audio signal, optionally with text, output scalar within [0., 1.]
    '''
    def __init__(self, model, name):
        self.name = name

    def __call__(self, s_signals, s_texts):
        '''
        Args:
            s_signals: tensor variable
                shape: [batch_size, n_signal, length, fft_size]

            s_texts: tensor variable
                shape: [batch_size, n_signal, text_length, charset_size]

        Returns:
            tensor variable of shape [batch_size, n_signal]
        '''
        raise NotImplementedError()


@hparams.register_extractor('toy')
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
        out_shape.insert(1, self.n_signal)
        fft_size = inp_shape[-1]
        with tf.name_scope(self.name):
            s_mid = ops.lyr_linear(
                'linear0', s_input, fft_size*2, axis=-1)
            s_mid = ops.relu(s_mid, hparams.RELU_LEAKAGE)
            s_output = ops.lyr_linear(
                'linear1', s_mid, fft_size*self.n_signal, axis=-1)
            s_output = ops.relu(s_output, hparams.RELU_LEAKAGE)
            s_output = tf.reshape(s_output, out_shape)
        return s_output


@hparams.register_separator('toy')
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


@hparams.register_recognizer('toy')
class ToyRecognizer(Recognizer):
    '''
    Toy recognizer for debugging purposes

    This always output max-text-length logits sequence using 3-layer MLP
    '''
    def __init__(self, model, name):
        self.name = name

    def __call__(self, s_signals):
        inp_shape = s_signals.get_shape().as_list()
        fft_size = inp_shape[-1]
        charset_size = hparams.CHARSET_SIZE
        text_length = hparams.MAX_TEXT_LENGTH

        out_shape = inp_shape.copy()
        out_shape[-1] = charset_size
        out_shape[-2] = text_length

        with tf.name_scope(self.name):
            s_mean = tf.reduce_mean(s_signals, axis=-2)
            s_mid = ops.lyr_linear('linear0', s_mean, fft_size*2, axis=-1)
            s_mid = ops.relu(s_mid, hparams.RELU_LEAKAGE)
            s_logits = ops.lyr_linear(
                'linear1', s_mid, charset_size*text_length, axis=-1)
            s_logits = tf.reshape(s_logits, out_shape)
        return s_logits


@hparams.register_discriminator('toy')
class ToyDiscriminator(Discriminator):
    '''
    This Discriminator is for debugging purposes.
    '''
    def __init__(self, model, name):
        self.name = name

    def __call__(self, s_signals, s_texts=None):
        with tf.name_scope(self.name):
            s_signals_mean = tf.reduce_mean(s_signals, axis=-2)
            if s_texts is not None:
                s_texts_mean = tf.reduce_mean(s_texts, axis=-2)
                s_input = tf.concat(
                    [s_signals_mean, s_texts_mean], axis=-1)
            else:
                s_input = s_signals_mean
            inp_shape = s_input.get_shape().as_list()
            inp_ndim = inp_shape[-1]
            n_signal = inp_shape[1]

            s_input = tf.reshape(s_input, inp_shape)
            s_mid = ops.lyr_linear('linear0', s_input, inp_ndim*2, axis=-1)
            s_mid = ops.relu(s_mid, hparams.RELU_LEAKAGE)
            s_output = ops.lyr_linear('linear1', s_mid, n_signal, axis=-1)
            s_output = tf.sigmoid(s_output)
        return s_output


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

    def build(self):

        # create sub-modules
        extractor = hparams.get_extractor()(
            self, 'extractor', n_signal=hparams.MAX_N_SIGNAL)
        separator = hparams.get_separator()(
            self, 'separator')
        recognizer = hparams.get_recognizer()(
            self, 'recognizer')
        discriminator = hparams.get_discriminator()(
            self, 'discriminator')

        input_shape = [
            hparams.BATCH_SIZE,
            hparams.MAX_N_SIGNAL,
            hparams.SIGNAL_LENGTH,
            hparams.FFT_SIZE]
        input_text_shape = [
            hparams.BATCH_SIZE,
            hparams.MAX_N_SIGNAL,
            hparams.MAX_TEXT_LENGTH]
        single_signal_shape = input_shape.copy()
        del(single_signal_shape[1])

        s_source_signals = tf.placeholder(
            hparams.FLOATX,
            input_shape,
            name='source_signal')
        s_source_texts = tf.placeholder(
            hparams.INTX,
            input_text_shape,
            name='source_text'
        )
        with tf.name_scope('G'):
            s_texts = tf.one_hot(
                s_source_texts, hparams.CHARSET_SIZE, dtype=hparams.FLOATX)
            # TODO add mixing coeff ?
            s_noise_signal = tf.random_normal(
                single_signal_shape,
                stddev=0.1,
                dtype=hparams.FLOATX)
            s_mixed_signals = \
                tf.reduce_sum(s_source_signals, axis=1) + s_noise_signal
            s_extracted_signals = extractor(s_mixed_signals)
            s_dismat = separator(s_extracted_signals)
            # since we use L2-normalized vector to get dismat
            # diagonal elements does not matter (always 1)
            s_dissim_loss = tf.reduce_mean(
                tf.square(tf.reduce_sum(s_dismat, axis=[1, 2])))
            s_predicted_texts = recognizer(s_extracted_signals)

        with tf.name_scope('D'):
            s_all_signals = tf.concat(
                [s_source_signals, s_extracted_signals], axis=1)
            s_all_texts = tf.concat(
                [s_texts, tf.nn.softmax(s_predicted_texts)], axis=1)
            s_guess = discriminator(s_all_signals, s_all_texts)
            s_truth = tf.concat([
                tf.zeros(
                    [hparams.BATCH_SIZE, hparams.MAX_N_SIGNAL],
                    dtype=hparams.FLOATX),
                tf.ones(
                    [hparams.BATCH_SIZE, hparams.MAX_N_SIGNAL],
                    dtype=hparams.FLOATX)], axis=-1)
            s_gan_loss = tf.reduce_mean(s_guess - s_truth, axis=None)

        ozer = hparams.get_optimizer()

        v_params_li = tf.trainable_variables()
        s_gparams_li = [v for v in v_params_li if v.name.startswith('G/')]
        s_dparams_li = [v for v in v_params_li if v.name.startswith('D/')]

        op_fit_generator = ozer.minimize(
            s_dissim_loss - s_gan_loss, var_list=s_gparams_li)
        op_fit_discriminator = ozer.minimize(
            s_gan_loss, var_list=s_dparams_li)

        self.feeds = [s_source_signals, s_source_texts]
        self.train_fetches = [
            dict(gan_loss=s_gan_loss, dissim_loss=s_dissim_loss),
            op_fit_generator, op_fit_discriminator]

        self.infer_fetches = [dict(
            signals=s_extracted_signals,
            texts=s_predicted_texts)]

    def train(self, n_epoch, dataset=None):
        pass

    def reset(self):
        '''re-initialize parameters, resets timestep'''
        # TODO
        pass

if __name__ == '__main__':
    # TODO parse cmd args
    # TODO manage device
    model = Model()
    model.build()
    # TODO train or inference
    # TODO write summary file
    pass
