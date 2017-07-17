import tensorflow as tf

import app.hparams as hparams
import app.ops as ops

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


class Separator(ModelModule):
    '''
    Separate signal and noise from input signal
    '''
    def __init__(self, model, name):
        self.name = name

    def __call__(self, s_mixture):
        '''
        Args:
            s_mixture: tensor variable
                3d tensor of shape [batch_size, length, fft_size]

        Returns:
            [batch_size, n_signal+1, length, fft_size]

        Notes:
            `length` is a not constant
        '''
        raise NotImplementedError()


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
                [batch_size, n_signal+1, length, fft_size]

        Returns:
            logits, tensor variable of shape:
            [batch_size, n_signal+1, out_length, charset_size]

            perform argmax on last axis to obtain symbol idx

        Notes:
            - `length` is a not constant
            - `out_length` may not be the same as `length`
        '''
        pass


class Discriminator(ModelModule):
    '''
    Feed audio signal, optionally with text, output 3d logits
    '''
    def __init__(self, model, name):
        self.name = name

    def __call__(self, s_signals, s_texts):
        '''
        Args:
            s_signals: tensor variable
                shape: [batch_size, (n_signal+1)*2, length, fft_size]

            s_texts: tensor variable
                shape: [batch_size, (n_signal+1)*2, text_length, charset_size]

        Returns:
            tensor variable of shape [batch_size, (n_signal+1)*2, 3]
        '''
        raise NotImplementedError()


@hparams.register_separator('toy')
class ToySeparator(Separator):
    '''
    This separator is a 3 layer MLP for debugging purposes
    '''
    def __init__(self, model, name):
        self.name = name

    def __call__(self, s_signals):
        with tf.name_scope(self.name):
            s_mid = ops.lyr_linear(
                'linear0', s_signals, hparams.FFT_SIZE*2, axis=-1)
            s_mid = ops.relu(s_mid, hparams.RELU_LEAKAGE)
            s_out = ops.lyr_linear(
                'linear1', s_mid,
                hparams.FFT_SIZE * (hparams.MAX_N_SIGNAL+1), axis=-1)
            s_out = tf.reshape(
                s_out,
                [hparams.BATCH_SIZE,
                    (hparams.MAX_N_SIGNAL+1),
                    -1, hparams.FFT_SIZE])
        return s_out


@hparams.register_recognizer('toy')
class ToyRecognizer(Recognizer):
    '''
    Toy recognizer for debugging purposes

    This always output length 16 logits sequence using 3-layer MLP
    '''
    def __init__(self, model, name):
        self.name = name

    def __call__(self, s_signals):
        inp_shape = s_signals.get_shape().as_list()
        fft_size = inp_shape[-1]
        charset_size = hparams.CHARSET_SIZE
        text_length = 16

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

            s_input = tf.reshape(s_input, inp_shape)
            s_mid = ops.lyr_linear('linear0', s_input, inp_ndim*2, axis=-1)
            s_mid = ops.relu(s_mid, hparams.RELU_LEAKAGE)
            s_output = ops.lyr_linear('linear1', s_mid, 3, axis=-1)
        return s_output

@hparams.register_discriminator('bilstm-v1')
class BiLstmV0Discriminator(Discriminator):
    def __init__(self, model, name):
        self.name = name
        self.model = model

    def __call__(self, s_signals, s_texts=None):
        with tf.name_scope(self.name):
            pass
