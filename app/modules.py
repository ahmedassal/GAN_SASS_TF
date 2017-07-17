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
        with tf.variable_scope(self.name):
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

        with tf.variable_scope(self.name):
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
        with tf.variable_scope(self.name):
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


@hparams.register_discriminator('bilstm-3way-v1')
class BiLstmW3V0Discriminator(Discriminator):
    '''
    This discriminator applys two layer Bi-LSTM to both
    signal and text, then send the activation of last time frame
    to a linear projection layer to obtain logits
    '''
    def __init__(self, model, name):
        self.name = name
        self.model = model

    def __call__(self, s_signals, s_texts=None):
        rev_signal = (slice(None), slice(None), slice(None, None, -1))
        rev_text = (slice(None), slice(None, None, -1))
        with tf.variable_scope(self.name):
            s_signals_mid_fwd = self.model.lyr_lstm(
                'LSTM0_fwd', s_signals, 128, t_axis=-2)
            s_signals_mid_bwd = self.model.lyr_lstm(
                'LSTM0_bwd', s_signals[rev_signal], 128, t_axis=-2)
            s_signals_mid = tf.concat(
                [s_signals_mid_fwd, s_signals_mid_bwd[rev_signal]], axis=-1)

            s_signals_out_fwd = self.model.lyr_lstm(
                'LSTM1_fwd', s_signals_mid, 64, t_axis=-2)
            s_signals_out_bwd = self.model.lyr_lstm(
                'LSTM1_bwd', s_signals_mid[rev_signal], 64, t_axis=-2)
            s_signals_out = tf.concat(
                [s_signals_out_fwd[:, :, -1], s_signals_out_bwd[:, :, -1]],
                axis=-1)

            if s_texts is None:
                s_out = s_signals_out
            else:
                s_texts_mid_fwd = self.model.lyr_lstm(
                    'LSTM0_txt_fwd', s_texts, 64, t_axis=-2)
                s_texts_mid_bwd = self.model.lyr_lstm(
                    'LSTM0_txt_bwd', s_texts[rev_text], 64, t_axis=-2)
                s_texts_mid = tf.concat(
                    [s_texts_mid_fwd, s_texts_mid_bwd[rev_text]], axis=-1)

                s_texts_out_fwd = self.model.lyr_lstm(
                    'LSTM1_txt_fwd', s_texts_mid, 32, t_axis=-2)
                s_texts_out_bwd = self.model.lyr_lstm(
                    'LSTM1_txt_bwd', s_texts_mid[rev_text], 32, t_axis=-2)
                s_out = tf.concat([
                    s_texts_out_fwd[:, -1],
                    s_texts_out_bwd[:, -1],
                    s_signals_out], axis=-1)

            s_logits = ops.lyr_linear('linear_logits', s_out, 3, axis=-1)
        return s_logits


@hparams.register_recognizer('bilstm-ctc-v1')
class BiLstmCtcRecognizer(Recognizer):
    def __init__(self, model, name):
        self.name = name
        self.model = model

    def __call__(self, s_signals):
        raise NotImplementedError()


@hparams.register_separator('bilstm-v1')
class BiLstmSeparator(Separator):
    def __init__(self, model, name):
        self.name = name
        self.model = model

    def __call__(self, s_signals):
        n_outs = hparams.MAX_N_SIGNAL + 1
        fft_size = hparams.FFT_SIZE
        rev_signal = (slice(None), slice(None), slice(None, None, -1))
        with tf.variable_scope(self.name):
            s_mid0_fwd = self.model.lyr_lstm(
                'lstm0_fwd', s_signals, 128, t_axis=-2)
            s_mid0_bwd = self.model.lyr_lstm(
                'lstm0_bwd', s_signals[rev_signal], 128, t_axis=-2)
            s_mid0 = tf.concat(
                [s_mid0_fwd, s_mid0_bwd[rev_signal]], axis=-1)

            s_mid1_fwd = self.model.lyr_lstm(
                'lstm1_fwd', s_mid0, 256, t_axis=-2)
            s_mid1_bwd = self.model.lyr_lstm(
                'lstm1_bwd', s_mid0[rev_signal], 256, t_axis=-2)
            s_mid1 = tf.concat(
                [s_mid1_fwd, s_mid1_bwd[rev_signal]], axis=-1)

            s_out_fwd = self.model.lyr_lstm(
                'lstm2_fwd', s_mid1, 256, t_axis=-2)
            s_out_bwd = self.model.lyr_lstm(
                'lstm2_bwd', s_mid1[rev_signal], 256, t_axis=-2)
            s_out = tf.concat(
                [s_out_fwd, s_out_bwd[rev_signal]], axis=-1)
            s_out = ops.lyr_linear(
                'output', s_out, fft_size * n_outs, bias=False)
            s_out = tf.reshape(
                s_out, [hparams.BATCH_SIZE, -1, n_outs, fft_size])
            s_out = tf.transpose(s_out, [0, 2, 1, 3])
        return s_out


@hparams.register_separator('dc-v1')
class DeepClusterSeparator(Separator):
    '''
    This uses deep clustering to calculate mask
    '''
    def __init__(self, model, name):
        self.name = name
        self.model = model

    def __call__(self, s_signals):
        raise NotImplementedError()
