'''
TensorFlow Implementation of "GAN for Single Source Audio Separation"

TODO docs
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from sys import stdout
from itertools import product
from functools import reduce
import os

import nltk
import numpy as np
import tensorflow as tf

try:
    import warpctc_tensorflow
except Exception as e:
    if type(e).__name__ in ['ImportError', 'ModuleNotFoundError']:
        print("Warning: not using warp_ctc, performance may be worse.")
    else:
        raise

import app.hparams as hparams
import app.ops as ops
import app.ozers as ozers
import app.datasets as datasets
import app.modules as modules


# Global vars
g_sess = tf.Session()
g_ctc_decoder = dict(
    beam=tf.nn.ctc_beam_search_decoder,
    greedy=tf.nn.ctc_greedy_decoder)[hparams.CTC_DECODER_TYPE]

def _dict_add(dst, src):
    for k,v in src.items():
        if k not in dst:
            dst[k] = v
        else:
            dst[k] += v


def _dict_format(di):
    return ' '.join('='.join((k, str(v))) for k,v in di.items())


def batch_levenshtein(x, y):
    '''
    Batched version of nltk.edit_distance, over character
    This performs edit distance over last axis, trimming trailing zeros.

    Args:
        x: int array, hypothesis
        y: int array, target

    Returns: int32 array
    '''
    x_shp = x.shape
    y_shp = y.shape
    assert x_shp[:-1] == y_shp[:-1]
    idx_iter = product(*map(range, x_shp[:-1]))

    z = np.empty(x_shp[:-1], dtype='int32')
    for idx in idx_iter:
        u, v = x[idx], y[idx]
        u = np.trim_zeros(u, 'b')
        v = np.trim_zeros(v, 'b')
        z[idx] = nltk.edit_distance(u, v)
    return z


def batch_wer(x, y, fn_decoder):
    '''
    Batched version of nltk.edit_distance, over words
    This performs edit distance over last axis, trimming trailing zeros.

    Args:
        x: int array, hypothesis
        y: int array, target
        fn_decoder: function to convert int vector into string

    Returns: int32 array
    '''
    x_shp = x.shape
    y_shp = y.shape
    assert x_shp[:-1] == y_shp[:-1]
    idx_iter = product(*map(range, x_shp[:-1]))

    z = np.empty(x_shp[-1], dtype='int32')
    for idx in idx_iter:
        x_str = fn_decoder(x[idx]).strip(' $').split(' ')
        y_str = fn_decoder(y[idx]).strip(' $').split(' ')
        z[idx] = nltk.edit_distance(x_str, y_str)
    return z


class Model(object):
    '''
    Base class for a fully trainable model

    Should be singleton
    '''
    def __init__(self, name='BaseModel'):
        self.name = name
        self.s_states_di = {}

    def lyr_lstm(
            self, name, s_x, hdim,
            axis=-1, t_axis=0, op_linear=ops.lyr_linear, reuse=False):
        '''
        Args:
            name: string
            s_x: input tensor
            hdim: size of hidden layer
            axis: which axis will RNN op get performed on
            t_axis: which axis would be the timeframe
            op_rnn: RNN layer function, defaults to ops.lyr_lstm
        '''
        x_shp = s_x.get_shape().as_list()
        ndim = len(x_shp)
        assert -ndim <= axis < ndim
        assert -ndim <= t_axis < ndim
        axis = axis % ndim
        t_axis = t_axis % ndim
        assert axis != t_axis
        # make sure t_axis is 0, to make scan work
        if t_axis != 0:
            if axis == 0:
                axis = t_axis % ndim
            perm = list(range(ndim))
            perm[0], perm[t_axis] = perm[t_axis], perm[0]
            s_x = tf.transpose(s_x, perm)
        x_shp[t_axis], x_shp[0] = x_shp[0], x_shp[t_axis]
        idim = x_shp[axis]
        assert isinstance(idim, int)
        h_shp = x_shp[1:].copy()
        h_shp[axis-1] = hdim
        with tf.variable_scope(name):
            zero_init = tf.constant_initializer(0.)
            v_cell = tf.get_variable(
                dtype=hparams.FLOATX,
                shape=h_shp, name='cell',
                trainable=False,
                initializer=zero_init)
            v_hid = tf.get_variable(
                dtype=hparams.FLOATX,
                shape=h_shp, name='hid',
                trainable=False,
                initializer=zero_init)
            self.s_states_di[v_cell.name] = v_cell
            self.s_states_di[v_hid.name] = v_hid

            op_lstm = lambda _h, _x: ops.lyr_lstm_flat(
                name='LSTM',
                s_x=_x, v_cell=_h[0], v_hid=_h[1],
                axis=axis-1, op_linear=op_linear)
            s_cell_seq, s_hid_seq = tf.scan(
                op_lstm, s_x, initializer=(v_cell, v_hid))
        return s_hid_seq if t_axis == 0 else tf.transpose(s_hid_seq, perm)

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
        separator = hparams.get_separator()(
            self, 'separator')
        recognizer = hparams.get_recognizer()(
            self, 'recognizer')
        discriminator = hparams.get_discriminator()(
            self, 'discriminator')

        bsize = hparams.BATCH_SIZE * hparams.MAX_N_SIGNAL
        input_shape = [bsize, None, hparams.FFT_SIZE]
        single_signal_shape = [hparams.BATCH_SIZE, None, hparams.FFT_SIZE]
        del(single_signal_shape[1])

        # build the GAN model
        s_src_signals = tf.placeholder(
            hparams.FLOATX,
            input_shape,
            name='source_signal')
        sS_src_texts = tf.sparse_placeholder(
            hparams.INTX,
            name='source_text'
        )
        with tf.variable_scope('G'):
            # s_texts_dense = tf.sparse_to_dense(
                # sS_src_texts.indices,
                # sS_src_texts.dense_shape,
                # sS_src_texts.values)
            # s_texts_dense = tf.reshape(
                # s_texts_dense, [hparams.BATCH_SIZE, hparams.MAX_N_SIGNAL, -1])
            # s_texts_dense = tf.one_hot(
                # s_texts_dense, hparams.CHARSET_SIZE, dtype=hparams.FLOATX)
            # TODO add mixing coeff ?
            s_mixed_signals = tf.reduce_sum(
                tf.reshape(s_src_signals, [
                    hparams.BATCH_SIZE,
                    hparams.MAX_N_SIGNAL,
                    -1, hparams.FFT_SIZE]), axis=1)
            s_noise_signal = tf.random_normal(
                tf.shape(s_mixed_signals),
                stddev=0.1,
                dtype=hparams.FLOATX)
            s_mixed_signals += s_noise_signal
            s_separated_signals = separator(s_mixed_signals)
            sS_pred_texts = recognizer(s_separated_signals)
            if recognizer.IS_CTC:
                bsize = hparams.BATCH_SIZE * (hparams.MAX_N_SIGNAL+1)
                sS_pred_texts = tf.transpose(
                    sS_pred_texts, [1, 0, 2])
                s_seqlen = tf.tile(
                    tf.shape(sS_pred_texts)[1:2],
                    [bsize])
                s_asr_loss = tf.nn.ctc_loss(
                    sS_src_texts, sS_pred_texts, s_seqlen)
                sS_pred_texts = g_ctc_decoder(sS_pred_texts, s_seqlen)[0][0]
            s_autoencoder_loss = tf.reduce_mean(
                tf.square(
                    tf.reduce_sum(
                        tf.reshape(s_separated_signals, [
                            hparams.BATCH_SIZE,
                            hparams.MAX_N_SIGNAL+1,
                            -1, hparams.FFT_SIZE]), axis=1
                        ) - s_mixed_signals),
                axis=None)

        with tf.variable_scope('D'):
            s_truth_signals = tf.concat([
                tf.reshape(s_src_signals, [
                    hparams.BATCH_SIZE,
                    hparams.MAX_N_SIGNAL,
                    -1, hparams.FFT_SIZE]),
                tf.expand_dims(s_noise_signal, 1)], axis=1)
            s_truth_signals = tf.reshape(s_truth_signals, [
                hparams.BATCH_SIZE * (hparams.MAX_N_SIGNAL+1),
                -1, hparams.FFT_SIZE])

            # convert source text to dense one-hot
            s_src_texts = tf.sparse_to_dense(
                sS_src_texts.indices,
                sS_src_texts.dense_shape,
                sS_src_texts.values,
                default_value=hparams.CHARSET_SIZE)
            s_src_texts = tf.reshape(
                s_src_texts, [hparams.BATCH_SIZE, hparams.MAX_N_SIGNAL, -1])
            s_src_texts = tf.one_hot(
                s_src_texts, hparams.CHARSET_SIZE, dtype=hparams.FLOATX)
            # TODO use non-zero const padding once the feature is up
            pad_const = 1. / hparams.CHARSET_SIZE
            s_truth_texts = tf.pad(
                s_src_texts - pad_const,
                ((0,0), (0,1), (0,0), (0,0)), mode='CONSTANT') + pad_const
            s_truth_texts = tf.reshape(s_truth_texts, [
                hparams.BATCH_SIZE * (hparams.MAX_N_SIGNAL+1),
                -1, hparams.CHARSET_SIZE])

            # convert predicted text to dense one-hot
            s_pred_texts = tf.sparse_to_dense(
                sS_pred_texts.indices,
                sS_pred_texts.dense_shape,
                sS_pred_texts.values,
                default_value=hparams.CHARSET_SIZE)
            s_guess_texts = tf.one_hot(
                s_pred_texts,
                hparams.CHARSET_SIZE,
                dtype=hparams.FLOATX)
            # pad additional zero at end to make sure no zero length text
            s_guess_texts = tf.pad(s_guess_texts, ((0,0), (0,1), (0,0)))

            with tf.variable_scope('dtor', reuse=False) as scope:
                s_guess_t = discriminator(s_truth_signals, s_truth_texts)
                scope.reuse_variables()
                s_guess_f = discriminator(
                    s_separated_signals,
                    s_guess_texts)
            s_truth = tf.concat([
                tf.constant(
                    hparams.CLS_REAL_SIGNAL,
                    hparams.INTX,
                    [hparams.MAX_N_SIGNAL]),
                tf.constant(
                    hparams.CLS_REAL_NOISE,
                    hparams.INTX,
                    [1])], axis=-1)
            s_truth = tf.one_hot(tf.tile(s_truth, [hparams.BATCH_SIZE]), 3)
            s_lies = tf.constant(
                    hparams.CLS_FAKE_SIGNAL,
                    hparams.INTX,
                    [(hparams.MAX_N_SIGNAL+1)*hparams.BATCH_SIZE])
            s_lies = tf.one_hot(s_lies, 3)
            s_gan_loss_t = tf.nn.softmax_cross_entropy_with_logits(
                labels=s_truth, logits=s_guess_t)
            s_gan_loss_f = tf.nn.softmax_cross_entropy_with_logits(
                labels=s_lies, logits=s_guess_f)
            s_gan_loss = tf.reduce_mean(s_gan_loss_t + s_gan_loss_f)

        # prepare summary
        # TODO add impl & summary for word error rate
        # TODO add impl & summary for SNR

        # FIXME gan_loss summary is broken
        with tf.name_scope('summary'):
            tf.summary.scalar('gan_loss', s_gan_loss)
            tf.summary.scalar('ae_loss', s_autoencoder_loss)

        # apply optimizer
        ozer = hparams.get_optimizer()(learn_rate=hparams.LR)

        v_params_li = tf.trainable_variables()
        s_gparams_li = [v for v in v_params_li if v.name.startswith('G/')]
        s_dparams_li = [v for v in v_params_li if v.name.startswith('D/')]

        op_fit_generator = ozer.minimize(
            s_autoencoder_loss - s_gan_loss, var_list=s_gparams_li)
        op_fit_discriminator = ozer.minimize(
            s_gan_loss, var_list=s_dparams_li)
        self.op_init_params = tf.variables_initializer(v_params_li)
        self.op_init_states = tf.variables_initializer(
            list(self.s_states_di.values()))

        self.all_summary = tf.summary.merge_all()
        self.feed_keys = [s_src_signals, sS_src_texts]
        self.train_fetches = [
            self.all_summary,
            dict(gan_loss=s_gan_loss, ae_loss=s_autoencoder_loss),
            op_fit_generator, op_fit_discriminator]

        self.infer_fetches = [dict(
            signals=s_separated_signals,
            texts=sS_pred_texts)]

        # FOR DEBUGGING
        # g_sess.run([self.op_init_params])
        # _feed = {
            # s_src_signals : np.random.rand(4, 3, 128, 256),
            # sS_src_texts : np.random.randint(0, hparams.CHARSET_SIZE, (4,3,32)) }
        # ret = g_sess.run([s_gan_loss], _feed)


    def train(self, n_epoch, dataset=None):
        train_writer = tf.summary.FileWriter(hparams.SUMMARY_DIR, g_sess.graph)
        for i_epoch in range(n_epoch):
            cli_report = {}
            for data_pt in dataset.epoch('train', hparams.BATCH_SIZE * hparams.MAX_N_SIGNAL):
                to_feed = dict(zip(self.feed_keys, data_pt))
                step_summary, step_fetch = g_sess.run(
                    self.train_fetches, to_feed)[:2]
                train_writer.add_summary(step_summary)
                # stdout.write('.')
                # stdout.flush()
                _dict_add(cli_report, step_fetch)
            print('Epoch %d/%d %s' % (
                i_epoch+1, n_epoch, _dict_format(cli_report)))
            stdout.write('\n')
            stdout.flush()

    def reset(self):
        '''re-initialize parameters, resets timestep'''
        g_sess.run([self.op_init_params, self.op_init_states])

    def reset_state(self):
        g_sess.run([self.op_init_states])

    def parameter_count(self):
        '''
        Returns: integer
        '''
        v_vars_li = tf.trainable_variables()
        return sum(
            reduce(int.__mul__, v.get_shape().as_list()) for v in v_vars_li)


def main():
    # TODO parse cmd args
    # TODO manage device
    print('Preparing dataset ... ', end='')
    stdout.flush()
    g_dataset = hparams.get_dataset()()
    g_dataset.install_and_load()
    print('done')
    stdout.flush()

    print('Building model ... ', end='')
    model = Model()
    model.build()
    print('done')
    stdout.flush()
    model.reset()
    model.train(n_epoch=10, dataset=g_dataset)

    # TODO inference

def debug_test():
    x = np.random.randint(0, 2, (32, 32, 32))
    y = np.random.randint(0, 2, (32, 32, 32))
    d = batch_levenshtein(x, y)
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
    # debug_test()
