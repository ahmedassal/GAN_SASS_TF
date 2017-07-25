'''
TensorFlow Implementation of "GAN for Single Source Audio Separation"

TODO docs
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from math import sqrt
import argparse
from sys import stdout
from collections import OrderedDict
from functools import reduce
import os
import copy

import numpy as np
import scipy.signal
import scipy.io.wavfile
import tensorflow as tf
# remove annoying "I tensorflow ..." logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import warpctc_tensorflow
except Exception as e:
    if type(e).__name__ in ['ImportError', 'ModuleNotFoundError']:
        stdout.write("Warning: not using warp-ctc, performance may degrade.\n")
        stdout.flush()
    else:
        raise

import app.datasets as datasets
import app.hparams as hparams
import app.modules as modules
import app.ops as ops
import app.ozers as ozers
import app.utils as utils


# Global vars
g_sess = tf.Session()
g_ctc_decoder = dict(
    beam=tf.nn.ctc_beam_search_decoder,
    greedy=tf.nn.ctc_greedy_decoder)[hparams.CTC_DECODER_TYPE]
g_args = None
g_model = None
g_dataset = None

def _dict_add(dst, src):
    for k,v in src.items():
        if k not in dst:
            dst[k] = v
        else:
            dst[k] += v


def _dict_mul(di, coeff):
    for k,v in di.items():
        di[k] = v * coeff


def _dict_format(di):
    return ' '.join('='.join((k, str(v))) for k,v in di.items())


def load_wavfile(filename):
    '''
    This loads a WAV file, resamples to 16k/sec rate,
    then preprocess it

    Args:
        filename: string

    Returns:
        numpy array of shape [time, FFT_SIZE]
    '''
    if filename is None:
        # TODO in this case, draw a sample from dataset instead of raise ?
        raise FileNotFoundError(
                'WAV file not specified, '
                'please specify via --input-file argument.')
    smprate, data = scipy.io.wavfile.read(filename)
    if data.ndim != 1:
        print(
            'Warning: WAV file is not of single channel'
            ', using the first channel')
        data = data[(0,)*(data.ndim-1)]
    if smprate != 16000:
        nsmp = data.shape[-1]
        new_nsmp = int(max(nsmp * (16000 / smprate), 1))
        data = scipy.signal.resample(data, new_nsmp, axis=0)
        padsize = hparams.FFT_SIZE - ((data.shape[-1] - 1) % hparams.FFT_SIZE) - 1
        data = np.pad(
            data, [(0,0)]*(data.ndim-1) + [(0, padsize)], mode='constant')

    spectrogram = scipy.signal.stft(data, nperseg=hparams.FFT_SIZE)[2]
    feature = utils.spectrum_to_feature(spectrogram)
    return feature


def save_wavfile(filename, feature):
    '''
    Saves time series of features into a WAV file

    Args:
        filename: string
        feature: 2D float array of shape [time, FFT_SIZE]
    '''
    spectrogram = utils.feature_to_spectrum(feature)
    _, data = scipy.signal.istft(spectrogram, nperseg=hparams.FFT_SIZE)
    data_min = np.min(data)
    data_max = np.max(data)
    data -= data_min
    data *= (32767. / (data_max - data_min))
    scipy.io.wavfile.write(filename, 16000, data.astype(np.int16))


def prompt_yesno(q_):
    while True:
        action=input(q_ + ' [Y]es [n]o : ')
        if action == 'Y':
            return True
        elif action == 'n':
            return False


def prompt_overwrite(filename_):
    '''
    If save file obj_.__getattribute__(attr_) exists, prompt user to
    give up, overwrite, or make copy.

    obj_.__getattribute__(attr_) should be a string, the string may be
    changed after prompt
    '''
    try:
        savfile = open(filename_, 'x')
    except FileExistsError:
        while True:
            action = input(
                'file %s exists, overwrite? [Y]es [n]o [c]opy : '%filename_)
            if action == 'Y':
                return filename_
            elif action == 'n':
                return ''
            elif action == 'c':
                i=0
                while True:
                    new_filename = filename_+'.'+str(i)
                    try:
                        savfile = open(new_filename, 'x')
                    except FileExistsError:
                        i+=1
                        continue
                    break
                return new_filename
    else:
        savfile.close()


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
            axis=-1, t_axis=0, op_linear=ops.lyr_linear):
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
        h_shp = copy.copy(x_shp[1:])
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

            init_range = 0.1 / sqrt(hdim)
            op_lstm = lambda _h, _x: ops.lyr_lstm_flat(
                name='LSTM',
                s_x=_x, v_cell=_h[0], v_hid=_h[1],
                axis=axis-1, op_linear=op_linear,
                w_init=tf.random_uniform_initializer(
                    -init_range, init_range, dtype=hparams.FLOATX))
            s_cell_seq, s_hid_seq = tf.scan(
                op_lstm, s_x, initializer=(v_cell, v_hid))
        return s_hid_seq if t_axis == 0 else tf.transpose(s_hid_seq, perm)

    def lyr_gru(
            self, name, s_x, hdim,
            axis=-1, t_axis=0, op_linear=ops.lyr_linear):
        '''
        Args:
            name: string
            s_x: input tensor
            hdim: size of hidden layer
            axis: which axis will RNN op get performed on
            t_axis: which axis would be the timeframe
            op_rnn: RNN layer function, defaults to ops.lyr_gru
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
        h_shp = copy.copy(x_shp[1:])
        h_shp[axis-1] = hdim
        with tf.variable_scope(name):
            zero_init = tf.constant_initializer(0.)
            v_cell = tf.get_variable(
                dtype=hparams.FLOATX,
                shape=h_shp, name='cell',
                trainable=False,
                initializer=zero_init)
            self.s_states_di[v_cell.name] = v_cell

            init_range = 0.1 / sqrt(hdim)
            op_gru = lambda _h, _x: ops.lyr_gru_flat(
                'GRU', _x, _h[0],
                axis=axis-1, op_linear=op_linear,
                w_init=tf.random_uniform_initializer(
                    -init_range, init_range, dtype=hparams.FLOATX))
            s_cell_seq, = tf.scan(
                op_gru, s_x, initializer=(v_cell,))
        return s_cell_seq if t_axis == 0 else tf.transpose(s_cell_seq, perm)

    def save_params(self, filename, step=None):
        global g_sess
        save_dir = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.saver.save(g_sess,
                        filename,
                        global_step=step)

    def load_params(self, filename):
        # if not os.path.exists(filename):
            # stdout.write('Parameter file "%s" does not exist\n' % filename)
            # return False
        self.saver.restore(g_sess, filename)
        return True

    def build(self):
        # TODO refactor this super long function
        # create sub-modules
        separator = hparams.get_separator()(
            self, 'separator')
        recognizer = hparams.get_recognizer()(
            self, 'recognizer')
        discriminator = hparams.get_discriminator()(
            self, 'discriminator')


        # ===================
        # build the GAN model

        bsize = hparams.BATCH_SIZE * hparams.MAX_N_SIGNAL
        input_shape = [bsize, None, hparams.FFT_SIZE]
        single_signal_shape = [hparams.BATCH_SIZE, None, hparams.FFT_SIZE]
        del(single_signal_shape[1])

        s_src_signals = tf.placeholder(
            hparams.FLOATX,
            input_shape,
            name='source_signal')
        sS_src_texts = tf.sparse_placeholder(
            hparams.INTX,
            name='source_text')
        s_dropout_keep = tf.placeholder(
            hparams.FLOATX,
            [], name='dropout_keep')
        reger = hparams.get_regularizer()
        with tf.variable_scope('G', regularizer=reger):
            # TODO add mixing coeff ?

            # get mixed signal
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
            s_mixed_signals_log = ops.to_log_signal(s_mixed_signals)

            s_separated_signals_log = separator(
                s_mixed_signals_log, s_dropout_keep=s_dropout_keep)
            s_separated_signals = ops.to_exp_signal(s_separated_signals_log)
            s_pred_texts = recognizer(
                s_separated_signals_log, s_dropout_keep=s_dropout_keep)
            if recognizer.IS_CTC:
                bsize = hparams.BATCH_SIZE * (hparams.MAX_N_SIGNAL+1)
                s_pred_texts = tf.transpose(
                    s_pred_texts, [1, 0, 2])
                s_seqlen = tf.tile(
                    tf.shape(s_pred_texts)[1:2],
                    [bsize])
                sS_pred_texts = g_ctc_decoder(s_pred_texts, s_seqlen)[0][0]
            s_autoencoder_loss = tf.reduce_mean(
                tf.square(
                    tf.reduce_sum(
                        tf.reshape(s_separated_signals, [
                            hparams.BATCH_SIZE,
                            hparams.MAX_N_SIGNAL+1,
                            -1, hparams.FFT_SIZE]), axis=1
                        ) - s_mixed_signals),
                axis=None)

        with tf.variable_scope('D', regularizer=reger):
            s_truth_signals = tf.concat([
                tf.reshape(s_src_signals, [
                    hparams.BATCH_SIZE,
                    hparams.MAX_N_SIGNAL,
                    -1, hparams.FFT_SIZE]),
                tf.expand_dims(s_noise_signal, 1)], axis=1)
            s_truth_signals = tf.reshape(s_truth_signals, [
                hparams.BATCH_SIZE * (hparams.MAX_N_SIGNAL+1),
                -1, hparams.FFT_SIZE])
            s_truth_signals_log = ops.to_log_signal(s_truth_signals)

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
            s_pred_texts_oh = tf.sparse_to_dense(
                sS_pred_texts.indices,
                sS_pred_texts.dense_shape,
                sS_pred_texts.values,
                default_value=hparams.CHARSET_SIZE)
            s_guess_texts = tf.one_hot(
                s_pred_texts_oh,
                hparams.CHARSET_SIZE,
                dtype=hparams.FLOATX)
            # pad additional zero at end to make sure no zero length text
            s_guess_texts = tf.pad(s_guess_texts, ((0,0), (0,1), (0,0)))

            with tf.variable_scope('dtor', reuse=False) as scope:
                if hparams.USE_ASR:
                    s_guess_t = discriminator(
                        s_truth_signals_log,
                        s_truth_texts,
                        s_dropout_keep=s_dropout_keep)
                    scope.reuse_variables()
                    s_guess_f = discriminator(
                        s_separated_signals_log,
                        s_guess_texts,
                        s_dropout_keep=s_dropout_keep)
                else:
                    s_guess_t = discriminator(
                        s_truth_signals_log,
                        s_dropout_keep=s_dropout_keep)
                    scope.reuse_variables()
                    s_guess_f = discriminator(
                        s_separated_signals_log,
                        s_dropout_keep=s_dropout_keep)
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
            s_cross_snr = ops.batch_cross_snr(
                tf.reshape(s_src_signals, [
                    hparams.BATCH_SIZE,
                    hparams.MAX_N_SIGNAL,
                    -1, hparams.FFT_SIZE]),
                tf.reshape(s_separated_signals, [
                    hparams.BATCH_SIZE,
                    hparams.MAX_N_SIGNAL+1,
                    -1, hparams.FFT_SIZE]))
            # drop the one with lowest SNR (likely separated noise)
            s_snr = tf.reduce_mean(tf.reduce_max(
                s_cross_snr, axis=-1))

        # ===============
        # prepare summary
        # TODO add impl & summary for word error rate

        # FIXME gan_loss summary is broken
        with tf.name_scope('summary'):
            s_gan_loss_summary = tf.summary.scalar('gan_loss', s_gan_loss)
            s_ae_loss_summary = tf.summary.scalar('ae_loss', s_autoencoder_loss)
            s_snr_summary = tf.summary.scalar('SNR', s_snr)

            s_gan_loss_summary_test = tf.summary.scalar('gan_loss_test', s_gan_loss)
            s_ae_loss_summary_test = tf.summary.scalar('ae_loss_test', s_autoencoder_loss)
            s_snr_summary_test = tf.summary.scalar('SNR_test', s_snr)

        # apply optimizer
        ozer = hparams.get_optimizer()(
            learn_rate=hparams.LR, lr_decay=hparams.LR_DECAY)

        v_params_li = tf.trainable_variables()
        v_gparams_li = [v for v in v_params_li if v.name.startswith('G/')]
        v_dparams_li = [v for v in v_params_li if v.name.startswith('D/')]

        op_fit_generator = ozer.minimize(
            s_autoencoder_loss - s_gan_loss, var_list=v_gparams_li)
        op_fit_discriminator = ozer.minimize(
            s_gan_loss, var_list=v_dparams_li)
        self.op_init_params = tf.variables_initializer(v_params_li)
        self.op_init_states = tf.variables_initializer(
            list(self.s_states_di.values()))

        self.train_feed_keys = [
            s_src_signals, sS_src_texts, s_dropout_keep]
        train_summary = tf.summary.merge(
            [s_gan_loss_summary, s_ae_loss_summary, s_snr_summary])
        self.train_fetches = [
            train_summary,
            dict(
                gan_loss=s_gan_loss,
                ae_loss=s_autoencoder_loss,
                SNR=s_snr),
            op_fit_generator, op_fit_discriminator]
        # TODO once ASR is done, add WER as metric

        self.test_feed_keys = self.train_feed_keys
        test_summary = tf.summary.merge(
            [s_gan_loss_summary_test, s_ae_loss_summary_test, s_snr_summary_test])
        self.test_fetches = [
            test_summary,
            dict(
                gan_loss=s_gan_loss,
                ae_loss=s_autoencoder_loss,
                SNR=s_snr)]

        self.infer_feed_keys = [s_mixed_signals]
        if hparams.USE_ASR:
            self.infer_fetches = dict(
                signals=s_separated_signals,
                texts=sS_pred_texts)
        else:
            self.infer_fetches = dict(signals=s_separated_signals)

        self.saver = tf.train.Saver(var_list=v_params_li)

        # ===================
        # build the ASR model

        # FIXME signal -> log_signal
        ASR_BATCH_SIZE = hparams.BATCH_SIZE * (hparams.MAX_N_SIGNAL+1)
        input_shape = [ASR_BATCH_SIZE, None, hparams.FFT_SIZE]
        s_src_signals_asr = tf.placeholder(
            hparams.FLOATX,
            input_shape,
            name='source_signal_asr')

        with tf.variable_scope('G', reuse=True):
            s_pred_texts = recognizer(s_src_signals_asr)
            # batch-major -> time-major
            s_pred_texts = tf.transpose(s_pred_texts, (1,0,2))
            s_seqlen = tf.tile(tf.shape(s_pred_texts)[1:2], [ASR_BATCH_SIZE])
            if recognizer.IS_CTC:
                s_asr_loss = tf.nn.ctc_loss(
                    sS_src_texts,
                    s_pred_texts,
                    s_seqlen,
                    ignore_longer_outputs_than_inputs=True)
                s_asr_loss = tf.reduce_mean(s_asr_loss)
            else:
                raise NotImplementedError(
                    'Non-CTC ASR training routine not implemented.')
            sS_pred_texts = g_ctc_decoder(s_pred_texts, s_seqlen)[0][0]
            sS_pred_texts = tf.cast(sS_pred_texts, hparams.INTX)
            s_wer = tf.edit_distance(
                sS_pred_texts, sS_src_texts, normalize=True)
            s_wer = tf.reduce_mean(s_wer)

        self.op_fit_asr = ozer.minimize(s_asr_loss, var_list=v_gparams_li)
        asr_train_summary = []
        with tf.name_scope('summary'):
            asr_train_summary.append(tf.summary.scalar('WER', s_wer))
            asr_train_summary.append(tf.summary.scalar('CTC_loss', s_asr_loss))
        asr_train_summary = tf.summary.merge(asr_train_summary)

        self.asr_train_feed_keys = [s_src_signals_asr, sS_src_texts]
        self.asr_train_fetches = [
            asr_train_summary,
            dict(WER=s_wer, CTC_loss=s_asr_loss),
            self.op_fit_asr]


    def train(self, n_epoch, dataset):
        global g_args
        train_writer = tf.summary.FileWriter(hparams.SUMMARY_DIR, g_sess.graph)
        for i_epoch in range(n_epoch):
            cli_report = OrderedDict()
            for i_batch, data_pt in enumerate(dataset.epoch(
                    'train',
                    hparams.BATCH_SIZE * hparams.MAX_N_SIGNAL,
                    shuffle=True)):
                to_feed = dict(zip(self.train_feed_keys, data_pt + (hparams.DROPOUT_KEEP_PROB,)))
                step_summary, step_fetch = g_sess.run(
                    self.train_fetches, to_feed)[:2]
                self.reset_state()
                train_writer.add_summary(step_summary)
                stdout.write(':')
                stdout.flush()
                _dict_add(cli_report, step_fetch)
            _dict_mul(cli_report, 1. / (i_batch+1))
            if not g_args.no_save_on_epoch:
                if float('nan') in cli_report.values():
                    if i_epoch:
                        stdout.write('\nEpoch %d/%d got NAN values, restoring last checkpoint ... ')
                        stdout.flush()
                        i_epoch -= 1
                        # FIXME: this path don't work windows
                        self.load_params('saves/' + self.name + ('_e%d' % (i_epoch+1)))
                        stdout.write('done')
                        stdout.flush()
                        continue
                    else:
                        stdout.write('\nRun into NAN during 1st epoch, exiting ...')
                        sys.exit(-1)
                self.save_params('saves/' + self.name + ('_e%d' % (i_epoch+1)))
                stdout.write('S')
            stdout.write('\nEpoch %d/%d %s\n' % (
                i_epoch+1, n_epoch, _dict_format(cli_report)))
            stdout.flush()
            if g_args.no_test_on_epoch:
                continue
            cli_report = OrderedDict()
            for i_batch, data_pt in enumerate(dataset.epoch(
                    'test',
                    hparams.BATCH_SIZE * hparams.MAX_N_SIGNAL,
                    shuffle=False)):
                # note: disable dropout during test
                to_feed = dict(zip(self.train_feed_keys, data_pt + (1.,)))
                step_summary, step_fetch = g_sess.run(
                    self.test_fetches, to_feed)[:2]
                self.reset_state()
                train_writer.add_summary(step_summary)
                stdout.write('.')
                stdout.flush()
                _dict_add(cli_report, step_fetch)
            _dict_mul(cli_report, 1. / (i_batch+1))
            stdout.write('\nTest  %d/%d %s\n' % (
                i_epoch+1, n_epoch, _dict_format(cli_report)))
            stdout.flush()

    def train_asr(self, n_epoch, dataset):
        global g_args
        train_writer = tf.summary.FileWriter(
            hparams.ASR_SUMMARY_DIR, g_sess.graph)
        ASR_BATCH_SIZE = hparams.BATCH_SIZE * (hparams.MAX_N_SIGNAL+1)
        for i_epoch in range(n_epoch):
            cli_report = {}
            for data_pt in dataset.epoch(
                    'train', ASR_BATCH_SIZE):
                to_feed = dict(zip(self.asr_train_feed_keys, data_pt + (hparams.DROPOUT_KEEP_PROB,)))
                step_summary, step_fetch = g_sess.run(
                    self.asr_train_fetches, to_feed)[:2]
                self.reset_state()
                train_writer.add_summary(step_summary)
                stdout.write('.')
                stdout.flush()
                _dict_add(cli_report, step_fetch)
            if not g_args.no_save_on_epoch:
                self.save_params(self.name, i_epoch+1)
                stdout.write('S')
            stdout.write('\nEpoch %d/%d %s' % (
                i_epoch+1, n_epoch, _dict_format(cli_report)))
            # TODO add validation set
            stdout.write('\n')
            stdout.flush()

    def test(self, dataset):
        global g_args
        train_writer = tf.summary.FileWriter(
            hparams.ASR_SUMMARY_DIR, g_sess.graph)
        cli_report = {}
        for data_pt in dataset.epoch(
                'test', hparams.BATCH_SIZE * hparams.MAX_N_SIGNAL):
            to_feed = dict(zip(self.train_feed_keys, data_pt + (1.,)))
            step_summary, step_fetch = g_sess.run(
                self.test_fetches, to_feed)[:2]
            train_writer.add_summary(step_summary)
            stdout.write('.')
            stdout.flush()
            _dict_add(cli_report, step_fetch)
        stdout.write('Test: %s\n' % (
            _dict_format(cli_report)))

    def asr_test(self, dataset):
        raise NotImplementedError()

    def reset(self):
        '''re-initialize parameters, resets timestep'''
        g_sess.run(tf.global_variables_initializer())

    def reset_state(self):
        '''reset RNN states'''
        g_sess.run([self.op_init_states])

    def parameter_count(self):
        '''
        Returns: integer
        '''
        v_vars_li = tf.trainable_variables()
        return sum(
            reduce(int.__mul__, v.get_shape().as_list()) for v in v_vars_li)


def main():
    global g_args, g_model, g_dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name',
        default='UnamedExperiment',
        help='name of experiment, affects checkpoint saves')
    parser.add_argument('-m', '--mode',
        default='train', help='Mode, "train", "test", "demo" or "interactive"')
    parser.add_argument('-i', '--input-pfile',
        help='path to input model parameter file')
    parser.add_argument('-o', '--output-pfile',
        help='path to output model parameters file')
    parser.add_argument('-ne', '--num-epoch',
        type=int, default=10, help='number of training epoch')
    parser.add_argument('--no-save-on-epoch',
        action='store_true', help="don't save parameter after each epoch")
    parser.add_argument('--no-test-on-epoch',
        action='store_true', help="don't sweep test set after training epoch")
    parser.add_argument('-if', '--input-file',
        help='input WAV file for "demo" mode')
    g_args = parser.parse_args()

    # TODO manage device
    stdout.write('Preparing dataset "%s" ... ' % hparams.DATASET_TYPE)
    stdout.flush()
    g_dataset = hparams.get_dataset()()
    g_dataset.install_and_load()
    stdout.write('done\n')
    stdout.flush()

    print('Separator type: "%s"' % hparams.SEPARATOR_TYPE)
    print('Recognizer type: "%s"' % (hparams.RECOGNIZER_TYPE if hparams.USE_ASR else "<not used>"))
    print('Discriminator type: "%s"' % hparams.DISCRIMINATOR_TYPE)

    stdout.write('Building model ... ')
    stdout.flush()
    g_model = Model(name=g_args.name)
    g_model.build()
    stdout.write('done\n')

    if g_args.input_pfile is not None:
        stdout.write('Loading paramters from %s ... ' % g_args.input_pfile)
        g_model.load_params(g_args.input_pfile)
        stdout.write('done\n')
    stdout.flush()
    g_model.reset()

    if g_args.mode == 'interactive':
        print('Now in interactive mode, you should run this with python -i')
        return
    elif g_args.mode == 'train':
        g_model.train(n_epoch=g_args.num_epoch, dataset=g_dataset)
        if g_args.output_pfile is not None:
            stdout.write('Saving parameters into %s ... ' % g_args.output_pfile)
            stdout.flush()
            g_model.save_params(g_args.output_pfile)
            stdout.write('done\n')
            stdout.flush()
    elif g_args.mode == 'test':
        g_model.test(dataset=g_dataset)
    elif g_args.mode == 'demo':
        # prepare data point
        if g_args.input_file is None:
            filename = 'demo.wav'
            for features in g_dataset.epoch('test', hparams.MAX_N_SIGNAL):
                break
            save_wavfile(filename, features[0][0] + features[0][1])
            features = np.sum(features[0], axis=0, keepdims=True)
        else:
            filename = g_args.input_file
            features = load_wavfile(g_args.input_file)

        # run with inference mode and save results
        # TODO this has to use whole BATCH_SIZE, inefficient !
        data_pt = (np.tile(features, [hparams.BATCH_SIZE] + [1]*2),)
        result = g_sess.run(
            g_model.infer_fetches,
            dict(zip(g_model.infer_feed_keys, data_pt + (hparams.DROPOUT_KEEP_PROB,))))
        signals = result['signals'][:(hparams.MAX_N_SIGNAL+1)]
        filename, fileext = os.path.splitext(filename)
        for i, s in enumerate(signals):
            save_wavfile(
                filename + ('_separated_%d' % (i+1)) + fileext, s)

        if hparams.USE_ASR:
            texts = result['texts'][0]
            for i, t in enumerate(texts):
                with open(filename + ('_text_%d' % (i+1)) + '.txt', 'w') as f:
                    f.write(g_dataset.decode_to_str(t))
                    f.write('\n')
    else:
        raise ValueError(
            'Unknown mode "%s"' % g_args.mode)


def debug_test():
    stdout.write('Building model ... ')
    g_model = Model()
    g_model.build()
    stdout.write('done')
    stdout.flush()
    g_model.reset()


if __name__ == '__main__':
    main()
    # debug_test()
