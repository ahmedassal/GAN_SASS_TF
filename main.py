'''
TensorFlow Implementation of "GAN for Single Source Audio Separation"

TODO docs
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from sys import stdout
import os

import numpy as np
import tensorflow as tf

import app.hparams as hparams
import app.ops as ops
import app.ozers as ozers
import app.datasets as datasets
import app.modules as modules


# Global vars
g_sess = tf.Session()

def _dict_add(dst, src):
    for k,v in src.items():
        if k not in dst:
            dst[k] = v
        else:
            dst[k] += v

def _dict_format(di):
    return ' '.join('='.join((k, str(v))) for k,v in di.items())


class Model(object):
    '''
    base class for a full trainable model
    '''
    def __init__(self, name='BaseModel'):
        self.name = name
        self.s_states_di = {}

    def lyr_lstm(self, name, s_x, hdim, axis=-1, t_axis=0, op_linear=ops.lyr_linear, reuse=False):
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
        with tf.name_scope(name):
            zero_init = tf.constant_initializer(0.)
            v_cell = tf.get_variable(
                dtype=hparams.FLOATX,
                shape=h_shp, name=name+'_cell',
                trainable=False,
                initializer=zero_init)
            v_hid = tf.get_variable(
                dtype=hparams.FLOATX,
                shape=h_shp, name=name+'_hid',
                trainable=False,
                initializer=zero_init)
            self.s_states_di[v_cell.name] = v_cell
            self.s_states_di[v_hid.name] = v_hid

            op_lstm = lambda _h, _x: ops.lyr_lstm_flat(
                name=name+'_LSTM',
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

        input_shape = [
            hparams.BATCH_SIZE,
            hparams.MAX_N_SIGNAL,
            None,
            hparams.FFT_SIZE]
        input_text_shape = [
            hparams.BATCH_SIZE,
            hparams.MAX_N_SIGNAL,
            None]
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
            s_mixed_signals = tf.reduce_sum(s_source_signals, axis=1)
            s_noise_signal = tf.random_normal(
                tf.shape(s_mixed_signals),
                stddev=0.1,
                dtype=hparams.FLOATX)
            s_mixed_signals += s_noise_signal
            s_separated_signals = separator(s_mixed_signals)
            s_predicted_texts = recognizer(s_separated_signals)
            s_autoencoder_loss = tf.reduce_mean(
                tf.square(
                    tf.reduce_sum(s_separated_signals, axis=1) - s_mixed_signals),
                axis=None)

        with tf.name_scope('D'):
            s_truth_signals = tf.concat(
                [s_source_signals, tf.expand_dims(s_noise_signal, 1)], axis=1)
            # TODO use non-zero const padding once the feature is up
            pad_const = 1. / hparams.CHARSET_SIZE
            s_truth_texts = tf.pad(
                s_texts - pad_const,
                ((0,0), (0,1), (0,0), (0,0)), mode='CONSTANT') + pad_const
            s_guess_t = discriminator(s_truth_signals, s_truth_texts)
            s_guess_f = discriminator(
                s_separated_signals,
                tf.nn.softmax(s_predicted_texts))
            s_truth = tf.concat([
                tf.constant(
                    hparams.CLS_REAL_SIGNAL,
                    hparams.INTX,
                    [1, hparams.MAX_N_SIGNAL]),
                tf.constant(
                    hparams.CLS_REAL_NOISE,
                    hparams.INTX,
                    [1, 1])], axis=-1)
            s_lies = tf.constant(
                    hparams.CLS_FAKE_SIGNAL,
                    hparams.INTX,
                    [1, hparams.MAX_N_SIGNAL+1])
            s_truth = tf.one_hot(s_truth, 3)
            s_lies = tf.one_hot(s_lies, 3)
            # TODO optimize log(softmax(...))
            s_gan_loss_t = tf.reduce_sum(
                - tf.log(tf.nn.softmax(s_guess_t) + hparams.EPS) * s_truth,
                axis=None)
            s_gan_loss_f = tf.reduce_sum(
                - tf.log(tf.nn.softmax(s_guess_f) + hparams.EPS) * s_lies,
                axis=None)
            s_gan_loss = (s_gan_loss_t + s_gan_loss_f) / hparams.BATCH_SIZE

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
        self.feed_keys = [s_source_signals, s_source_texts]
        self.train_fetches = [
            self.all_summary,
            dict(gan_loss=s_gan_loss, ae_loss=s_autoencoder_loss),
            op_fit_generator, op_fit_discriminator]

        self.infer_fetches = [dict(
            signals=s_separated_signals,
            texts=s_predicted_texts)]

        # FOR DEBUGGING
        # g_sess.run([self.op_init_params])
        # _feed = {
            # s_source_signals : np.random.rand(4, 3, 128, 256),
            # s_source_texts : np.random.randint(0, hparams.CHARSET_SIZE, (4,3,32)) }
        # ret = g_sess.run([s_gan_loss], _feed)


    def train(self, n_epoch, dataset=None):
        train_writer = tf.summary.FileWriter(hparams.SUMMARY_DIR, g_sess.graph)
        for i_epoch in range(n_epoch):
            cli_report = {}
            for data_pt in dataset.epoch('train', hparams.BATCH_SIZE):
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


def main():
    # TODO parse cmd args
    # TODO manage device
    print('Preparing dataset ... ', end='')
    stdout.flush()
    dataset = hparams.get_dataset()()
    dataset.install_and_load()
    print('done')
    stdout.flush()

    print('Building model ... ', end='')
    model = Model()
    model.build()
    print('done')
    stdout.flush()
    model.reset()
    model.train(n_epoch=10, dataset=dataset)

    # TODO inference

def debug_test():
    # REMOVEME
    mdl = Model()
    s_x = tf.placeholder('float32', [17,5,6])
    s_y = mdl.lyr_lstm('RNN', s_x, 8)
    g_sess.run([tf.variables_initializer(tf.all_variables())])
    ret = g_sess.run(s_y, {s_x:np.random.rand(17,5,6)})
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    # main()
    debug_test()
