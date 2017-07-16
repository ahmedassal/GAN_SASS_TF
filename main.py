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
        # import pdb; pdb.set_trace()


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
        g_sess.run([self.op_init_params])


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


if __name__ == '__main__':
    main()
