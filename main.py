'''
TensorFlow Implementation of "GAN for Single Source Audio Separation"

TODO docs
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from sys import stdout
import os
import pdb

import numpy as np
import tensorflow as tf

from app import *


# Global vars
g_sess = tf.Session()


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
            s_gan_loss = tf.reduce_mean(tf.square(s_guess - s_truth), axis=None)

        # prepare summary
        # TODO add impl & summary for word error rate
        # TODO add impl & summary for SNR

        with tf.name_scope('summary'):
            tf.summary.scalar('similarity', s_dissim_loss)
            tf.summary.scalar('gan_loss', s_gan_loss)


        # apply optimizer
        ozer = hparams.get_optimizer()(learn_rate=hparams.LR)

        v_params_li = tf.trainable_variables()
        s_gparams_li = [v for v in v_params_li if v.name.startswith('G/')]
        s_dparams_li = [v for v in v_params_li if v.name.startswith('D/')]

        op_fit_generator = ozer.minimize(
            s_dissim_loss - s_gan_loss, var_list=s_gparams_li)
        op_fit_discriminator = ozer.minimize(
            s_gan_loss, var_list=s_dparams_li)
        self.op_init_params = tf.variables_initializer(v_params_li)

        self.all_summary = tf.summary.merge_all()
        self.feed_keys = [s_source_signals, s_source_texts]
        self.train_fetches = [
            self.all_summary, op_fit_generator, op_fit_discriminator]

        self.infer_fetches = [dict(
            signals=s_extracted_signals,
            texts=s_predicted_texts)]


    def train(self, n_epoch, dataset=None):
        train_writer = tf.summary.FileWriter(hparams.SUMMARY_DIR, g_sess.graph)
        for i_epoch in range(n_epoch):
            for data_pt in dataset.epoch('train', hparams.BATCH_SIZE):
                to_feed = dict(zip(self.feed_keys, data_pt))
                step_summary = g_sess.run(self.train_fetches, to_feed)[0]
                train_writer.add_summary(step_summary)
                stdout.write('.')
                stdout.flush()
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
