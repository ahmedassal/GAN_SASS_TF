'''
collection of commonly used Ops and layers
'''

import tensorflow as tf

import app.hparams as hparams


def lyr_linear(
        name, s_x, odim,
        axis=-1, bias=True, w_init=None, b_init=None, reuse=False):
    '''
    Like tf.xw_plus_b, but works on arbitrary shape

    Args:
        name: string
        s_x: tensor variable
        odim: integer
        axis: integer
        bias: boolean, whether to use bias
        w_init: initializer for W
        b_init: initializer for B
        reuse: bool whether to reuse parameter variables
    '''
    assert isinstance(odim, int)
    x_shape = s_x.get_shape().as_list()
    idim = x_shape[axis]
    ndim = len(x_shape)
    assert -ndim <= axis < ndim
    assert isinstance(idim, int)
    with tf.variable_scope(name):
        v_w = tf.get_variable(
            'W', [idim, odim],
            initializer=w_init,
            dtype=hparams.FLOATX)
        if ndim == 1:
            s_y = tf.matmul(tf.expand_dims(s_x, 0), v_w)
            s_y = tf.squeeze(s_y, 0)
        elif ndim == 2:
            if axis % 2 == 1:
                s_y = tf.matmul(s_x, v_w)
            else:
                s_y = tf.matmul(tf.transpose(s_x), v_w)
                s_y = tf.transpose(s_x)
        else:
            s_y = tf.tensordot(s_x, v_w, [[axis], [0]])
        if bias:
            if b_init is None:
                b_init = tf.constant_initializer(1., dtype=hparams.FLOATX)
            v_b = tf.get_variable(
                    'B', [odim],
                    initializer=b_init,
                    dtype=hparams.FLOATX)
            s_b = tf.reshape(v_b, [odim] + [1] * (ndim - (axis % ndim) - 1))
            s_y = s_y + s_b
    return s_y


def relu(s_x, alpha=0.):
    '''
    Leaky relu. Same as relu when alpha is 0.

    Same as theano.tensor.nnet.relu

    Args:
        s_x: input
        alpha: float constant, default 0.
    '''
    if alpha == 0.:
        s_y = tf.nn.relu(s_x)
    else:
        s_y = tf.maximum(s_x*alpha, s_x)
    return s_y


def lyr_lstm_flat(name, s_x, v_cell, v_hid, axis=-1, op_linear=lyr_linear):
    '''
    Generic LSTM layer that works with arbitrary shape & linear operator

    Args:
        name: string
        s_x: symbolic tensor
        v_cell: tensor variable
        v_hid: tensor variable
        axis: integer, which axis to perform linear operation
        op_linear: linear operation

    Returns:
        (s_cell_tp1, s_hid_tp1)

    Notes:
        - It's a *flat* layer, which means it doesn't create state variable
        - The size of s_x along axis must be known
    '''
    idim = s_x.get_shape().as_list()[axis]
    assert idim is not None
    cell_shp = v_cell.get_shape().as_list()
    hid_shp = v_hid.get_shape().as_list()
    hdim = cell_shp[axis]
    assert hdim == hid_shp[axis]

    with tf.variable_scope(name):
        s_inp = tf.concat([s_x, v_hid], axis=axis)
        s_act = op_linear('linear', s_inp, hdim*4, axis=axis)
        s_cell_new, s_gates = tf.split(s_act, [hdim, hdim*3], axis=axis)
        s_cell_new = tf.tanh(s_cell_new)
        s_igate, s_fgate, s_ogate = tf.split(
            tf.nn.sigmoid(s_gates), 3, axis=axis)
        s_cell_tp1 = s_igate*s_cell_new + s_fgate*v_cell
        s_hid_tp1 = s_ogate * s_cell_tp1
    return (s_cell_tp1, s_hid_tp1)


def lyr_gru(name, s_x, v_cell, axis=-1, op_linear=lyr_linear):
    '''
    Generic GRU layer that works with arbitrary shape & linear operator

    Args:
        name: string
        s_x: symbolic tensor
        v_cell: tensor variable, state of GRU RNN
        axis: integer, which axis to perform linear operation
        op_linear: linear operation

    Returns:
        s_cell_tp1

    Notes:
        The size of s_x along axis must be known
    '''
    idim = s_x.get_shape().as_list()[axis]
    assert idim is not None
    cell_shp = v_cell.get_shape().as_list()
    hdim = cell_shp[axis]

    with tf.variable_scope(name):
        s_inp = tf.concat([s_x, v_cell], axis=axis)
        s_act = op_linear('gates', s_inp, hdim*2, axis=axis)
        s_rgate, s_igate = tf.split(tf.nn.sigmoid(s_act), 2, axis=axis)
        s_inp2 = tf.concat([s_x, v_cell * s_rgate], axis=axis)
        s_cell_new = op_linear('linear', s_inp2, hdim, axis=axis)
        s_cell_new = tf.tanh(s_cell_new)
        s_cell_tp1 = v_cell * s_igate + s_cell_new * (1.-s_igate)
    return s_cell_tp1


def batch_snr(clear_signal, noisy_signal):
    '''
    batched signal to noise raio, assuming zero mean

    Args:
        clear_signal: batched array
        noisy_signal: batched_array
    '''
    clear_signal_shp = clear_signal.get_shape().as_list()
    noisy_signal_shp = noisy_signal.get_shape().as_list()
    ndim = len(clear_signal_shp)
    reduce_axes = list(range(1, ndim))
    assert len(noisy_signal_shp) == ndim
    noise = clear_signal - noisy_signal

    if reduce_axes:
        signal_pwr = tf.reduce_mean(
            tf.square(clear_signal), axis=reduce_axes)
        noise_pwr = tf.reduce_mean(
            tf.square(noise), axis=reduce_axes)
    else:
        signal_pwr = tf.square(clear_signal)
        noise_pwr = tf.square(noise)

    coeff = 4.342944819
    return coeff * (tf.log(signal_pwr) - tf.log(noise_pwr))

def batch_cross_snr(clear_signal, noisy_signal):
    '''
    signal to noise raio, assuming zero mean

    Args:
        clear_signal: array of shape [batch_size, m, ...]
        noisy_signal: array of shape [batch_size, n, ...]

    Returns:
        array of shape [batch_size, m, n]
    '''
    clear_signal_shp = clear_signal.get_shape().as_list()
    noisy_signal_shp = noisy_signal.get_shape().as_list()
    ndim = len(clear_signal_shp)
    assert len(noisy_signal_shp) == ndim
    assert ndim >= 2


    clear_signal = tf.expand_dims(clear_signal, 2)  # [b, m, 1, ...]
    noisy_signal = tf.expand_dims(clear_signal, 1)  # [b, 1, n, ...]
    noise = clear_signal - noisy_signal
    reduce_axes = list(range(3, ndim+1))

    if reduce_axes:
        signal_pwr = tf.reduce_mean(
            tf.square(clear_signal), axis=reduce_axes)
        noise_pwr = tf.reduce_mean(
            tf.square(noise), axis=reduce_axes)
    else:
        signal_pwr = tf.square(clear_signal)
        noise_pwr = tf.square(noise)

    coeff = 4.342944819
    return coeff * (tf.log(signal_pwr) - tf.log(noise_pwr))
