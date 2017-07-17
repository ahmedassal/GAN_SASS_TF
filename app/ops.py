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
    assert isinstance(idim, int)
    with tf.variable_scope(name):
        v_w = tf.get_variable(
            'W', [idim, odim],
            initializer=w_init,
            dtype=hparams.FLOATX)
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
