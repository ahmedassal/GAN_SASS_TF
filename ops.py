'''
collection of commonly used Ops and layers
'''

import tensorflow as tf

from hparams import FLOATX


def lyr_linear(
        name, s_x,
        odim,
        axis=-1, bias=True, w_init=None, b_init=None):
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
    '''
    assert isinstance(odim, int)
    idim = s_x.get_shape().as_list()[axis]
    assert isinstance(idim, int)
    with tf.name_scope(name, reuse=True):
        v_w = tf.get_variable(
            'W', [idim, odim],
            initializer=w_init,
            dtype=FLOATX)
        s_y = tf.tensordot(s_x, v_w, [[axis], [0]])
        if bias:
            v_b = tf.get_variable(
                    'B', [idim, odim],
                    initializer=b_init,
                    dtype=FLOATX)
            s_y = s_y + v_b
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
