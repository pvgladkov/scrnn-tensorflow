import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl
import collections

_linear = rnn_cell_impl._linear


_SCStateTuple = collections.namedtuple("SCStateTuple", ("c", "h"))


class SCStateTuple(_SCStateTuple):
    """Tuple used by SC Cells for `state_size`, `zero_state`, and output state.

    Stores two elements: `(c, h)`, in that order.

    Only used when `state_is_tuple=True`.
    """
    __slots__ = ()

    @property
    def dtype(self):
        (c, h) = self
        if c.dtype != h.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" % (str(c.dtype), str(h.dtype)))
        return c.dtype


class SCRNCell(RNNCell):

    """
    Tensor Flow port of Structurally Constrained Recurrent Neural Network model

    Links:

    Learning Longer Memory in Recurrent Neural Networks
    http://arxiv.org/abs/1412.7753

    Original implentation in Torch
    https://github.com/facebookarchive/SCRNNs
    """

    def __init__(self, num_units, context_units, alpha=None, reuse=None):
        super(SCRNCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._context_units = context_units
        self._alpha = alpha
        self._initializer = tf.glorot_uniform_initializer()
        self._batch_size = None
        self._input_size = None

    @property
    def state_size(self):
        return self._num_units + self._context_units

    @property
    def output_size(self):
        return self._num_units + self._context_units

    def call(self, inputs, state, **kwargs):

        self._batch_size = inputs.shape[0].value or array_ops.shape(inputs)[0]
        self._input_size = inputs.shape[1].value or array_ops.shape(inputs)[1]

        # todo make alpha trainable
        alpha = self._alpha

        state_h = array_ops.slice(
            state, begin=(0, 0),
            size=(self._batch_size, self._num_units))

        state_c = array_ops.slice(
            state, begin=(0, self._num_units),
            size=(self._batch_size, self._context_units))

        with vs.variable_scope(vs.get_variable_scope(), initializer=self._initializer):

            B = vs.get_variable(
                'B_matrix', shape=[self._input_size, self._context_units])

            V = vs.get_variable(
                'V_matrix', shape=[self._context_units, self._num_units])

            U = vs.get_variable(
                'U_matrix', shape=[self._num_units, self._num_units])

            # context_state.shape = (batch_size x context_units)
            context_state = (1 - alpha) * math_ops.matmul(inputs, B) + alpha * state_c

            # hidden_state.shape = (batch_size x num_units)
            hidden_state = math_ops.sigmoid(_linear([context_state, inputs, state_h], self._num_units, False))

            # output.shape = (batch_size x num_units)
            output = math_ops.matmul(hidden_state, U) + math_ops.matmul(context_state, V)

        new_state = array_ops.concat(
            values=[hidden_state, context_state], axis=1)

        return output, new_state
