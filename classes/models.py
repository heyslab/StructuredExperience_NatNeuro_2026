import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.keras import backend
import keras.saving
from keras.src import ops
import numpy as np

class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


@keras.saving.register_keras_serializable(package='leaky')
class LeakyRNNCell(tf.keras.layers.SimpleRNNCell):
    def __init__(self, *args, **kwargs):
        self._gamma = kwargs.pop('gamma')
        self._noise_std = kwargs.pop('noise_std')
        super(self.__class__, self).__init__(*args, **kwargs)

    def _call_noisy(self, sequence, states, training=None):
        prev_output = states[0] if isinstance(states, (list, tuple)) else states
        dp_mask = self.get_dropout_mask(sequence)
        rec_dp_mask = self.get_recurrent_dropout_mask(prev_output)

        if training and dp_mask is not None:
            sequence = sequence * dp_mask
        h = ops.matmul(sequence, self.kernel)
        if self.bias is not None:
            h += self.bias

        if self._noise_std is not None:
            h = tf.math.add(
                h, tf.random.normal((self.units,), mean=0, stddev=self._noise_std))

        if training and rec_dp_mask is not None:
            prev_output = prev_output * rec_dp_mask
        output = h + ops.matmul(prev_output, self.recurrent_kernel)
        if self.activation is not None:
            output = self.activation(output)

        new_state = [output] if isinstance(states, (list, tuple)) else output
        return output, new_state

    def call(self, inputs, states, training=None):
        prev_output = states[0] if isinstance(states, (list, tuple)) else states
        output, _ = self._call_noisy(inputs, states, training=training)
        output = (1 - self._gamma) * prev_output + self._gamma * output
        new_state = [output] if isinstance(states, (list, tuple)) else output
        return output, new_state

    def get_config(self):
        config = super(self.__class__, self).get_config()
        config.update({'gamma': self._gamma, 'noise_std': self._noise_std})
        return config


@keras.saving.register_keras_serializable(package='leaky')
class LeakyRNN(tf.keras.layers.SimpleRNN):
    def __init__(self,
            units,
            activation="tanh",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal",
            bias_initializer="zeros",
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            dropout=0.0,
            recurrent_dropout=0.0,
            return_sequences=False,
            return_state=False,
            go_backwards=False,
            stateful=False,
            unroll=False,
            seed=None,
            gamma=None,
            noise_std=None,
            **kwargs,
            ):

        cell = LeakyRNNCell(
                units,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                recurrent_initializer=recurrent_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                recurrent_regularizer=recurrent_regularizer,
                bias_regularizer=bias_regularizer,
                kernel_constraint=kernel_constraint,
                recurrent_constraint=recurrent_constraint,
                bias_constraint=bias_constraint,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                seed=seed,
                dtype=kwargs.get("dtype", None),
                trainable=kwargs.get("trainable", True),
                name="leaky_rnn_cell",
                gamma=gamma,
                noise_std=noise_std
                )

        super(tf.keras.layers.SimpleRNN, self).__init__(
                cell,
                return_sequences=return_sequences,
                return_state=return_state,
                go_backwards=go_backwards,
                stateful=stateful,
                unroll=unroll,
                **kwargs,
        )

        self._gamma = gamma
        self._noise_std = noise_std
        self.input_spec = [tf.keras.layers.InputSpec(ndim=3)]

    def get_config(self):
        config = super(self.__class__, self).get_config()
        config.update({'gamma': self._gamma, 'noise_std': self._noise_std})
        return config

    def build_from_config(self, config):
        """Builds the layer's states with the supplied config dict.

        By default, this method calls the `build(config["input_shape"])` method,
        which creates weights based on the layer's input shape in the supplied
        config. If your config contains other information needed to load the
        layer's state, you should override this method.

        Args:
            config: Dict containing the input shape associated with this layer.
        """
        input_shape = config["input_shape"]
        if input_shape is not None:
            self.build(input_shape)


