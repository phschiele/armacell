from typing import Any, Optional, Tuple

import tensorflow as tf
from tensorflow import TensorShape
from tensorflow.python.keras import backend
from tensorflow.python.keras.layers import AbstractRNNCell, RNN, Layer
from tensorflow.python.ops import array_ops


class ConvARMACell(AbstractRNNCell):
    def __init__(
            self,
            q: int,
            image_shape: Tuple[Any, ...],
            units: int = 1,
            kernel_size: Tuple[int, int] = (3, 3),
            activation: str = "relu",
            use_bias: bool = True,
            return_lags: bool = True,
            **kwargs: dict
    ) -> None:
        self.units = units
        self.kernel_size = kernel_size
        self.activation = tf.keras.activations.deserialize(activation)
        self.image_shape = image_shape
        self.q = q
        self.use_bias = use_bias
        self.return_lags = return_lags

        # Set during build()
        self.kernel = None
        self.recurrent_kernel = None
        self.bias = None
        self.ar_parameters = None
        self.ma_parameters = None
        self.p = None
        self.arma_bias = None
        self.ar_conv_bias = None
        self.ma_conv_bias = None

        super(ConvARMACell, self).__init__(**kwargs)

    @property
    def state_size(self) -> TensorShape:
        return TensorShape((self.q, *self.image_shape[1:-1], self.units))

    @property
    def output_size(self) -> TensorShape:
        return TensorShape(
            (self.q, *self.image_shape[1:-1], self.units)) if self.return_lags else TensorShape(
            (*self.image_shape[:-1], self.units))

    def build(self, input_shape: Tuple[int, ...]) -> None:
        input_dim = input_shape[-1]  # channels last
        self.p = input_shape[-4]
        assert self.image_shape == tuple(input_shape[-4:]), (self.image_shape, input_shape[-4:])
        kernel_shape = self.kernel_size + (input_dim, self.p, self.units)
        self.kernel = self.add_weight(shape=kernel_shape, initializer="glorot_uniform", name="kernel")
        recurrent_kernel_shape = self.kernel_size + (self.units, self.q, self.units)
        self.recurrent_kernel = self.add_weight(shape=recurrent_kernel_shape, initializer="glorot_uniform",
                                                name="recurrent_kernel")

        self.ar_parameters = self.add_weight(shape=(self.units, self.p), initializer="glorot_uniform", name="ar")
        self.ma_parameters = self.add_weight(shape=(self.units, self.q), initializer="glorot_uniform", name="ma")

        if self.use_bias:
            self.arma_bias = self.add_weight(shape=(self.units,), initializer="zeros", name="arma_bias")
            self.ar_conv_bias = self.add_weight(shape=(self.p, self.units), name="ar_conv_bias", initializer="zeros")
            self.ma_conv_bias = self.add_weight(
                shape=(
                    self.q,
                    self.units,
                ),
                name="ma_conv_bias",
                initializer="zeros",
            )
        self.built = True

    def call(self, inputs: tf.Tensor, states: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

        input_states = states[0]
        splits_ar = array_ops.split(inputs, self.p, axis=-4)
        conv_outs_ar = []
        for i, split in enumerate(splits_ar):
            split = split[:, 0, :, :, :]
            conv_out = self.input_conv(split, self.kernel[:, :, :, i, :], self.ar_conv_bias[i, :])
            conv_outs_ar.append(conv_out)
        ar = tf.multiply(tf.stack(conv_outs_ar, axis=-1), self.ar_parameters)

        splits_ma = array_ops.split(input_states, self.q, axis=-4)
        conv_outs_ma = []
        for i, split in enumerate(splits_ma):
            split = split[:, 0, :, :, :]
            conv_out = self.input_conv(split, self.recurrent_kernel[:, :, :, i, :], self.ma_conv_bias[i, :])
            conv_outs_ma.append(conv_out)
        ma = tf.multiply(tf.stack(conv_outs_ma, axis=-1), self.ma_parameters)

        output = tf.reduce_sum(ar, axis=-1) + tf.reduce_sum(ma, axis=-1)
        if self.use_bias:
            output = output + self.arma_bias
        output = self.activation(output)
        output_states = tf.concat([tf.expand_dims(output, axis=1), input_states[:, :-1, :, :, :]], axis=1)

        if self.return_lags:
            return output_states, output_states
        else:
            return output, output_states


    def input_conv(self, x: tf.Tensor, w: tf.Tensor, b: Optional[tf.Tensor] = None, padding: str = "same") -> tf.Tensor:
        conv_out = backend.conv2d(x, w, strides=(1, 1), padding=padding, data_format="channels_last")
        if b is not None:
            conv_out = backend.bias_add(conv_out, b, data_format="channels_last")
        return conv_out

    def get_config(self) -> dict:
        config = {
            "q": self.q,
            "image_shape": self.image_shape,
            "kernel_size": self.kernel_size,
            "units": self.units,
            "activation": tf.keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
        }
        base_config = super(ConvARMACell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvARMA(RNN):
    def __init__(
            self,
            q: int,
            image_shape: Tuple[Any, ...],
            units: int = 1,
            kernel_size: Tuple[int, int] = (3, 3),
            activation: str = "relu",
            use_bias: bool = True,
            return_lags: bool = True,
            return_sequences: bool = False,
            **kwargs: dict
    ) -> None:
        cell = ConvARMACell(
            q, image_shape, units, kernel_size, activation, use_bias, return_lags, **kwargs
        )
        super().__init__(cell, return_sequences)


class SpatialDiffs(Layer):

    def __init__(self, shift_val: int):
        self.shift_val = shift_val
        super().__init__()

    def build(self, input_shape):
        pass

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        shift_val = self.shift_val
        diff_1 = tf.roll(inputs, shift_val, axis=-3) - inputs
        diff_1 = tf.concat([diff_1[:, :, :, :shift_val] * 0, diff_1[:, :, :, shift_val:]], axis=-3)

        diff_2 = tf.roll(inputs, -shift_val, axis=-3) - inputs
        diff_2 = tf.concat([diff_2[:, :, :, :shift_val], diff_2[:, :, :, shift_val:] * 0], axis=-3)

        diff_3 = tf.roll(inputs, shift_val, axis=-2) - inputs
        diff_3 = tf.concat([diff_3[:, :, :, :, :shift_val] * 0, diff_3[:, :, :, :, shift_val:]], axis=-2)

        diff_4 = tf.roll(inputs, -shift_val, axis=-2) - inputs
        diff_4 = tf.concat([diff_4[:, :, :, :, :shift_val], diff_4[:, :, :, :, shift_val:] * 0], axis=-2)

        inputs_and_diffs = tf.concat([inputs, diff_1, diff_2, diff_3, diff_4], axis=-1)
        return inputs_and_diffs
