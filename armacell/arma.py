from typing import Any, Optional, Tuple

import tensorflow as tf
from tensorflow import TensorShape
from tensorflow.python.keras.layers import AbstractRNNCell, RNN


class ArmaCell(AbstractRNNCell):
    def __init__(
        self,
        q: int,
        input_dim: Tuple[int, int],
        p: Optional[int] = None,
        units: int = 1,
        activation: str = "linear",
        use_bias: bool = False,
        return_lags: bool = False,
        **kwargs: Any
    ):
        self.units = units
        self.activation = tf.keras.activations.deserialize(activation)
        self.q = q
        self.p = p if p is not None else input_dim[1]
        self.k = input_dim[0]
        assert self.p <= input_dim[1]
        assert self.p > 0
        assert self.q > 0
        assert self.k > 0
        self.q_overhang = self.q > self.p
        self.use_bias = use_bias
        self.return_lags = return_lags

        # Set during build()
        self.kernel = None
        self.recurrent_kernel = None
        self.bias = None
        super(ArmaCell, self).__init__(**kwargs)

    @property
    def state_size(self) -> TensorShape:
        return TensorShape((self.k * self.units, self.q))

    @property
    def output_size(self) -> TensorShape:
        return TensorShape((self.k * self.units, self.q)) if self.return_lags else TensorShape((self.k * self.units, 1))

    def build(self, input_shape: Tuple[int]) -> None:

        self.kernel = self.add_weight(shape=(self.p, self.units, self.k, self.k), initializer="uniform", name="kernel")
        self.recurrent_kernel = self.add_weight(
            shape=(self.q, self.units, self.k, self.k),
            initializer="uniform",
            name="recurrent_kernel",
        )

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.k * self.units), name="bias", initializer="zeros")
        else:
            self.bias = None
        self.built = True

    def call(self, inputs: tf.Tensor, states: Tuple[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        # Input:   BATCH x k x max(p,q)
        # Output:  BATCH x (k*units) x 1      if return_lags = False
        #          BATCH x (k*units) x q      if return_lags = True

        # STATE:
        # BATCH x (k*units) x q -> BATCH x k x units x q
        input_state = states[0]
        input_state = tf.expand_dims(input_state, axis=-2)
        input_state = tf.reshape(input_state, (-1, self.k, self.units, self.q))

        # AR
        ar_out = []
        for i in range(self.p):
            ar_out.append(inputs[:, :, i] @ self.kernel[i, :, :, :])  # type: ignore
        ar = tf.transpose(sum(ar_out), [1, 2, 0])  # BATCH x k x units
        ar = tf.reshape(ar, (-1, self.k * self.units))  # BATCH x (k * units)

        # MA
        ma_out = []
        for i in range(self.q):
            ma_unit = []
            if i + 1 > self.p:
                lhs = input_state - tf.expand_dims(inputs, axis=-2)
            else:
                lhs = input_state

            for j in range(self.units):
                ma_unit.append(lhs[:, :, j, i] @ self.recurrent_kernel[i, j, :, :])  # type: ignore
            ma_out.append(tf.stack(ma_unit, axis=-1))
        ma = sum(ma_out)
        ma = tf.reshape(ma, (-1, self.k * self.units))

        output = ar + ma

        if self.use_bias:
            output = output + self.bias

        output = self.activation(output)
        output = tf.expand_dims(output, axis=-1)
        output_state = tf.concat([output, states[0][:, :, :-1]], axis=-1)

        if self.return_lags:
            return output_state, output_state
        else:
            return output, output_state


class ARMA(RNN):
    def __init__(
            self,
            q: int,
            input_dim: Tuple[int, int],
            p: Optional[int] = None,
            units: int = 1,
            activation: str = "linear",
            use_bias: bool = False,
            return_lags: bool = False,
            return_sequences: bool = False,
            **kwargs: Any
    ) -> None:
        cell = ArmaCell(
            q, input_dim, p, units, activation, use_bias, return_lags, **kwargs
        )
        super().__init__(cell, return_sequences)

