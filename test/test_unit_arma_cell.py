import numpy as np
import tensorflow as tf

from armacell.arma import ArmaCell


def test_AR() -> None:
    k = 4
    p = 3
    batch_size = 5
    cell = ArmaCell(2, (k, p))
    kernel = (
        np.arange(int(np.prod((cell.p, cell.units, cell.k, cell.k))))
        .reshape((cell.p, cell.units, cell.k, cell.k))
        .astype(float)
    )
    cell.kernel = tf.convert_to_tensor(kernel)
    recurrent_kernel = np.zeros((cell.q, cell.units, cell.k, cell.k)).astype(float)
    cell.recurrent_kernel = tf.convert_to_tensor(recurrent_kernel)
    inputs = tf.convert_to_tensor(np.ones((batch_size, k, p)).astype(float))
    state = (
        tf.convert_to_tensor(np.zeros((batch_size, *cell.state_size)).astype(float)),
    )
    res = cell.call(inputs, state)
    ar_output = np.array(res[0])
    expected_output = np.apply_over_axes(np.sum, kernel, [0, 1, 2])
    assert np.all(ar_output[0, :, 0] == expected_output.flatten())


def test_AR_multiple_units() -> None:
    k = 4
    p = 3
    batch_size = 5
    cell = ArmaCell(2, (k, p), units=2)
    kernel = (
        np.arange(int(np.prod((cell.p, 1, cell.k, cell.k))))
        .reshape((cell.p, 1, cell.k, cell.k))
        .astype(float)
    )
    kernel = np.tile(kernel, [1, 2, 1, 1])
    kernel[:, 1, :, :] = 2 * kernel[:, 1, :, :]
    cell.kernel = tf.convert_to_tensor(kernel)
    recurrent_kernel = np.zeros((cell.q, cell.units, cell.k, cell.k)).astype(float)
    cell.recurrent_kernel = tf.convert_to_tensor(recurrent_kernel)
    inputs = tf.convert_to_tensor(np.ones((batch_size, k, p)).astype(float))
    state = (
        tf.convert_to_tensor(np.zeros((batch_size, *cell.state_size)).astype(float)),
    )
    res = cell.call(inputs, state)
    ar_output = np.array(res[0])
    assert ((ar_output[0, 1::2, 0] / ar_output[0, 0::2, 0]) == 2).all()


def test_different_batches() -> None:
    k = 4
    p = 3
    batch_size = 5
    cell = ArmaCell(2, (k, p))
    kernel = (
        np.arange(int(np.prod((cell.p, cell.units, cell.k, cell.k))))
        .reshape((cell.p, cell.units, cell.k, cell.k))
        .astype(float)
    )
    cell.kernel = tf.convert_to_tensor(kernel)
    recurrent_kernel = np.zeros((cell.q, cell.units, cell.k, cell.k)).astype(float)
    cell.recurrent_kernel = tf.convert_to_tensor(recurrent_kernel)
    input_array = np.ones((batch_size, k, p)).astype(float)
    for i in range(batch_size):
        input_array[i,] = (
            i + 1
        )
    inputs = tf.convert_to_tensor(input_array)
    state = (
        tf.convert_to_tensor(np.zeros((batch_size, *cell.state_size)).astype(float)),
    )
    res = cell.call(inputs, state)
    output = np.array(res[0])
    multiples = np.apply_over_axes(np.sum, output, [1, 2]).flatten()
    multiples = multiples / multiples.min()
    assert (multiples == np.array([1, 2, 3, 4, 5])).all()


def test_MA() -> None:
    k = 4
    p = 3
    batch_size = 5
    cell = ArmaCell(2, (k, p))
    kernel = np.zeros((cell.p, cell.units, cell.k, cell.k)).astype(float)
    cell.kernel = tf.convert_to_tensor(kernel)
    recurrent_kernel = (
        np.arange(int(np.prod((cell.q, cell.units, cell.k, cell.k))))
        .reshape((cell.q, cell.units, cell.k, cell.k))
        .astype(float)
    )
    cell.recurrent_kernel = tf.convert_to_tensor(recurrent_kernel)
    inputs = tf.convert_to_tensor(np.zeros((batch_size, k, p)).astype(float))
    state = (
        tf.convert_to_tensor(np.ones((batch_size, *cell.state_size)).astype(float)),
    )
    res = cell.call(inputs, state)
    ma_output = np.array(res[0])
    expected_output = np.apply_over_axes(np.sum, recurrent_kernel, [0, 1, 2])
    assert np.all(ma_output[0, :, 0] == expected_output)


def test_MA_multiple_units() -> None:
    k = 4
    p = 3
    batch_size = 5
    cell = ArmaCell(2, (k, p), units=2)
    kernel = np.zeros((cell.p, cell.units, cell.k, cell.k)).astype(float)
    cell.kernel = tf.convert_to_tensor(kernel)
    recurrent_kernel = (
        np.arange(int(np.prod((cell.q, 1, cell.k, cell.k))))
        .reshape((cell.q, 1, cell.k, cell.k))
        .astype(float)
    )
    recurrent_kernel = np.tile(recurrent_kernel, [1, 2, 1, 1])
    recurrent_kernel[:, 1, :, :] = 2 * recurrent_kernel[:, 1, :, :]
    cell.recurrent_kernel = tf.convert_to_tensor(recurrent_kernel)
    inputs = tf.convert_to_tensor(np.zeros((batch_size, k, p)).astype(float))
    state = (
        tf.convert_to_tensor(np.ones((batch_size, *cell.state_size)).astype(float)),
    )
    res = cell.call(inputs, state)
    ma_output = np.array(res[0])
    assert ((ma_output[0, 1::2, 0] / ma_output[0, 0::2, 0]) == 2).all()


def test_states() -> None:
    k = 4
    p = 3
    q = 3
    batch_size = 5
    cell = ArmaCell(q, (k, p))
    kernel = (
        np.arange(int(np.prod((cell.p, cell.units, cell.k, cell.k))))
        .reshape((cell.p, cell.units, cell.k, cell.k))
        .astype(float)
    )
    cell.kernel = tf.convert_to_tensor(kernel)
    recurrent_kernel = (
        np.arange(int(np.prod((cell.q, cell.units, cell.k, cell.k))))
        .reshape((cell.q, cell.units, cell.k, cell.k))
        .astype(float)
    )
    cell.recurrent_kernel = tf.convert_to_tensor(recurrent_kernel)
    inputs = tf.convert_to_tensor(np.ones((batch_size, k, p)).astype(float))
    state = (
        tf.convert_to_tensor(np.zeros((batch_size, *cell.state_size)).astype(float)),
    )
    _, output_state = cell.call(inputs, state)
    _, output_state2 = cell.call(inputs, (output_state,))
    _, output_state3 = cell.call(inputs, (output_state2,))
    assert (np.array(output_state[0, :, 0]) == np.array(output_state2[0, :, 1])).all()
    assert (np.array(output_state[0, :, 0]) == np.array(output_state3[0, :, 2])).all()
    assert (
        np.array(output_state3[0, :, 0])
        == np.array([263688.0, 300756.0, 337824.0, 374892.0])
    ).all()


def test_return_state() -> None:
    k = 1
    p = 3
    q = 3
    batch_size = 5
    cell = ArmaCell(q, (k, p), return_lags=True)
    kernel = (
        np.arange(int(np.prod((cell.p, cell.units, cell.k, cell.k))))
        .reshape((cell.p, cell.units, cell.k, cell.k))
        .astype(float)
    )
    cell.kernel = tf.convert_to_tensor(kernel)
    recurrent_kernel = (
        np.arange(int(np.prod((cell.q, cell.units, cell.k, cell.k))))
        .reshape((cell.q, cell.units, cell.k, cell.k))
        .astype(float)
    )
    cell.recurrent_kernel = tf.convert_to_tensor(recurrent_kernel)
    inputs = tf.convert_to_tensor(np.ones((batch_size, k, p)).astype(float))
    state = (
        tf.convert_to_tensor(np.zeros((batch_size, *cell.state_size)).astype(float)),
    )
    output, output_state = cell.call(inputs, state)
    assert (np.array(output) == np.array(output_state)).all()


def test_nonlinear_activation() -> None:
    k = 4
    p = 3
    batch_size = 5
    cell = ArmaCell(2, (k, p), activation="tanh")
    kernel = (
        np.arange(int(np.prod((cell.p, cell.units, cell.k, cell.k))))
        .reshape((cell.p, cell.units, cell.k, cell.k))
        .astype(float)
    )
    cell.kernel = tf.convert_to_tensor(kernel)
    recurrent_kernel = np.zeros((cell.q, cell.units, cell.k, cell.k)).astype(float)
    cell.recurrent_kernel = tf.convert_to_tensor(recurrent_kernel)
    inputs = tf.convert_to_tensor(np.ones((batch_size, k, p)).astype(float))
    state = (
        tf.convert_to_tensor(np.zeros((batch_size, *cell.state_size)).astype(float)),
    )
    res = cell.call(inputs, state)
    ar_output = np.array(res[0])
    expected_output = np.tanh(np.apply_over_axes(np.sum, kernel, [0, 1, 2]))
    assert np.all(ar_output[0, :, 0] == expected_output)
