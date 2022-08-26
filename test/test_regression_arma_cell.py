from typing import Any

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
from tensorflow import keras

from armacell import ARMA
from armacell.helpers import (
    SaveWeights,
    prepare_arma_input,
    restore_arma_parameters,
    set_all_seeds,
    simulate_arma_process,
    simulate_varma_process,
)
from armacell.plotting import plot_convergence


def test_ARMA_1_1() -> None:
    set_all_seeds()
    arparams = np.array([0.1])
    maparams = np.array([-0.4])
    run_p_q_test(arparams, maparams)


def test_ARMA_2_1() -> None:
    set_all_seeds()
    arparams = np.array([0.1, 0.3])
    maparams = np.array([-0.4])
    run_p_q_test(arparams, maparams)


def test_ARMA_2_2() -> None:
    set_all_seeds()
    arparams = np.array([0.1, 0.3])
    maparams = np.array([-0.4, -0.2])
    run_p_q_test(arparams, maparams)


def test_ARMA_1_2() -> None:
    set_all_seeds()
    arparams = np.array([0.3])
    maparams = np.array([-0.4, -0.2])
    run_p_q_test(arparams, maparams, p=1)


def test_ARMA_1_1_bias() -> None:
    set_all_seeds()
    arparams = np.array([0.1])
    maparams = np.array([-0.4])
    alpha = 0.2
    run_p_q_test(arparams, maparams, alpha)


def test_ARMA_1_2_bias() -> None:
    set_all_seeds()
    arparams = np.array([0.1])
    maparams = np.array([-0.4, -0.2])
    alpha = 0.2
    run_p_q_test(arparams, maparams, alpha, p=1)


def test_ARMA_2_2_multi_unit() -> None:
    set_all_seeds()
    arparams = np.array([0.1, 0.3])
    maparams = np.array([-0.4, -0.2])
    p = len(arparams)
    q = len(maparams)

    y = simulate_arma_process(arparams, maparams, 0, n_steps=25000, std=2)

    arima_model = ARIMA(endog=y, order=(p, 0, q), trend="n").fit()  # order = (p,d,q)

    X_train, y_train = prepare_arma_input(max(p, q), y)
    Y_train = np.stack([y_train, y_train], axis=-1)
    tf_model = get_trained_ARMA_p_q_model(
        q, X_train, Y_train, units=2, plot_training=True
    )

    raw_ar_weights = tf_model.get_weights()[0]
    raw_ma_weights = tf_model.get_weights()[1]
    weights1 = [raw_ar_weights[:, 0, :, :], raw_ma_weights[:, 0, :, :]]
    weights2 = [raw_ar_weights[:, 1, :, :], raw_ma_weights[:, 1, :, :]]
    beta1, gamma1, _ = restore_arma_parameters(weights1, p)
    beta2, gamma2, _ = restore_arma_parameters(weights2, p)

    # Target 1
    assert np.all(np.abs(beta1 - arima_model.arparams) < 0.05)
    assert np.all(np.abs(gamma1 - arima_model.maparams) < 0.05)

    # Target 2
    assert np.all(np.abs(beta2 - arima_model.arparams) < 0.05)
    assert np.all(np.abs(gamma2 - arima_model.maparams) < 0.05)


def test_VARMA_1_1_2() -> None:
    set_all_seeds()
    VAR = np.array([[0.1, -0.2], [0.0, 0.1]])
    VAR = np.expand_dims(VAR, axis=-1)
    VMA = np.array([[-0.4, 0.2], [0.0, -0.4]])
    VMA = np.expand_dims(VMA, axis=-1)
    alpha = np.zeros(2)
    y = simulate_varma_process(VAR, VMA, alpha, n_steps=10000)
    p = 1
    q = 1

    varma_model = VARMAX(y, order=(1, 1)).fit()

    X_train, y_train = prepare_arma_input(max(p, q), y)
    tf_model = get_trained_ARMA_p_q_model(q, X_train, y_train)
    gamma = -tf_model.get_weights()[1][0, 0].T
    beta = tf_model.get_weights()[0][0, 0].T - gamma

    assert np.all(np.abs(beta - varma_model.coefficient_matrices_var[0]) < 0.05)
    assert np.all(np.abs(gamma - varma_model.coefficient_matrices_vma[0]) < 0.05)


def run_p_q_test(
    arparams: np.array,
    maparams: np.array,
    alpha_true: float = 0,
    plot_training: bool = False,
    **kwargs: int
) -> None:
    p = len(arparams)
    q = len(maparams)
    add_intercept = alpha_true != 0

    y = simulate_arma_process(arparams, maparams, alpha_true, n_steps=25000, std=2)

    arima_model = ARIMA(
        endog=y, order=(p, 0, q), trend="c" if add_intercept else "n"
    ).fit()  # order = (p,d,q)

    X_train, y_train = prepare_arma_input(max(p, q), y)
    tf_model = get_trained_ARMA_p_q_model(
        q, X_train, y_train, add_intercept, plot_training, **kwargs
    )

    if plot_training:
        plot_convergence(tf_model, p, add_intercept, arima_model)

    beta, gamma, alpha = restore_arma_parameters(
        tf_model.get_weights(), p, add_intercept
    )
    assert np.all(np.abs(beta - arima_model.arparams) < 0.05)
    assert np.all(np.abs(gamma - arima_model.maparams) < 0.05)
    if add_intercept:
        assert np.all(np.abs(alpha - arima_model.params[0]) < 0.05)


def get_trained_ARMA_p_q_model(
    q: int,
    X_train: np.array,
    y_train: np.array,
    add_intercept: bool = False,
    plot_training: bool = False,
    **kwargs: Any
) -> keras.Sequential:
    tf_model = keras.Sequential(
        [
            ARMA(q=q, input_dim=X_train.shape[2:], use_bias=add_intercept, **kwargs),
        ]
    )
    tf_model.compile(loss="mse", optimizer="adam", run_eagerly=False)
    callback = [SaveWeights()] if plot_training else []
    tf_model.fit(
        X_train, y_train, epochs=100, verbose=True, batch_size=200, callbacks=callback
    )
    return tf_model
