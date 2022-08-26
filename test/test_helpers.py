import numpy as np

from armacell.helpers import (
    prepare_arma_input,
    restore_arma_parameters,
    simulate_varma_process,
)


def test_restore_arma_parameters_1_1() -> None:
    beta_breve = np.array([-0.3])
    gamma_breve = np.array([0.4])
    beta, gamma, _ = restore_arma_parameters([beta_breve, gamma_breve], 1)
    assert np.allclose(beta, np.array([0.1]))
    assert np.allclose(gamma, np.array([-0.4]))


def test_restore_arma_parameters_2_1() -> None:
    beta_breve = np.array([-0.3, 0.3])
    gamma_breve = np.array([0.4])
    beta, gamma, _ = restore_arma_parameters([beta_breve, gamma_breve], 2)
    assert np.allclose(beta, np.array([0.1, 0.3]))
    assert np.allclose(gamma, np.array([-0.4]))


def test_restore_arma_parameters_1_2() -> None:
    beta_breve = np.array([-0.3])
    gamma_breve = np.array([0.4, 0.2])
    beta, gamma, _ = restore_arma_parameters([beta_breve, gamma_breve], 1)
    assert np.allclose(beta, np.array([0.1]))
    assert np.allclose(gamma, np.array([-0.4, -0.2]))


def test_restore_arma_parameters_2_2() -> None:
    beta_breve = np.array([-0.3, 0.1])
    gamma_breve = np.array([0.4, 0.2])
    beta, gamma, _ = restore_arma_parameters([beta_breve, gamma_breve], 2)
    assert np.allclose(beta, np.array([0.1, 0.3]))
    assert np.allclose(gamma, np.array([-0.4, -0.2]))


def test_restore_arma_parameters_1_1_intercept() -> None:
    beta_breve = np.array([-0.3])
    gamma_breve = np.array([0.4])
    alpha = np.array([0.2])
    beta, gamma, alpha = restore_arma_parameters(
        [beta_breve, gamma_breve, alpha], 1, add_intercept=True
    )
    assert np.allclose(beta, np.array([0.1]))
    assert np.allclose(gamma, np.array([-0.4]))


def test_restore_arma_parameters_1_2_intercept() -> None:
    beta_breve = np.array([-0.3])
    gamma_breve = np.array([0.4, 0.2])
    alpha = np.array([0.2])
    beta, gamma, alpha = restore_arma_parameters(
        [beta_breve, gamma_breve, alpha], 1, add_intercept=True
    )
    assert np.allclose(beta, np.array([0.1]))
    assert np.allclose(gamma, np.array([-0.4, -0.2]))
    assert np.allclose(alpha, np.array([0.2]))


def test_restore_arma_parameters_2_2_intercept() -> None:
    beta_breve = np.array([-0.3, 0.1])
    gamma_breve = np.array([0.4, 0.2])
    alpha = np.array([0.2])
    beta, gamma, alpha = restore_arma_parameters(
        [beta_breve, gamma_breve, alpha], 2, add_intercept=True
    )
    assert np.allclose(beta, np.array([0.1, 0.3]))
    assert np.allclose(gamma, np.array([-0.4, -0.2]))
    assert np.allclose(alpha, np.array([0.2]))


def test_restore_arma_parameters_2_1_intercept() -> None:
    beta_breve = np.array([-0.3, 0.3])
    gamma_breve = np.array([0.4])
    alpha = np.array([0.2])
    beta, gamma, alpha = restore_arma_parameters(
        [beta_breve, gamma_breve, alpha], 2, add_intercept=True
    )
    assert np.allclose(beta, np.array([0.1, 0.3]))
    assert np.allclose(gamma, np.array([-0.4]))


def test_prepare_arma_input() -> None:
    endog = np.arange(10)
    X, y = prepare_arma_input(1, endog, 3)
    assert X.shape == (7, 3, 1, 1)
    assert y.shape == (7, 1)
    assert np.all(X[0, ...] == np.array([[[0]], [[1]], [[2]]]))
    assert np.all(y[0] == np.array([3]))

    X, y = prepare_arma_input(3, endog, 3)
    assert X.shape == (5, 3, 1, 3)
    assert y.shape == (5, 1)
    assert np.all(X[0, ...] == np.array([[[2, 1, 0]], [[3, 2, 1]], [[4, 3, 2]]]))
    assert np.all(y[0] == np.array([5]))


def test_simulate_VARMA() -> None:
    VAR = np.array([[0.1, 0.2], [0.3, 0.4]])
    VAR = np.expand_dims(VAR, axis=-1)
    VMA = np.zeros((2, 2, 1))
    alpha = np.zeros(2)
    Y = simulate_varma_process(VAR, VMA, alpha, n_steps=250)
    assert Y.shape == (250, 2)
