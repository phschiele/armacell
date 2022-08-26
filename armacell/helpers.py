import random
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf

from tensorflow import keras


class SaveWeights(keras.callbacks.Callback):
    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        if epoch == 0:
            self.model.history.history["weights"] = [self.model.get_weights()]
        else:
            self.model.history.history["weights"].append(self.model.get_weights())


def restore_arma_parameters(model_weights: List[np.array], p: int, add_intercept: bool = False)\
        -> Tuple[np.array, np.array, Optional[np.array]]:

    validate_weight_list_length(model_weights, add_intercept)

    beta_breve = model_weights[0].flatten()
    gamma_breve = model_weights[1].flatten()

    gamma = -gamma_breve
    if len(gamma) < len(beta_breve):
        beta = beta_breve - np.pad(gamma, (0, len(beta_breve) - len(gamma)))
    else:
        beta = beta_breve - gamma[:p]

    if add_intercept:
        alpha = model_weights[-1].flatten()
    else:
        alpha = None
    return beta[:p], gamma, alpha


def validate_weight_list_length(weight_list: list, add_intercept: bool) -> None:
    assert len(weight_list) == 3 if add_intercept else 2


def prepare_arma_input(p: int, endog: np.array, sequence_length: int = 10) -> Tuple[np.array, np.array]:
    # Input:      T x k or T,
    # Output: X: (T - sequence_length - p + 1) x sequence_length x k x p
    #         y: (T - sequence_length - p + 1) x k

    if endog.ndim == 1:
        endog = endog.reshape((-1, 1))
    endog = np.expand_dims(endog, axis=-1)
    endog_rep = np.concatenate([endog[p - 1 :, ...]] + [endog[p - i - 1 : -i, ...] for i in range(1, p)], axis=-1)
    ts_gen = keras.preprocessing.sequence.TimeseriesGenerator(endog_rep, endog_rep, sequence_length)
    X = np.vstack([ts_gen[i][0] for i in range(len(ts_gen))])
    y = np.vstack([ts_gen[i][1][..., 0] for i in range(len(ts_gen))])
    return X, y


def simulate_arma_process(ar: np.array, ma: np.array, alpha: float, n_steps: int = 1000, std: float = 1.0, burn_in: int = 50) -> np.array:
    steps_incl_burn_in = n_steps + burn_in
    eps = np.random.normal(0, std, steps_incl_burn_in)
    res = np.zeros(steps_incl_burn_in)
    for i in range(steps_incl_burn_in):
        res[i] = alpha + eps[i]
        for j, ar_par in enumerate(ar):
            if i > j:
                res[i] += res[i - j - 1] * ar_par
        for j, ma_par in enumerate(ma):
            if i > j:
                res[i] += eps[i - j - 1] * ma_par
    return res[burn_in:]


def simulate_varma_process(ar: np.array, ma: np.array, alpha: np.array, n_steps: int = 1000, std: float = 1.0, burn_in: int = 50) -> np.array:
    # AR Shape: k x k x p
    # MA Shape: k x k x q
    # Output: n_steps x k
    assert ar.ndim == ma.ndim == 3
    assert ar.shape[0] == ar.shape[1] == ma.shape[0] == ma.shape[1]
    assert isinstance(alpha, np.ndarray)

    k = ar.shape[0]

    steps_incl_burn_in = n_steps + burn_in
    eps = np.random.normal(0, std, (steps_incl_burn_in, k))

    res = np.zeros((steps_incl_burn_in, k))

    for i in range(steps_incl_burn_in):
        res[i, :] = alpha + eps[i, :]
        for j in range(ar.shape[2]):
            if i > j:
                res[i, :] += ar[:, :, j] @ res[i - j - 1, :]
        for j in range(ma.shape[2]):
            if i > j:
                res[i, :] += ma[:, :, j] @ eps[i - j - 1, :]
    return res[burn_in:, :]


def set_all_seeds(seed: int = 0) -> None:
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
