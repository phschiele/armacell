import matplotlib.pyplot as plt
import numpy as np
from armacell.helpers import restore_arma_parameters
from statsmodels.tsa.arima.model import ARIMAResults
from tensorflow import keras


def plot_convergence(model: keras.Model, p: int, add_intercept: bool, arima_model: ARIMAResults, path: str = "") -> None:
    transformed_parameters = [restore_arma_parameters(w, p, add_intercept) for w in model.history.history["weights"]]
    beta = np.stack(w[0] for w in transformed_parameters)
    gamma = np.stack(w[1] for w in transformed_parameters)

    plt.figure()

    plt.axhline(y=0, color="#909090", linestyle="-")

    for i in range(beta.shape[1]):
        plt.axhline(arima_model.arparams[i], c="g", linestyle="--")
        if i > 0:
            plt.plot(beta[:, i], c="g")
        else:
            plt.plot(beta[:, i], c="g", label="AR")

    for i in range(gamma.shape[1]):
        plt.axhline(arima_model.maparams[i], c="r", linestyle="--")
        if i > 0:
            plt.plot(gamma[:, i], c="r")
        else:
            plt.plot(gamma[:, i], c="r", label="MA")

    if add_intercept:
        alpha = np.hstack(w[2] for w in transformed_parameters)
        plt.plot(alpha, c="b", label="Intercept")
        plt.axhline(arima_model.params[0], c="b", linestyle="--")

    plt.xlim(0, len(transformed_parameters) - 1)

    plt.xlabel("Epochs")
    plt.ylabel("Coefficient Value")

    ax = plt.gca()
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if path:
        plt.savefig(path)
    plt.show()
