import reservoirpy as rpy
import numpy as np


class Forecaster:
    """
    A class that provides a wrapper around the ESN (Echo State Network) model from reservoirpy
    for generative predictions.

    Attributes:
        model (rpy.model.Model): Instance of the ESN model from reservoirpy.
        num_features (int): Number of features expected in the input data.
    """

    model: rpy.model.Model
    num_features: int

    def __init__(self, esn: rpy.model.Model, num_features: int) -> None:
        """
        Initialize the Forecaster with an ESN model and the number of features.

        Args:
            esn (rpy.model.Model): Instance of the ESN model from reservoirpy.
            num_features (int): Number of features in the input data.
        """
        self.num_features = num_features
        self.model = esn

    def fit(self, X: np.ndarray, y: np.ndarray, warmup: int = 0) -> None:
        """
        Fit the model to the data using an offline learning rule.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The target data.
            warmup (int, optional): Number of warmup samples. Default is 0.
        """
        self.model = self.model.fit(X, y, warmup=warmup)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on the data using an online learning rule.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The target data.
        """
        self.model.train(X, y)

    def forecast(self, T: int, warmup_X: np.ndarray) -> np.ndarray:
        """
        Generate a prediction sequence of length T using the ESN model.

        Args:
            T (int): The length of the prediction sequence to be generated.
            warmup_X (np.ndarray): The input data used for warmup. This should have the same number
                                   of features as specified during initialization.

        Returns:
            np.ndarray: The generated prediction sequence of size (T, 1).

        Raises:
            AssertionError: If the number of features in warmup_X is not consistent with num_features.
        """
        assert warmup_X.shape[1] == self.num_features

        ypred = np.empty((T, 1))

        # Reset internal state of the model and feed it a warmup signal
        warmup_y = self.model.run(warmup_X, reset=True)
        last_X = warmup_X[-1]

        # Generate the first prediction
        x = np.concatenate((last_X[-(self.num_features - 1) :].flatten(), warmup_y[-1]))

        # Generate subsequent predictions
        for i in range(T):
            prediction = self.model.run(x)
            x = np.concatenate((x[-(self.num_features - 1) :], prediction.flatten()))
            ypred[i] = prediction

        return ypred
