import reservoirpy as rpy
import numpy as np

rpy.verbosity(0)


class Forecaster:
    """
    A class that provides a wrapper around the ESN (Echo State Network) model from reservoirpy
    for generative predictions.

    Attributes:
        model (rpy.model.Model): Instance of the ESN model from reservoirpy.
        num_features (int): Number of features expected in the input data.
    """

    def __init__(self, esn: rpy.model.Model, num_features: int) -> None:
        """
        Initialize the Forecaster with an ESN model and the number of features.

        Args:
            esn (rpy.model.Model): Instance of the ESN model from reservoirpy.
            num_features (int): Number of features in the input data.
        """
        self.num_features = num_features
        self.model = esn

    def forecast(self, prediction_length: int, warmup_X: np.ndarray) -> np.ndarray:
        """
        Generate a prediction sequence of specific length using the ESN model.

        Args:
            prediction_length (int): The length of the prediction sequence to be generated.
            warmup_X (np.ndarray): The input data used for warmup. This should have the same number
                                   of features as specified during initialization.

        Returns:
            np.ndarray: The generated prediction sequence of size (prediction_length, 1).

        Raises:
            AssertionError: If the number of features in warmup_X is not consistent with num_features.
        """
        assert warmup_X.shape[1] == self.num_features

        ypred = np.empty((prediction_length, 1))

        # Reset internal state of the model and feed it a warmup signal
        warmup_y = self.model.run(warmup_X, reset=True)
        last_X = warmup_X[-1]

        # Generate the first prediction
        x = np.concatenate(
            (last_X[-(self.num_features - 1):].flatten(), warmup_y[-1]))

        # Generate subsequent predictions
        for i in range(prediction_length):
            prediction = self.model.run(x)
            x = np.concatenate(
                (x[-(self.num_features - 1):], prediction.flatten()))
            ypred[i] = prediction

        return ypred
