import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


def create_training_data(ts_data: np.ndarray, num_features: int) -> tuple[np.ndarray]:
    """
    Create a training dataset from a 1D time series array.

    Args:
        ts_data (np.ndarray): 1D array representing the time series data.
        num_features (int): Number of features (time steps) to use for each sample.

    Returns:
        tuple[np.ndarray]: A tuple containing the training matrix X and the target vector y.

    Example:
        ts_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        num_features = 3
        X, y = create_training_data(ts_data, num_features)
    """
    num_samples = len(ts_data) - num_features

    # Create the indices for slicing the data
    indices = np.arange(num_samples).reshape(-1, 1) + np.arange(num_features)

    # Create the X matrix by slicing the data
    X = ts_data[indices.flatten()].reshape(num_samples, num_features)

    # Create the y vector by slicing the data
    y = ts_data[num_features:]

    return X, y


def create_multivariate_training_data(
    ts_data: np.ndarray, num_features: int, train_size: int
) -> Tuple[np.ndarray]:
    """
    Create training and testing datasets from multiple time series.

    Args:
        ts_data (np.ndarray): 2D array representing multiple time series data.
        num_features (int): Number of features (time steps) to use for each sample.
        train_size (float): Proportion of the data to allocate for training.

    Returns:
        tuple[np.ndarray]: A tuple containing Xtrain, ytrain, Xtest, and ytest.

    Example:
        ts_data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        num_features = 3
        train_size = 0.5
        Xtrain, ytrain, Xtest, ytest = create_multivariate_training_data(ts_data, num_features, train_size)
    """
    num_time_series = ts_data.shape[0]
    all_Xtrain = []
    all_ytrain = []
    all_Xtest = []
    all_ytest = []

    # Create training and testing sets for each time series
    for ts_idx in range(num_time_series):
        ts = ts_data[ts_idx]
        X, y = create_training_data(ts, num_features)
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X, y, train_size=train_size, shuffle=False
        )

        all_Xtrain.append(Xtrain)
        all_ytrain.append(ytrain)

        all_Xtest.append(Xtest)
        all_ytest.append(ytest)

    # Concatenate the training and testing sets
    Xtrain = np.concatenate(all_Xtrain)
    ytrain = np.concatenate(all_ytrain)
    Xtest = np.concatenate(all_Xtest)
    ytest = np.concatenate(all_ytest)

    return Xtrain, ytrain, Xtest, ytest
