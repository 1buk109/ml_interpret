from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def toy_dataset_pfi(
        beta: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        N: int = 1_000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a toy dataset for permutation feature importance.

    Parameters
    ----------
    N : int
        Number of samples to generate.
    beta : np.ndarray
        The true regression coefficients.
    mu : np.ndarray
        The mean of the multivariate normal distribution.
    sigma : np.ndarray
        The covariance matrix of the multivariate normal distribution.
    """
    X = np.random.multivariate_normal(mu, sigma, N)
    epsilon = np.random.normal(0, 0.1, N)
    y = np.dot(X, beta) + epsilon
    return train_test_split(X, y, test_size=0.2, random_state=42)


def toy_dataset_pdp(
        N: int = 1_000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a toy dataset for partial dependence.

    Parameters
    -------
    N : int
        Number of samples to generate.
    """
    X = np.random.uniform(-np.pi * 2, np.pi * 2, [N, 2])
    epsilon = np.random.normal(0, 0.1, N)
    y = 10 * np.sin(X[:, 0]) + X[:, 1] + epsilon
    return train_test_split(X, y, test_size=0.2, random_state=42)


def toy_dataset_ice(
        N: int = 1_000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a toy dataset for individual conditional expectation.

    Parameters
    -------
    N : int
        Number of samples to generate.
    """
    x0 = np.random.uniform(-1, 1, N)
    x1 = np.random.uniform(-1, 1, N)
    x2 = np.random.binomial(1, 0.5, N)
    epsilon = np.random.normal(0, 0.1, N)
    X = np.column_stack((x0, x1, x2))
    y = x0 - 5 * x1 + 10 * x1 * x2 + epsilon
    return train_test_split(X, y, test_size=0.2, random_state=42)


def toy_dataset_shap(
        N: int = 1_000,
        J: int = 2
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a toy dataset for sharpley additive explanations.

    Parameters
    -------
    N : int
        Number of samples to generate.
    """
    beta = np.array([0, 1])

    X = np.random.normal(0, 1, [N, J])
    e = np.random.normal(0, 0.1, N)
    y = X @ beta + e
    return train_test_split(X, y, test_size=0.2, random_state=42)
