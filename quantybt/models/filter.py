import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError, cholesky, eigvals, inv
import math

# full implemented UKF, ideal for non linear systems
class UnscentedKalmanFilter:
    def __init__(self, f, h, Q, R, initial_state, initial_covariance, alpha=1e-3, beta=2, kappa=0):
        """
        Unscented Kalman Filter (UKF) Implementation.

        Parameters:
            f (callable): State transition function (x_prev -> x_pred).
            h (callable): Measurement function (x -> z).
            Q (np.ndarray): Process noise covariance matrix (n x n).
            R (np.ndarray): Measurement noise covariance matrix (m x m).
            initial_state (np.ndarray): Initial state estimate (n-dimensional).
            initial_covariance (np.ndarray): Initial covariance estimate (n x n).
            alpha (float): UKF scaling parameter (default: 1e-3).
            beta (float): UKF parameter for distribution type (default: 2).
            kappa (float): UKF secondary scaling parameter (default: 0).
        """
        self.f = f
        self.h = h
        self.Q = Q.astype(np.float64)
        self.R = R.astype(np.float64)
        self.state = initial_state.astype(np.float64)
        self.covariance = initial_covariance.astype(np.float64)
        self.n = len(initial_state)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lmbda = alpha**2 * (self.n + kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lmbda)

    @staticmethod
    def _is_positive_definite(matrix):
        try:
            cholesky(matrix)
            return True
        except LinAlgError:
            return False

    @classmethod
    def _nearest_positive_definite(cls, matrix):
        B = (matrix + matrix.T) / 2
        _, s, V = np.linalg.svd(B)
        H = V.T @ np.diag(s) @ V
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        if cls._is_positive_definite(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(matrix))
        I = np.eye(matrix.shape[0])
        k = 1
        max_iter = 100
        for _ in range(max_iter):
            mineig = np.min(np.real(eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1
            if cls._is_positive_definite(A3):
                break

        return A3

    def sigma_points(self):
        """Generate sigma points using the Unscented Transform."""
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = self.state

        try:
            sqrt_cov = cholesky(self.covariance + 1e-6 * np.eye(self.n))
        except LinAlgError:
            adjusted_cov = self._nearest_positive_definite(self.covariance + 1e-6 * np.eye(self.n))
            sqrt_cov = cholesky(adjusted_cov)

        for i in range(self.n):
            sigma_points[i + 1] = self.state + self.gamma * sqrt_cov[i]
            sigma_points[self.n + i + 1] = self.state - self.gamma * sqrt_cov[i]

        return sigma_points

    def predict(self):
        """prediction step."""
        sigma_points = self.sigma_points()
        predicted_points = np.array([self.f(sp) for sp in sigma_points])
        self.state = np.dot(self._weights_mean(), predicted_points)
        self.covariance = self.Q.copy()
        for i, w in enumerate(self._weights_cov()):
            diff = predicted_points[i] - self.state
            self.covariance += w * np.outer(diff, diff)
        if not self._is_positive_definite(self.covariance):
            self.covariance = self._nearest_positive_definite(self.covariance)

    def update(self, measurement):
        """update step."""
        sigma_points = self.sigma_points()
        pred_measurements = np.array([self.h(sp) for sp in sigma_points])

        # Measurement mean and covariance
        z_mean = np.dot(self._weights_mean(), pred_measurements)
        z_cov = self.R.copy()
        cross_cov = np.zeros((self.n, len(measurement)))

        for i, w in enumerate(self._weights_cov()):
            z_diff = pred_measurements[i] - z_mean
            z_cov += w * np.outer(z_diff, z_diff)
            x_diff = sigma_points[i] - self.state
            cross_cov += w * np.outer(x_diff, z_diff)

        # Kalman gain and update
        if not self._is_positive_definite(z_cov):
            z_cov = self._nearest_positive_definite(z_cov)
        
        K = cross_cov @ inv(z_cov)
        innovation = measurement - z_mean
        self.state += K @ innovation
        self.covariance -= K @ z_cov @ K.T

        if not self._is_positive_definite(self.covariance):
            self.covariance = self._nearest_positive_definite(self.covariance)

    def _weights_mean(self):
        weights = np.full(2 * self.n + 1, 1 / (2 * (self.n + self.lmbda)))
        weights[0] = self.lmbda / (self.n + self.lmbda)
        return weights

    def _weights_cov(self):
        weights = self._weights_mean().copy()
        weights[0] += 1 - self.alpha**2 + self.beta
        return weights

# simple Kalmanfilter with linear-gaussian state model
class SimpleKalmanFilter:
    def __init__(self, Q, R, m0=0, P0=1):
        self.Q, self.R = Q, R
        self.m, self.P = m0, P0

    def update(self, y):
        m_pred = self.m
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.m = m_pred + K * (y - m_pred)
        self.P = (1 - K) * P_pred
        return 1 - 0.5 * (1 + math.erf(-self.m / np.sqrt(2 * self.P)))
    
