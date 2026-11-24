"""Dirichlet-based GP classification (Milios et al., NeurIPS 2018), on top of scikit-learn.

Label counts per point are mapped to a Dirichlet, approximated per class by a log-normal:

    alpha_k  = counts_k + a_eps
    sigma2_k = log(1 / alpha_k + 1)
    y_k      = log(alpha_k) - sigma2_k / 2

and one GP regression per class is fitted to y_k with *fixed, per-point* noise sigma2_k
(more seeds at a point -> larger alpha -> smaller noise). Class probabilities are the
softmax of the latent functions, so posterior samples come from sampling the latent GPs.
Seeds that disagree at a point (a boundary signal) automatically raise the aleatoric part.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from .base import softmax


class DirichletGPClassifier:
    def __init__(self, n_classes: int, a_eps: float = 0.01, length_scale: float = 0.2,
                 length_scale_bounds: tuple[float, float] = (1e-2, 1e1),
                 n_restarts_optimizer: int = 2, random_state: int = 0) -> None:
        self.n_classes = n_classes
        self.a_eps = a_eps
        self.length_scale = length_scale
        self.length_scale_bounds = tuple(length_scale_bounds)
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.models: list[GaussianProcessRegressor] = []
        self.offsets = np.zeros(n_classes)

    def fit(self, X: np.ndarray, counts: np.ndarray) -> None:
        X = np.asarray(X, float)
        alpha = np.asarray(counts, float) + self.a_eps
        sigma2 = np.log(1.0 / alpha + 1.0)
        y = np.log(alpha) - sigma2 / 2.0
        self.offsets = y.mean(axis=0)
        self.models = []
        for k in range(self.n_classes):
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                length_scale=np.full(X.shape[1], self.length_scale),
                length_scale_bounds=self.length_scale_bounds)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma2[:, k], normalize_y=False,
                                          n_restarts_optimizer=self.n_restarts_optimizer,
                                          random_state=self.random_state + k)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(X, y[:, k] - self.offsets[k])
            self.models.append(gp)

    def latent(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Latent mean and standard deviation, each (n_points, n_classes)."""
        mus, sds = [], []
        for k, gp in enumerate(self.models):
            mu, sd = gp.predict(np.asarray(X, float), return_std=True)
            mus.append(mu + self.offsets[k])
            sds.append(sd)
        return np.column_stack(mus), np.column_stack(sds)

    def sample_proba(self, X: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        if not self.models:
            raise RuntimeError("fit the surrogate first")
        mu, sd = self.latent(X)
        f = mu[None] + sd[None] * rng.standard_normal((n_samples, *mu.shape))
        return softmax(f, axis=-1)
