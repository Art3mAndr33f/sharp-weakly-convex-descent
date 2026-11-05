"""Test problems for sharp weakly convex minimization.

Based on Davis, Drusvyatskiy, MacPhee, Paquette,
"Subgradient methods for sharp weakly convex functions" (arXiv:1803.02461).

Problems
--------
1. Robust phase retrieval (nonsmooth, weakly convex, sharp):
       g(x) = (1/m) * sum_i (| <a_i, x> | - b_i)^2,
       b_i = |<a_i, x_true>|,  a_i ~ N(0, I_d).
   Solution set X** = {+/- x_true} (for m large enough).

2. Smooth sharp least squares (convex, sharp, L-smooth) -- used to compare
   plain gradient descent with Nesterov's accelerated gradient (the
   acceleration theme of arXiv:2009.11069):
       f(x) = (1/2) ||A x - b||^2,  A full column rank.

3. Toy 2D weakly convex sharp function:
       h(x) = ||x||_1 - (rho/2) ||x||^2  on ||x||_inf <= R
   (rho-weakly convex, minimizer at 0, sharp with mu = 1 near 0).
"""

import numpy as np


class PhaseRetrieval:
    """g(x) = (1/m) sum_i (|a_i^T x| - b_i)^2."""

    def __init__(self, d=50, m=200, seed=0):
        rng = np.random.default_rng(seed)
        self.A = rng.standard_normal((m, d)) / np.sqrt(m)
        self.x_true = rng.standard_normal(d)
        self.x_true /= np.linalg.norm(self.x_true)
        self.b = np.abs(self.A @ self.x_true)
        self.d, self.m = d, m
        # minimizers: +/- x_true
        self.X_star = [self.x_true, -self.x_true]
        self.f_star = 0.0

    def value(self, x):
        r = np.abs(self.A @ x) - self.b
        return float(np.mean(r ** 2))

    def subgradient(self, x, rng=None):
        """Any element of the Clarke subdifferential."""
        s = self.A @ x
        signs = np.sign(s)
        signs[signs == 0] = 1.0  # arbitrary choice at kinks
        r = np.abs(s) - self.b
        # d/dx (|a^T x| - b)^2 = 2 (|a^T x| - b) sign(a^T x) a
        return 2.0 * (self.A.T @ (signs * r)) / self.m

    def dist_to_solutions(self, x):
        return min(np.linalg.norm(x - xs) for xs in self.X_star)

    def prox(self, x, t):
        """Proximal operator of g with parameter t (via 1D problems).

        g(x) = (1/m) sum_i (|a_i^T x| - b_i)^2.  The prox is computed by
        solving the dual or by a few Newton steps on the smooth dual; here
        we use the fact that prox_{t g} can be obtained from the Moreau
        decomposition through the conjugate, but for simplicity we do
        projected Newton on the strongly convex dual-free formulation:
        prox_{t g}(x) = argmin_z 0.5||z-x||^2 + t g(z).
        We solve it with Newton's method (g is piecewise smooth).
        """
        z = x.copy()
        for _ in range(60):
            s = self.A @ z
            signs = np.sign(s)
            signs[signs == 0] = 1.0
            r = np.abs(s) - self.b
            grad = (z - x) + t * 2.0 * (self.A.T @ (signs * r)) / self.m
            # Hessian (ignoring kinks): I + t * 2/m * A^T diag(signs^2) A
            H = np.eye(self.d) + t * 2.0 * (self.A.T @ self.A) / self.m
            step = np.linalg.solve(H, grad)
            z = z - step
            if np.linalg.norm(step) < 1e-12:
                break
        return z


class SmoothLeastSquares:
    """f(x) = 0.5 ||A x - b||^2, A full column rank (sharp + smooth)."""

    def __init__(self, d=100, m=300, cond=100.0, seed=1):
        rng = np.random.default_rng(seed)
        U, _ = np.linalg.qr(rng.standard_normal((m, d)))
        V, _ = np.linalg.qr(rng.standard_normal((d, d)))
        s = np.geomspace(1.0, np.sqrt(cond), d)  # singular values
        self.A = U @ np.diag(s) @ V.T
        self.x_true = rng.standard_normal(d)
        self.b = self.A @ self.x_true
        self.d = d
        self.L = float(s.max() ** 2)
        self.mu = float(s.min() ** 2)  # sharpness constant of f
        self.f_star = 0.0
        self.x_star = self.x_true

    def value(self, x):
        r = self.A @ x - self.b
        return 0.5 * float(r @ r)

    def gradient(self, x):
        return self.A.T @ (self.A @ x - self.b)

    def dist_to_solutions(self, x):
        return float(np.linalg.norm(x - self.x_star))


class ToyWeaklyConvex:
    """h(x) = ||x||_1 - (rho/2)||x||^2, rho-weakly convex, sharp at 0."""

    def __init__(self, rho=0.5):
        self.rho = rho
        self.f_star = 0.0
        self.x_star = np.zeros(2)

    def value(self, x):
        return float(np.sum(np.abs(x)) - 0.5 * self.rho * (x @ x))

    def subgradient(self, x):
        return np.sign(x) - self.rho * x

    def dist_to_solutions(self, x):
        return float(np.linalg.norm(x - self.x_star))
