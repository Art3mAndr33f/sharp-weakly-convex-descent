"""Algorithms for sharp weakly convex minimization.

Implemented from Davis, Drusvyatskiy, MacPhee, Paquette (arXiv:1803.02461):
  * Algorithm 1: subgradient method with Polyak stepsize
  * Algorithm 2: subgradient method with constant stepsize
  * Algorithm 3: subgradient method with geometrically decreasing stepsize

Comparison / related methods:
  * subgradient method with diminishing stepsize alpha_k = c/sqrt(k)
    (classical global O(1/sqrt(k)) scheme, Davis & Drusvyatskiy 2018)
  * proximal point method (linearly convergent on sharp weakly convex f)
  * gradient descent and Nesterov accelerated gradient on smooth sharp
    problems (acceleration theme of arXiv:2009.11069)
"""

import numpy as np


def subgradient_polyak(problem, x0, f_star, n_iter, subgrad=None):
    """Algorithm 1 (arXiv:1803.02461): Polyak stepsize.

    x_{k+1} = x_k - alpha_k * zeta_k,
    alpha_k = (g(x_k) - f*) / ||zeta_k||^2,  zeta_k in d g(x_k).
    """
    subgrad = subgrad or problem.subgradient
    x = np.array(x0, dtype=float)
    hist = [problem.dist_to_solutions(x)]
    for _ in range(n_iter):
        z = subgrad(x)
        nz = np.linalg.norm(z)
        if nz < 1e-14:
            break
        alpha = max(problem.value(x) - f_star, 0.0) / (nz ** 2)
        x = x - alpha * z
        hist.append(problem.dist_to_solutions(x))
    return np.array(hist)


def subgradient_constant(problem, x0, alpha, n_iter, subgrad=None):
    """Algorithm 2 (arXiv:1803.02461): constant stepsize alpha."""
    subgrad = subgrad or problem.subgradient
    x = np.array(x0, dtype=float)
    hist = [problem.dist_to_solutions(x)]
    for _ in range(n_iter):
        z = subgrad(x)
        if np.linalg.norm(z) < 1e-14:
            break
        x = x - alpha * z
        hist.append(problem.dist_to_solutions(x))
    return np.array(hist)


def subgradient_geometric(problem, x0, lam, q, n_iter, subgrad=None):
    """Algorithm 3 (arXiv:1803.02461): alpha_k = lam * q^k."""
    subgrad = subgrad or problem.subgradient
    x = np.array(x0, dtype=float)
    hist = [problem.dist_to_solutions(x)]
    alpha = lam
    for _ in range(n_iter):
        z = subgrad(x)
        if np.linalg.norm(z) < 1e-14:
            break
        x = x - alpha * z
        hist.append(problem.dist_to_solutions(x))
        alpha *= q
    return np.array(hist)


def subgradient_diminishing(problem, x0, c, n_iter, subgrad=None):
    """Classical diminishing-step subgradient: alpha_k = c / sqrt(k+1)."""
    subgrad = subgrad or problem.subgradient
    x = np.array(x0, dtype=float)
    hist = [problem.dist_to_solutions(x)]
    for k in range(n_iter):
        z = subgrad(x)
        if np.linalg.norm(z) < 1e-14:
            break
        x = x - (c / np.sqrt(k + 1.0)) * z
        hist.append(problem.dist_to_solutions(x))
    return np.array(hist)


def proximal_point(problem, x0, t, n_iter):
    """Proximal point method: x_{k+1} = prox_{t g}(x_k)."""
    x = np.array(x0, dtype=float)
    hist = [problem.dist_to_solutions(x)]
    for _ in range(n_iter):
        x = problem.prox(x, t)
        hist.append(problem.dist_to_solutions(x))
    return np.array(hist)


def gradient_descent(problem, x0, step, n_iter):
    """Plain gradient descent on a smooth problem."""
    x = np.array(x0, dtype=float)
    hist = [problem.dist_to_solutions(x)]
    for _ in range(n_iter):
        g = problem.gradient(x)
        if np.linalg.norm(g) < 1e-14:
            break
        x = x - step * g
        hist.append(problem.dist_to_solutions(x))
    return np.array(hist)


def nesterov_accelerated(problem, x0, step, n_iter):
    """Nesterov's accelerated gradient (cf. arXiv:2009.11069, outer loop).

    y_k momentum scheme with the standard 1/sqrt(k)-free constant
    momentum for strongly convex f:  x_{k+1} = y_k - step * grad f(y_k),
    y_{k+1} = x_{k+1} + beta (x_{k+1} - x_k).
    """
    x = np.array(x0, dtype=float)
    y = x.copy()
    mu, L = problem.mu, problem.L
    beta = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))
    hist = [problem.dist_to_solutions(x)]
    for _ in range(n_iter):
        g = problem.gradient(y)
        x_new = y - step * g
        y = x_new + beta * (x_new - x)
        x = x_new
        hist.append(problem.dist_to_solutions(x))
    return np.array(hist)
