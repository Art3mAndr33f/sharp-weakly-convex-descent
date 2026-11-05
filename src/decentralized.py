"""Decentralized scaled subgradient method (Algorithm 1 of the draft paper).

Setting: n agents, agent i has a local weakly convex function f_i : R^d -> R,
global objective f(x) = (1/n) sum_i f_i(x).

Algorithm 1 (draft paper):
    y_i = x_i - (alpha_k / L) * zeta_i,   zeta_i in d f_i(x_i)
    X^{k+1} = Consensus(Y^k, T)           # T gossip steps

Consensus (gossip) is taken from arXiv:2009.11069: at each of the T rounds
one edge (i, j) of a time-varying network activates and the two agents
average their rows:
    W = I - tau (e_i - e_j)(e_i - e_j)^T,  tau in (0, 1/2],
    X <- W X.
A static variant (fixed doubly stochastic W applied T times) is also
provided for comparison.
"""

import numpy as np


class DecentralizedPhaseRetrieval:
    """Each agent i observes m_i measurements of a common x_true:
    f_i(x) = (1/m_i) sum_j (|a_ij^T x| - b_ij)^2,  b_ij = |a_ij^T x_true|.
    """

    def __init__(self, n=20, d=30, m=40, seed=0):
        rng = np.random.default_rng(seed)
        self.n, self.d, self.m = n, d, m
        self.x_true = rng.standard_normal(d)
        self.x_true /= np.linalg.norm(self.x_true)
        self.A = [rng.standard_normal((m, d)) for _ in range(n)]
        self.b = [np.abs(A @ self.x_true) for A in self.A]
        # Lipschitz constant of each local (sub)gradient
        self.L_i = np.array([4.0 * np.linalg.svd(A, compute_uv=False)[0] ** 2 / m
                             for A in self.A])
        self.L = float(self.L_i.max())
        self.f_star = 0.0

    def value_i(self, i, x):
        r = np.abs(self.A[i] @ x) - self.b[i]
        return float(np.mean(r ** 2))

    def value(self, X):
        return float(np.mean([self.value_i(i, X[i]) for i in range(self.n)]))

    def subgradient_i(self, i, x):
        s = self.A[i] @ x
        signs = np.sign(s)
        signs[signs == 0] = 1.0
        r = np.abs(s) - self.b[i]
        return 2.0 * (self.A[i].T @ (signs * r)) / self.m

    def dist_to_solutions(self, x):
        return min(np.linalg.norm(x - self.x_true),
                   np.linalg.norm(x + self.x_true))


def gossip_matrix(n, i, j, tau=0.5, rng=None):
    """W = I - tau (e_i - e_j)(e_i - e_j)^T (doubly stochastic for tau<=1/2)."""
    W = np.eye(n)
    W[i, i] -= tau
    W[j, j] -= tau
    W[i, j] += tau
    W[j, i] += tau
    return W


def consensus_time_varying(X, T, tau=0.5, rng=None):
    """T gossip rounds on a time-varying network: each round a random edge."""
    n = X.shape[0]
    for _ in range(T):
        i, j = rng.choice(n, size=2, replace=False)
        W = gossip_matrix(n, i, j, tau)
        X = W @ X
    return X


def consensus_static(X, W, T):
    """T rounds with a fixed doubly stochastic matrix W (static network)."""
    for _ in range(T):
        X = W @ X
    return X


def static_ring_W(n, tau=0.5):
    """Static ring network gossip matrix (doubly stochastic)."""
    W = np.eye(n) * (1.0 - 2 * tau)
    for i in range(n):
        W[i, (i - 1) % n] = tau
        W[i, (i + 1) % n] = tau
    return W


def algorithm1(problem, x0, alpha_seq, T, n_iter, tau=0.5, seed=1,
               static_W=None):
    """Algorithm 1 of the draft paper. Returns histories of
    dist(x_bar, X**), consensus error, mean objective, stationarity."""
    rng = np.random.default_rng(seed)
    n, d = problem.n, problem.d
    X = np.tile(np.asarray(x0, dtype=float), (n, 1))
    hist = {"dist": [], "consensus": [], "obj": [], "stationarity": []}
    for k in range(n_iter + 1):
        x_bar = X.mean(axis=0)
        cons = np.linalg.norm(X - np.tile(x_bar, (n, 1)), "fro")
        zetas = np.array([problem.subgradient_i(i, X[i]) for i in range(n)])
        hist["dist"].append(problem.dist_to_solutions(x_bar))
        hist["consensus"].append(cons)
        hist["obj"].append(problem.value(X))
        hist["stationarity"].append(np.linalg.norm(zetas.mean(axis=0)))
        if k == n_iter:
            break
        alpha = alpha_seq(k)
        Y = X - (alpha / problem.L) * zetas
        if static_W is not None:
            X = consensus_static(Y, static_W, T)
        else:
            X = consensus_time_varying(Y, T, tau, rng)
    return {k_: np.array(v) for k_, v in hist.items()}


def centralized_subgradient(problem, x0, alpha_seq, n_iter):
    """Baseline: centralized subgradient method on f = (1/n) sum f_i."""
    x = np.asarray(x0, dtype=float)
    hist = {"dist": [], "obj": [], "stationarity": []}
    for k in range(n_iter + 1):
        z = np.mean([problem.subgradient_i(i, x) for i in range(problem.n)],
                    axis=0)
        hist["dist"].append(problem.dist_to_solutions(x))
        hist["obj"].append(problem.value(np.tile(x, (problem.n, 1))))
        hist["stationarity"].append(np.linalg.norm(z))
        if k == n_iter:
            break
        x = x - (alpha_seq(k) / problem.L) * z
    return {k_: np.array(v) for k_, v in hist.items()}
