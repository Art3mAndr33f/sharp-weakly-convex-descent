"""Experiments for Algorithm 1 of the draft paper (decentralized scaled
subgradient method + T gossip consensus steps, cf. arXiv:2009.11069).

Figures (Russian labels, PDF + PNG, 300 dpi) in ../figures:
  dec_fig1_consensus_T.pdf   -- dist(x_bar, X**) for different T
  dec_fig2_consensus_err.pdf -- consensus error ||X - 1 x_bar^T||_F for different T
  dec_fig3_vs_centralized.pdf-- Alg. 1 vs centralized subgradient vs T=0 (local only)
  dec_fig4_networks.pdf      -- time-varying random gossip vs static ring
  dec_fig5_stepsize.pdf      -- constant vs diminishing alpha_k (stationarity)
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decentralized import (DecentralizedPhaseRetrieval, algorithm1,
                           centralized_subgradient, static_ring_W)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
FIGW = 5.31  # textwidth 135 mm


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"))
    fig.savefig(os.path.join(FIG_DIR, name + ".png"))
    print("saved", name)


def main():
    prob = DecentralizedPhaseRetrieval(n=20, d=30, m=40, seed=0)
    rng = np.random.default_rng(42)
    x0 = prob.x_true + 0.3 * rng.standard_normal(prob.d)
    x0 /= np.linalg.norm(x0)
    n_iter = 300
    alpha_const = lambda k: 1.0

    # --- fig 1: effect of the number of consensus steps T ---
    fig, ax = plt.subplots(figsize=(FIGW, 3.4))
    for T in [0, 1, 5, 20]:
        h = algorithm1(prob, x0, alpha_const, T, n_iter, seed=1)
        ax.semilogy(np.maximum(h["dist"], 1e-16), label=f"$T={T}$")
    ax.set_xlabel("итерация $k$")
    ax.set_ylabel(r"$\mathrm{dist}(\bar{x}^{(k)}, X^{**})$")
    ax.set_title("Алгоритм 1: зависимость от числа шагов консенсуса $T$")
    ax.legend()
    save(fig, "dec_fig1_consensus_T")
    plt.close(fig)

    # --- fig 2: consensus error ---
    fig, ax = plt.subplots(figsize=(FIGW, 3.4))
    for T in [0, 1, 5, 20]:
        h = algorithm1(prob, x0, alpha_const, T, n_iter, seed=1)
        ax.semilogy(np.maximum(h["consensus"], 1e-16), label=f"$T={T}$")
    ax.set_xlabel("итерация $k$")
    ax.set_ylabel(r"$\|X^{(k)}-\mathbf{1}\bar{x}^{(k)\top}\|_F$")
    ax.set_title("Ошибка консенсуса")
    ax.legend()
    save(fig, "dec_fig2_consensus_err")
    plt.close(fig)

    # --- fig 3: vs centralized and local-only ---
    fig, ax = plt.subplots(figsize=(FIGW, 3.4))
    h_dec = algorithm1(prob, x0, alpha_const, 10, n_iter, seed=1)
    h_loc = algorithm1(prob, x0, alpha_const, 0, n_iter, seed=1)
    h_cen = centralized_subgradient(prob, x0, alpha_const, n_iter)
    ax.semilogy(np.maximum(h_dec["obj"], 1e-16), label="Алгоритм 1 ($T=10$)")
    ax.semilogy(np.maximum(h_loc["obj"], 1e-16), label="локальные шаги ($T=0$)")
    ax.semilogy(np.maximum(h_cen["obj"], 1e-16), label="централизованный субградиент")
    ax.set_xlabel("итерация $k$")
    ax.set_ylabel(r"$f(X^{(k)}) - f^*$")
    ax.set_title("Сравнение с централизованным методом и без консенсуса")
    ax.legend()
    save(fig, "dec_fig3_vs_centralized")
    plt.close(fig)

    # --- fig 4: time-varying vs static network ---
    W_ring = static_ring_W(prob.n, tau=0.4)
    fig, ax = plt.subplots(figsize=(FIGW, 3.4))
    h_tv = algorithm1(prob, x0, alpha_const, 10, n_iter, seed=1)
    h_st = algorithm1(prob, x0, alpha_const, 10, n_iter, seed=1,
                      static_W=W_ring)
    ax.semilogy(np.maximum(h_tv["dist"], 1e-16), label="time-varying gossip")
    ax.semilogy(np.maximum(h_st["dist"], 1e-16), label="статическое кольцо")
    ax.set_xlabel("итерация $k$")
    ax.set_ylabel(r"$\mathrm{dist}(\bar{x}^{(k)}, X^{**})$")
    ax.set_title("Алгоритм 1 ($T=10$): тип сети")
    ax.legend()
    save(fig, "dec_fig4_networks")
    plt.close(fig)

    # --- fig 5: stepsize rules (stationarity measure) ---
    alpha_dim = lambda k: 5.0 / np.sqrt(k + 1.0)
    fig, ax = plt.subplots(figsize=(FIGW, 3.4))
    for name, a_seq in [("постоянный $\\alpha_k=1$", alpha_const),
                        (r"убывающий $5/\sqrt{k}$", alpha_dim)]:
        h = algorithm1(prob, x0, a_seq, 10, n_iter, seed=1)
        ax.semilogy(np.maximum(h["stationarity"], 1e-16), label=name)
    ax.set_xlabel("итерация $k$")
    ax.set_ylabel(r"$\|\frac{1}{n}\sum_i \zeta_i^{(k)}\|$")
    ax.set_title("Мера estacionарности при разных шагах")
    ax.legend()
    save(fig, "dec_fig5_stepsize")
    plt.close(fig)

    print("done ->", os.path.abspath(FIG_DIR))


if __name__ == "__main__":
    main()
