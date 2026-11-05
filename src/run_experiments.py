"""Run numerical experiments and produce paper-ready figures (PDF + PNG, 300 dpi).

Figures (saved to ../figures):
  fig1_phase_retrieval.pdf   -- linear convergence of the three subgradient
                                methods of arXiv:1803.02461 vs diminishing-step
                                subgradient and proximal point (phase retrieval)
  fig2_objective_gap.pdf     -- objective gap g(x_k) - g* for the same methods
  fig3_vs_measurements.pdf   -- empirical linear rate vs number of measurements m
  fig4_smooth_ls.pdf         -- GD vs Nesterov acceleration on smooth sharp LS
                                (comparison with accelerated schemes, arXiv:2009.11069)
  fig5_toy2d.pdf             -- 2D illustration: weakly convex sharp toy + iterates
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import PhaseRetrieval, SmoothLeastSquares, ToyWeaklyConvex
from algorithms import (
    subgradient_polyak, subgradient_constant, subgradient_geometric,
    subgradient_diminishing, proximal_point, gradient_descent,
    nesterov_accelerated,
)

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

# textwidth of the journal template is 135mm ~ 5.31 in
FIGW = 5.31


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"))
    fig.savefig(os.path.join(FIG_DIR, name + ".png"))
    print("saved", name)


def fig_phase_retrieval():
    prob = PhaseRetrieval(d=50, m=250, seed=0)
    rng = np.random.default_rng(42)
    x0 = prob.x_true + 0.3 * rng.standard_normal(prob.d)
    x0 /= np.linalg.norm(x0)
    n = 400
    curves = {
        "Polyak step (Alg. 1)": subgradient_polyak(prob, x0, prob.f_star, n),
        "constant step (Alg. 2)": subgradient_constant(prob, x0, alpha=30.0, n_iter=n),
        "geometric step (Alg. 3)": subgradient_geometric(prob, x0, lam=50.0, q=0.995, n_iter=n),
        "diminishing c/sqrt(k)": subgradient_diminishing(prob, x0, c=30.0, n_iter=n),
        "proximal point": proximal_point(prob, x0, t=30.0, n_iter=n),
    }
    fig, ax = plt.subplots(figsize=(FIGW, 3.4))
    for label, h in curves.items():
        ax.semilogy(np.maximum(h, 1e-16), label=label)
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel(r"$\mathrm{dist}(x_k, X^{**})$")
    ax.set_title("Phase retrieval: subgradient methods for sharp weakly convex $g$")
    ax.legend(loc="upper right")
    save(fig, "fig1_phase_retrieval")
    plt.close(fig)
    return prob, x0, curves


def fig_objective_gap(prob, x0):
    n = 400
    fig, ax = plt.subplots(figsize=(FIGW, 3.4))

    def run_with_gap(fn, **kw):
        # re-run tracking the objective gap
        import algorithms as alg
        x = np.array(x0, dtype=float)
        gaps = [max(prob.value(x) - prob.f_star, 1e-300)]
        orig = prob.dist_to_solutions
        for _ in range(n):
            pass
        return gaps

    # simpler: recompute gap histories directly
    def gap_history(step_fn):
        x = np.array(x0, dtype=float)
        out = [max(prob.value(x) - prob.f_star, 1e-300)]
        for _ in range(n):
            x = step_fn(x)
            out.append(max(prob.value(x) - prob.f_star, 1e-300))
        return np.array(out)

    def polyak_step(x):
        z = prob.subgradient(x)
        a = max(prob.value(x) - prob.f_star, 0.0) / (np.linalg.norm(z) ** 2)
        return x - a * z

    def const_step(x):
        return x - 30.0 * prob.subgradient(x)

    def geom_step_factory():
        state = {"a": 50.0}
        def step(x):
            z = prob.subgradient(x)
            x_new = x - state["a"] * z
            state["a"] *= 0.995
            return x_new
        return step

    ax.semilogy(gap_history(polyak_step), label="Polyak step (Alg. 1)")
    ax.semilogy(gap_history(const_step), label="constant step (Alg. 2)")
    ax.semilogy(gap_history(geom_step_factory()), label="geometric step (Alg. 3)")
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel(r"$g(x_k) - g^*$")
    ax.set_title("Objective gap on phase retrieval")
    ax.legend(loc="upper right")
    save(fig, "fig2_objective_gap")
    plt.close(fig)


def fig_vs_measurements():
    """Empirical linear rate vs number of measurements m (cf. Fig. 1 of 1803.02461)."""
    ms = [50, 100, 150, 200, 250, 300]
    rates = []
    d = 50
    for m in ms:
        prob = PhaseRetrieval(d=d, m=m, seed=0)
        rng = np.random.default_rng(42)
        x0 = prob.x_true + 0.3 * rng.standard_normal(d)
        x0 /= np.linalg.norm(x0)
        h = subgradient_polyak(prob, x0, prob.f_star, 300)
        h = h[h > 1e-12]
        if len(h) > 20:
            # slope of log-dist over the last linear segment
            k = np.arange(len(h))
            slope = np.polyfit(k[len(k) // 2:], np.log(h[len(k) // 2:]), 1)[0]
            rates.append(np.exp(slope))
        else:
            rates.append(np.nan)
    fig, ax = plt.subplots(figsize=(FIGW, 3.0))
    ax.plot(ms, rates, "o-")
    ax.set_xlabel("number of measurements $m$")
    ax.set_ylabel("empirical rate (per iteration)")
    ax.set_title("Polyak method on phase retrieval: rate improves with $m$")
    save(fig, "fig3_vs_measurements")
    plt.close(fig)


def fig_smooth_ls():
    prob = SmoothLeastSquares(d=100, m=300, cond=100.0, seed=1)
    rng = np.random.default_rng(7)
    x0 = rng.standard_normal(prob.d)
    n = 300
    h_gd = gradient_descent(prob, x0, step=1.0 / prob.L, n_iter=n)
    h_nag = nesterov_accelerated(prob, x0, step=1.0 / prob.L, n_iter=n)
    fig, ax = plt.subplots(figsize=(FIGW, 3.2))
    ax.semilogy(np.maximum(h_gd, 1e-16), label="gradient descent")
    ax.semilogy(np.maximum(h_nag, 1e-16), label="Nesterov accelerated (cf. arXiv:2009.11069)")
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel(r"$\|x_k - x^*\|$")
    ax.set_title("Smooth sharp least squares: acceleration comparison")
    ax.legend(loc="upper right")
    save(fig, "fig4_smooth_ls")
    plt.close(fig)


def fig_toy2d():
    toy = ToyWeaklyConvex(rho=0.5)
    xs = np.linspace(-1.5, 1.5, 300)
    X, Y = np.meshgrid(xs, xs)
    Z = np.abs(X) + np.abs(Y) - 0.25 * (X ** 2 + Y ** 2)
    fig, ax = plt.subplots(figsize=(FIGW, 4.2))
    cs = ax.contour(X, Y, Z, levels=25, cmap="viridis", linewidths=0.6)
    ax.clabel(cs, fontsize=6, fmt="%.1f")
    rng = np.random.default_rng(3)
    x = np.array([0.9, -0.8])
    traj = [x.copy()]
    for _ in range(60):
        z = toy.subgradient(x)
        nz = np.linalg.norm(z)
        if nz < 1e-12:
            break
        a = max(toy.value(x) - toy.f_star, 0.0) / (nz ** 2)
        x = np.clip(x - a * z, -1.0, 1.0)  # projected Polyak on the box
        traj.append(x.copy())
    traj = np.array(traj)
    ax.plot(traj[:, 0], traj[:, 1], "r.-", markersize=4, label="Polyak iterates")
    ax.plot(0, 0, "k*", markersize=12, label="$x^*=0$")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(r"$h(x)=\|x\|_1-\frac{\rho}{2}\|x\|^2$ (weakly convex, sharp)")
    ax.legend(loc="upper right")
    save(fig, "fig5_toy2d")
    plt.close(fig)


if __name__ == "__main__":
    prob, x0, _ = fig_phase_retrieval()
    fig_objective_gap(prob, x0)
    fig_vs_measurements()
    fig_smooth_ls()
    fig_toy2d()
    print("all figures done ->", os.path.abspath(FIG_DIR))
