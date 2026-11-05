# sharp-weakly-convex-descent

Numerical implementation of the subgradient algorithms for sharp weakly convex
minimization from arXiv:1803.02461 (Polyak / constant / geometric stepsizes),
with comparison against diminishing-step subgradient, proximal point, and
Nesterov acceleration (cf. arXiv:2009.11069).

See `REPORT.md` (Russian) for the research summary. Paper-ready figures
(PDF + PNG, 300 dpi) are in `figures/`; regenerate with:

    python src/run_experiments.py       # sharp weakly convex (arXiv:1803.02461)
    python src/run_decentralized.py     # Algorithm 1 of the draft paper
                                        # (decentralized scaled subgradient
                                        #  + gossip consensus, arXiv:2009.11069)
