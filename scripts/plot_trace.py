"""Plot a trial trace like the paper's Fig. 7.

Solid: measured fingertip-centre x / y / z; dashed: the target fingertip
centre. Pink: pushes on the hand; purple: the object jumped (env keys k / o,
``scripts/eval_sim.py --push_at / --jump_at``).

    python scripts/plot_trace.py outputs/eval/<stamp>/<trial>/trace.npz [-o fig.png]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    d = np.load(args.trace)
    tr = d["trace"]
    t, x, x_tgt = tr[:, 0], tr[:, 1:4], tr[:, 4:7]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    for t0 in d["push_at"]:
        ax.axvspan(t0, t0 + 0.3, color="#f4c2d7", alpha=0.8, lw=0)
    for t0 in d["jump_at"]:
        ax.axvspan(t0, t0 + 0.15, color="#6b3fa0", alpha=0.6, lw=0)
    for k, (name, c) in enumerate(zip("xyz", ("#1f77b4", "#2ca02c", "#d62728"))):
        ax.plot(t, x[:, k], color=c, lw=1.8, label=name)
        ax.plot(t, x_tgt[:, k], color=c, lw=1.2, ls="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("fingertip centre (m)")
    ax.legend(loc="upper right", ncols=3, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = args.out or args.trace.replace(".npz", ".png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
