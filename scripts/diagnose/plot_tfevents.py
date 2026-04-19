#!/usr/bin/env python3
"""Parse RSL-RL tensorboard event files and plot training curves.

Does NOT require Isaac Sim — only tensorboard + matplotlib.

Usage:
    python scripts/diagnose/plot_tfevents.py                       # latest run
    python scripts/diagnose/plot_tfevents.py --run 2026-04-16_00-06-56
    python scripts/diagnose/plot_tfevents.py --compare 2026-04-15_22-56-56 2026-04-16_00-06-56
"""
from __future__ import annotations

import argparse
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    raise SystemExit(
        "tensorboard not installed. Run:\n"
        "  pip install tensorboard matplotlib\n"
        "or use Isaac Lab python: ./isaaclab.sh -p scripts/diagnose/plot_tfevents.py"
    )

LOG_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "logs", "rsl_rl", "target_aiming_direct")
)

# Key panels we care about (subset of tags RSL-RL writes).
# The actual tag prefix may be "Loss/" or "Policy/" depending on rsl_rl version.
PANELS = [
    ("Return / Episode length", ["Train/mean_reward", "Train/mean_episode_length"]),
    ("Loss", ["Loss/value_function", "Loss/surrogate", "Loss/entropy"]),
    ("Policy", ["Policy/mean_noise_std", "Policy/mean_kl"]),
    ("Rewards (from env)", ["Episode_Reward/*"]),
    ("Terminations", ["Episode_Termination/*"]),
]


def list_runs():
    runs = sorted(
        d for d in glob(os.path.join(LOG_ROOT, "*"))
        if os.path.isdir(d) and os.path.basename(d)[0].isdigit()
    )
    return runs


def load_run(run_dir: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        vals = np.array([e.value for e in events])
        out[tag] = (steps, vals)
    return out


def _match_tags(tags: list[str], pattern: str) -> list[str]:
    if not pattern.endswith("/*"):
        return [t for t in tags if t == pattern]
    prefix = pattern[:-1]  # keep trailing slash
    return [t for t in tags if t.startswith(prefix)]


def plot_run(run_dir: str, save_path: str | None = None) -> None:
    data = load_run(run_dir)
    all_tags = sorted(data.keys())
    print(f"[INFO] {os.path.basename(run_dir)}: {len(all_tags)} scalar tags")
    for t in all_tags:
        print(f"  - {t}  ({len(data[t][0])} points)")

    n_panels = len(PANELS)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.2 * n_panels), squeeze=False)
    for ax, (title, patterns) in zip(axes[:, 0], PANELS):
        plotted = False
        for p in patterns:
            for tag in _match_tags(all_tags, p):
                steps, vals = data[tag]
                if len(steps) == 0:
                    continue
                ax.plot(steps, vals, label=tag, linewidth=1.2)
                plotted = True
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        if plotted:
            ax.legend(fontsize=7, loc="best")
        else:
            ax.text(0.5, 0.5, "no matching tags", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
    fig.suptitle(f"Training curves: {os.path.basename(run_dir)}", fontsize=13)
    fig.tight_layout()
    if save_path is None:
        save_path = os.path.join(run_dir, "training_curves.png")
    fig.savefig(save_path, dpi=120)
    print(f"[INFO] Saved {save_path}")
    plt.close(fig)


def plot_compare(run_dirs: list[str], save_path: str) -> None:
    datasets = [(os.path.basename(d), load_run(d)) for d in run_dirs]

    compare_tags = [
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Loss/value_function",
        "Loss/surrogate",
        "Loss/entropy",
        "Policy/mean_kl",
    ]
    fig, axes = plt.subplots(len(compare_tags), 1, figsize=(12, 2.6 * len(compare_tags)), squeeze=False)
    for ax, tag in zip(axes[:, 0], compare_tags):
        for name, data in datasets:
            if tag in data:
                steps, vals = data[tag]
                ax.plot(steps, vals, label=name, linewidth=1.2)
        ax.set_title(tag)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    print(f"[INFO] Saved comparison to {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="run directory name under logs/rsl_rl/target_aiming_direct/ (default: latest)")
    parser.add_argument("--compare", nargs="+", help="list of runs to overlay on the same axes")
    parser.add_argument("--list", action="store_true", help="list available runs and exit")
    args = parser.parse_args()

    runs = list_runs()
    if args.list:
        for r in runs:
            print(os.path.basename(r))
        return

    if args.compare:
        dirs = [os.path.join(LOG_ROOT, r) for r in args.compare]
        for d in dirs:
            if not os.path.isdir(d):
                raise SystemExit(f"not found: {d}")
        out = os.path.join(LOG_ROOT, f"compare_{'_vs_'.join(args.compare)}.png")
        plot_compare(dirs, out)
        return

    if args.run:
        run_dir = os.path.join(LOG_ROOT, args.run)
    else:
        if not runs:
            raise SystemExit(f"no runs found under {LOG_ROOT}")
        run_dir = runs[-1]
    plot_run(run_dir)


if __name__ == "__main__":
    main()
