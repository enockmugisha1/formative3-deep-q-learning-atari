"""Task 3 hyperparameter sweep.

Runs the 10 required experiments, each with a different combination of
hyperparameters, by reusing the training loop in ``train.py`` unchanged.
For each experiment it:

  * overrides the hyperparameter constants in ``train`` for that run,
  * writes the model to ``models/<member>/expNN.pth``,
  * writes the per-episode log to ``plots/<member>/expNN_log.csv``,
  * writes the reward curve to ``plots/<member>/expNN_rewards.png``,
  * appends final metrics to ``models/<member>/summary.csv``.

The "Observed behaviour" column of the results table is meant to be filled
in from these logs/plots after the runs finish.

Run all 10:          uv run python experiments.py
Run one experiment:  uv run python experiments.py --only 3
Quick smoke test:    uv run python experiments.py --only 3 --steps 15000
"""

import argparse
import csv
import os

import numpy as np

import train  # we override its module-level constants per experiment

MEMBER = "eldrige"

# ---------------------------------------------------------------------------
# The 10 experiments. Experiment 1 is the baseline (train.py defaults); each
# later one changes one or two hyperparameters so their effect is isolated.
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    # lr,      gamma, batch, eps_start, eps_end, eps_decay
    dict(lr=1e-4, gamma=0.99,  batch=32, eps_start=1.0, eps_end=0.05, eps_decay=100_000),  # 1 baseline
    dict(lr=5e-4, gamma=0.99,  batch=32, eps_start=1.0, eps_end=0.05, eps_decay=100_000),  # 2 higher LR
    dict(lr=5e-5, gamma=0.99,  batch=32, eps_start=1.0, eps_end=0.05, eps_decay=100_000),  # 3 lower LR
    dict(lr=1e-4, gamma=0.95,  batch=32, eps_start=1.0, eps_end=0.05, eps_decay=100_000),  # 4 lower gamma
    dict(lr=1e-4, gamma=0.999, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=100_000),  # 5 higher gamma
    dict(lr=1e-4, gamma=0.99,  batch=64, eps_start=1.0, eps_end=0.05, eps_decay=100_000),  # 6 larger batch
    dict(lr=1e-4, gamma=0.99,  batch=16, eps_start=1.0, eps_end=0.05, eps_decay=100_000),  # 7 smaller batch
    dict(lr=1e-4, gamma=0.99,  batch=32, eps_start=1.0, eps_end=0.05, eps_decay=50_000),   # 8 faster eps decay
    dict(lr=1e-4, gamma=0.99,  batch=32, eps_start=1.0, eps_end=0.10, eps_decay=200_000),  # 9 slower decay, higher floor
    dict(lr=1e-4, gamma=0.99,  batch=32, eps_start=0.5, eps_end=0.01, eps_decay=75_000),   # 10 less early exploration
]


def apply_config(cfg, steps, member, index):
    """Point train's global constants and output paths at this experiment."""
    tag = f"exp{index:02d}"
    model_dir = os.path.join("models", member)
    plot_dir = os.path.join("plots", member)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    train.LEARNING_RATE = cfg["lr"]
    train.GAMMA = cfg["gamma"]
    train.BATCH_SIZE = cfg["batch"]
    train.EPSILON_START = cfg["eps_start"]
    train.EPSILON_END = cfg["eps_end"]
    train.EPSILON_DECAY_STEPS = cfg["eps_decay"]
    if steps is not None:
        train.TOTAL_STEPS = steps

    train.MODEL_PATH = os.path.join(model_dir, f"{tag}.pth")
    train.PLOT_PATH = os.path.join(plot_dir, f"{tag}_rewards.png")
    train.LOG_PATH = os.path.join(plot_dir, f"{tag}_log.csv")
    return train.LOG_PATH


def summarize(log_path):
    """Read a run's episode log and return (#episodes, best, final-20 mean)."""
    rewards = []
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            rewards.append(float(row["reward"]))
    if not rewards:
        return 0, 0.0, 0.0
    best = max(rewards)
    final_mean = float(np.mean(rewards[-20:]))
    return len(rewards), best, final_mean


def main():
    parser = argparse.ArgumentParser(description="Task 3 hyperparameter sweep.")
    parser.add_argument("--only", type=int, default=None,
                        help="run only experiment N (1-10) instead of all")
    parser.add_argument("--steps", type=int, default=None,
                        help="override TOTAL_STEPS (e.g. 15000 for a quick smoke run)")
    parser.add_argument("--member", default=MEMBER, help="member name for output folders")
    args = parser.parse_args()

    if args.only is not None:
        indices = [args.only]
    else:
        indices = range(1, len(EXPERIMENTS) + 1)

    summary_path = os.path.join("models", args.member, "summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    write_header = not os.path.exists(summary_path)
    summary = open(summary_path, "a", newline="")
    writer = csv.writer(summary)
    if write_header:
        writer.writerow(["experiment", "lr", "gamma", "batch", "eps_start",
                         "eps_end", "eps_decay", "episodes", "best_reward",
                         "final20_mean_reward"])

    for i in indices:
        cfg = EXPERIMENTS[i - 1]
        print("\n" + "=" * 70)
        print(f"Experiment {i}: {cfg}")
        print("=" * 70)
        log_path = apply_config(cfg, args.steps, args.member, i)
        train.main()
        episodes, best, final_mean = summarize(log_path)
        writer.writerow([i, cfg["lr"], cfg["gamma"], cfg["batch"], cfg["eps_start"],
                         cfg["eps_end"], cfg["eps_decay"], episodes,
                         round(best, 1), round(final_mean, 2)])
        summary.flush()
        print(f"Experiment {i} done: {episodes} episodes, best {best:.1f}, "
              f"final-20 mean {final_mean:.2f}")

    summary.close()
    print(f"\nAll requested experiments finished. Summary: {summary_path}")


if __name__ == "__main__":
    main()
