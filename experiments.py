"""Task 3 hyperparameter sweep.

Each group member runs 10 experiments, each with a different combination of
hyperparameters, by reusing the training loop in ``train.py`` unchanged.

There are four distinct config sets (A, B, C, D) so every member's 10
combinations are genuinely different from the others'. Pick a set with
``--set`` (or let it default from ``--member``; Eldrige = set A). For each
experiment the harness:

  * overrides the hyperparameter constants in ``train`` for that run,
  * writes the model to ``models/<member>/expNN.pth``,
  * writes the per-episode log to ``plots/<member>/expNN_log.csv``,
  * writes the reward curve to ``plots/<member>/expNN_rewards.png``,
  * appends final metrics to ``models/<member>/summary.csv``.

The "Observed behaviour" column of the results table is filled in from these
logs/plots after the runs finish.

Run all 10 (set from member):  uv run python experiments.py --member bob --set B
Run one experiment:            uv run python experiments.py --member bob --set B --only 3
Quick smoke test:              uv run python experiments.py --member bob --set B --only 3 --steps 15000
"""

import argparse
import csv
import os

import numpy as np

import train  # we override its module-level constants per experiment


def cfg(lr, gamma, batch, eps_start, eps_end, eps_decay):
    return dict(lr=lr, gamma=gamma, batch=batch,
                eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay)


# ---------------------------------------------------------------------------
# Four distinct 10-experiment sets. Within each set, every row changes one or
# two hyperparameters so their effect is isolated; across sets, no full
# combination repeats, so each member's table is their own.
# ---------------------------------------------------------------------------
SETS = {
    # Set A — baseline family, learning rate centred on 1e-4 (Eldrige)
    "A": [
        cfg(1e-4, 0.99,  32, 1.0, 0.05, 100_000),  # 1 baseline
        cfg(5e-4, 0.99,  32, 1.0, 0.05, 100_000),  # 2 higher LR
        cfg(5e-5, 0.99,  32, 1.0, 0.05, 100_000),  # 3 lower LR
        cfg(1e-4, 0.95,  32, 1.0, 0.05, 100_000),  # 4 lower gamma
        cfg(1e-4, 0.999, 32, 1.0, 0.05, 100_000),  # 5 higher gamma
        cfg(1e-4, 0.99,  64, 1.0, 0.05, 100_000),  # 6 larger batch
        cfg(1e-4, 0.99,  16, 1.0, 0.05, 100_000),  # 7 smaller batch
        cfg(1e-4, 0.99,  32, 1.0, 0.05, 50_000),   # 8 faster eps decay
        cfg(1e-4, 0.99,  32, 1.0, 0.10, 200_000),  # 9 slower decay, higher floor
        cfg(1e-4, 0.99,  32, 0.5, 0.01, 75_000),   # 10 less early exploration
    ],
    # Set B — mid learning rate centred on 2.5e-4, batch centred on 64
    "B": [
        cfg(2.5e-4, 0.99,  64,  1.0, 0.05, 100_000),  # 1
        cfg(1e-3,   0.99,  64,  1.0, 0.05, 100_000),  # 2 very high LR
        cfg(1e-4,   0.97,  64,  1.0, 0.05, 100_000),  # 3 lower LR, mid gamma
        cfg(2.5e-4, 0.90,  64,  1.0, 0.05, 100_000),  # 4 low gamma
        cfg(2.5e-4, 0.995, 64,  1.0, 0.05, 100_000),  # 5 high gamma
        cfg(2.5e-4, 0.99,  128, 1.0, 0.05, 100_000),  # 6 very large batch
        cfg(2.5e-4, 0.99,  32,  1.0, 0.05, 100_000),  # 7 smaller batch
        cfg(2.5e-4, 0.99,  64,  1.0, 0.05, 75_000),   # 8 faster decay
        cfg(2.5e-4, 0.99,  64,  1.0, 0.02, 150_000),  # 9 slow decay, low floor
        cfg(2.5e-4, 0.99,  64,  0.8, 0.01, 50_000),   # 10 less early exploration
    ],
    # Set C — low learning rate centred on 5e-5, exploring gamma/batch extremes
    "C": [
        cfg(5e-5, 0.99,  16,  1.0, 0.05, 100_000),  # 1
        cfg(5e-5, 0.99,  128, 1.0, 0.05, 100_000),  # 2 large batch
        cfg(5e-5, 0.90,  32,  1.0, 0.05, 100_000),  # 3 low gamma
        cfg(5e-5, 0.95,  32,  1.0, 0.05, 150_000),  # 4 mid gamma, slow decay
        cfg(5e-5, 0.999, 32,  1.0, 0.05, 100_000),  # 5 high gamma
        cfg(5e-5, 0.99,  64,  1.0, 0.10, 200_000),  # 6 higher floor
        cfg(5e-5, 0.99,  32,  1.0, 0.02, 75_000),   # 7 low floor, fast decay
        cfg(5e-5, 0.99,  32,  0.5, 0.05, 100_000),  # 8 low eps start
        cfg(5e-5, 0.97,  32,  1.0, 0.05, 50_000),   # 9 mid gamma, fast decay
        cfg(5e-5, 0.995, 16,  0.8, 0.01, 150_000),  # 10 mixed
    ],
    # Set D — high learning rate centred on 5e-4 (aggressive updates)
    "D": [
        cfg(5e-4, 0.99,  64,  1.0, 0.05, 100_000),  # 1
        cfg(5e-4, 0.95,  32,  1.0, 0.05, 100_000),  # 2 low gamma
        cfg(5e-4, 0.999, 32,  1.0, 0.05, 100_000),  # 3 high gamma
        cfg(5e-4, 0.99,  16,  1.0, 0.05, 100_000),  # 4 small batch
        cfg(5e-4, 0.99,  128, 1.0, 0.05, 100_000),  # 5 large batch
        cfg(5e-4, 0.99,  32,  1.0, 0.10, 200_000),  # 6 higher floor, slow decay
        cfg(5e-4, 0.99,  32,  1.0, 0.02, 50_000),   # 7 low floor, fast decay
        cfg(5e-4, 0.99,  32,  0.5, 0.01, 75_000),   # 8 less early exploration
        cfg(1e-3, 0.99,  32,  1.0, 0.05, 100_000),  # 9 very high LR
        cfg(1e-3, 0.95,  64,  1.0, 0.05, 150_000),  # 10 very high LR, low gamma
    ],
}

# Default set per known member name (lowercase). Others fall back to --set.
MEMBER_SETS = {"eldrige": "A"}


def apply_config(c, steps, member, index):
    """Point train's global constants and output paths at this experiment."""
    tag = f"exp{index:02d}"
    model_dir = os.path.join("models", member)
    plot_dir = os.path.join("plots", member)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    train.LEARNING_RATE = c["lr"]
    train.GAMMA = c["gamma"]
    train.BATCH_SIZE = c["batch"]
    train.EPSILON_START = c["eps_start"]
    train.EPSILON_END = c["eps_end"]
    train.EPSILON_DECAY_STEPS = c["eps_decay"]
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
    return len(rewards), max(rewards), float(np.mean(rewards[-20:]))


def main():
    parser = argparse.ArgumentParser(description="Task 3 hyperparameter sweep.")
    parser.add_argument("--member", default="eldrige", help="member name for output folders")
    parser.add_argument("--set", dest="set_name", choices=sorted(SETS),
                        help="which config set to use (A-D); defaults from --member")
    parser.add_argument("--only", type=int, default=None,
                        help="run only experiment N (1-10) instead of all")
    parser.add_argument("--steps", type=int, default=None,
                        help="override TOTAL_STEPS (e.g. 15000 for a quick smoke run)")
    args = parser.parse_args()

    set_name = args.set_name or MEMBER_SETS.get(args.member.lower())
    if set_name is None:
        parser.error(f"member '{args.member}' has no default set; pass --set A|B|C|D")
    experiments = SETS[set_name]
    print(f"Member '{args.member}' using config set {set_name} ({len(experiments)} experiments)")

    indices = [args.only] if args.only is not None else range(1, len(experiments) + 1)

    summary_path = os.path.join("models", args.member, "summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    write_header = not os.path.exists(summary_path)
    summary = open(summary_path, "a", newline="")
    writer = csv.writer(summary)
    if write_header:
        writer.writerow(["set", "experiment", "lr", "gamma", "batch", "eps_start",
                         "eps_end", "eps_decay", "episodes", "best_reward",
                         "final20_mean_reward"])

    for i in indices:
        c = experiments[i - 1]
        print("\n" + "=" * 70)
        print(f"Set {set_name} | Experiment {i}: {c}")
        print("=" * 70)
        log_path = apply_config(c, args.steps, args.member, i)
        train.main()
        episodes, best, final_mean = summarize(log_path)
        writer.writerow([set_name, i, c["lr"], c["gamma"], c["batch"], c["eps_start"],
                         c["eps_end"], c["eps_decay"], episodes,
                         round(best, 1), round(final_mean, 2)])
        summary.flush()
        print(f"Experiment {i} done: {episodes} episodes, best {best:.1f}, "
              f"final-20 mean {final_mean:.2f}")

    summary.close()
    print(f"\nAll requested experiments finished. Summary: {summary_path}")


if __name__ == "__main__":
    main()
