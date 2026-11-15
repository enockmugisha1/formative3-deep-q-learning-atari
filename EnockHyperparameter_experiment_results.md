# Task 3: Hyperparameter Tuning — Experiment Results

**Member Name:** Enock

My experiments use **config set C** in [`experiments.py`](experiments.py). Run them with:

```bash
uv run python experiments.py --member enock --set C            # all 10 (long)
uv run python experiments.py --member enock --set C --only 3   # a single experiment
uv run python experiments.py --member enock --set C --only 1 --steps 15000  # quick smoke test
```

Each run saves a model to `models/enock/expNN.pth`, an episode log to
`plots/enock/expNN_log.csv`, and a reward curve to `plots/enock/expNN_rewards.png`.
Final metrics are collected in `models/enock/summary.csv`. Fill in the
**Observed behaviour** column from those logs/plots after each run (e.g. best
reward, final 20-episode mean, whether learning was stable or diverged).

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1  | 0.00005 | 0.99 | 16 | 1 | 0.05 | 100,000 | **Run (150k steps, 642 episodes). Best 13, final-20 mean 4.0.** Stable, steady learning — mean reward roughly doubled over training (quintile means 1.45 → 1.51 → 1.74 → 2.27 → 3.50), no divergence. A solid mid-table result; the same config with a much larger batch (exp 2) learned far better. |
| 2  | 0.00005 | 0.99 | 128 | 1 | 0.05 | 100,000 | **Run (150k steps, 570 episodes). Best 24, final-20 mean 10.3.** Quintile means 1.39 → 1.32 → 2.18 → 4.19 → 6.97 (rising learning curve). |
| 3  | 0.00005 | 0.9 | 32 | 1 | 0.05 | 100,000 | **Run (150k steps, 678 episodes). Best 12, final-20 mean 2.0.** Learned, but noticeably slower and plateaued lower than exp 1 (quintile means 1.20 → 1.04 → 1.11 → 1.49 → 2.10). The lower discount factor (γ=0.90) shortens the reward horizon to ~10 steps, so the agent struggles to credit the delayed reward of breaking a brick several frames after the paddle moves — it behaves more myopically. **Worst configuration overall — the low γ was the single most harmful change.** |
| 4  | 0.00005 | 0.95 | 32 | 1 | 0.05 | 150,000 | **Run (150k steps, 722 episodes). Best 13, final-20 mean 2.75.** Quintile means 1.15 → 1.56 → 1.16 → 1.43 → 2.68 (rising learning curve). |
| 5  | 0.00005 | 0.999 | 32 | 1 | 0.05 | 100,000 | **Run (150k steps, 647 episodes). Best 21, final-20 mean 5.2.** Quintile means 1.43 → 1.19 → 1.94 → 2.4 → 4.28 (rising learning curve). |
| 6  | 0.00005 | 0.99 | 64 | 1 | 0.1 | 200,000 | **Run (150k steps, 657 episodes). Best 13, final-20 mean 5.5.** Quintile means 1.43 → 1.42 → 1.83 → 3.23 → 4.19 (rising learning curve). |
| 7  | 0.00005 | 0.99 | 32 | 1 | 0.02 | 75,000 | **Run (150k steps, 593 episodes). Best 12, final-20 mean 4.5.** Quintile means 1.46 → 1.39 → 1.42 → 2.75 → 4.45 (rising learning curve). |
| 8  | 0.00005 | 0.99 | 32 | 0.5 | 0.05 | 100,000 | **Run (150k steps, 585 episodes). Best 25, final-20 mean 5.4.** Quintile means 1.19 → 1.5 → 2.37 → 3.56 → 4.72 (rising learning curve). |
| 9  | 0.00005 | 0.97 | 32 | 1 | 0.05 | 50,000 | **Run (150k steps, 561 episodes). Best 13, final-20 mean 6.8.** Quintile means 1.32 → 0.91 → 2.74 → 2.85 → 5.15 (rising learning curve). |
| 10 | 0.00005 | 0.995 | 16 | 0.8 | 0.01 | 150,000 | **Run (150k steps, 696 episodes). Best 11, final-20 mean 3.35.** Quintile means 1.22 → 1.29 → 1.72 → 2.39 → 2.56 (rising learning curve). |

## Metrics to report per run (from `summary.csv`)

- `episodes` — number of completed episodes in the run.
- `best_reward` — highest single-episode reward reached.
- `final20_mean_reward` — mean reward over the last 20 episodes (settled performance).

## Summary of my runs

I ran **all 10** experiments from set C, each for the full 150,000 steps. Ranked by settled performance (mean reward over the last 20 episodes):

| Rank | Exp | γ | Batch | ε start→end (decay) | Best | Final-20 mean |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **2** | 0.99 | **128** | 1.0→0.05 (100k) | 24 | **10.3** |
| 2 | 9 | 0.97 | 32 | 1.0→0.05 (50k) | 13 | 6.8 |
| 3 | 6 | 0.99 | 64 | 1.0→0.10 (200k) | 13 | 5.5 |
| 4 | 8 | 0.99 | 32 | 0.5→0.05 (100k) | 25 | 5.4 |
| 5 | 5 | 0.999 | 32 | 1.0→0.05 (100k) | 21 | 5.2 |
| 6 | 7 | 0.99 | 32 | 1.0→0.02 (75k) | 12 | 4.5 |
| 7 | 1 | 0.99 | 16 | 1.0→0.05 (100k) | 13 | 4.0 |
| 8 | 10 | 0.995 | 16 | 0.8→0.01 (150k) | 11 | 3.35 |
| 9 | 4 | 0.95 | 32 | 1.0→0.05 (150k) | 13 | 2.75 |
| 10 | 3 | 0.90 | 32 | 1.0→0.05 (100k) | 12 | 2.0 |

**Best configuration — Experiment 2** (γ = 0.99, **batch 128**). Highest settled performance by a wide margin (final-20 mean 10.3, best episode 24) with a steadily rising curve. The only trade-off was training time: on a CPU, the batch-128 gradient updates were ~4× slower than batch 32.

**Worst configuration — Experiment 3** (γ = 0.90, batch 32). It learned but plateaued at the lowest settled reward (2.0).

### Which hyperparameters helped or harmed

Two effects stand out, each visible in a clean single-variable comparison from my runs:

- **Batch size — the biggest driver (helped most).** Comparing exp 1 and exp 2, which are *identical except batch size*: batch 16 → mean 4.0, batch **128 → mean 10.3**. A larger batch averages more experiences per update, giving a less noisy gradient and more stable, faster learning. The trend holds across the table (batch 128 > 64 > 32 ≈ 16).
- **Discount factor γ — second biggest (low γ harmed most).** Comparing exp 3 and exp 5, *identical except γ*: γ = 0.90 → mean 2.0, γ = **0.999 → mean 5.2**. Breakout rewards the agent only when the ball breaks a brick, several frames after the paddle moved, so a longer reward horizon (higher γ) is needed to credit those delayed rewards. γ = 0.90 is too short-sighted.
- **Exploration schedule — smaller effect.** Reaching the exploitation phase a bit sooner (faster ε-decay, exp 9; or a lower ε start, exp 8) gave modest gains, but mattered much less than batch size or γ.

**Final chosen configuration:** learning rate 5e-5, γ = 0.99, **batch size 128**, ε from 1.0 → 0.05 decayed over 100k steps (**Experiment 2**). Its saved model is `models/enock/exp02.pth`.

*Note: all rewards are modest in absolute terms because each run was limited to 150k steps on CPU; Atari Breakout typically needs millions of steps to reach high scores. The 10 runs are directly comparable because they used an identical 150k-step training budget.*
