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
| 1  | 0.00005 | 0.99 | 16 | 1 | 0.05 | 100,000 | **Run (150k steps, 642 episodes). Best 13, final-20 mean 4.0.** Stable, steady learning — mean reward roughly doubled over training (quintile means 1.45 → 1.51 → 1.74 → 2.27 → 3.50), no divergence. The high discount (γ=0.99) and small batch gave the best settled performance of my two runs. **Best configuration.** |
| 2  | 0.00005 | 0.99 | 128 | 1 | 0.05 | 100,000 | _(not run — batch 128 was prohibitively slow on this CPU-only machine)_ |
| 3  | 0.00005 | 0.9 | 32 | 1 | 0.05 | 100,000 | **Run (150k steps, 678 episodes). Best 12, final-20 mean 2.0.** Learned, but noticeably slower and plateaued lower than exp 1 (quintile means 1.20 → 1.04 → 1.11 → 1.49 → 2.10). The lower discount factor (γ=0.90) shortens the reward horizon to ~10 steps, so the agent struggles to credit the delayed reward of breaking a brick several frames after the paddle moves — it behaves more myopically. **Worst of my two runs; γ was the harmful change.** |
| 4  | 0.00005 | 0.95 | 32 | 1 | 0.05 | 150,000 | **Run (150k steps, 722 episodes). Best 13, final-20 mean 2.75.** Quintile means 1.15 → 1.56 → 1.16 → 1.43 → 2.68 (rising learning curve). |
| 5  | 0.00005 | 0.999 | 32 | 1 | 0.05 | 100,000 | **Run (150k steps, 647 episodes). Best 21, final-20 mean 5.2.** Quintile means 1.43 → 1.19 → 1.94 → 2.4 → 4.28 (rising learning curve). |
| 6  | 0.00005 | 0.99 | 64 | 1 | 0.1 | 200,000 | _(fill after run)_ |
| 7  | 0.00005 | 0.99 | 32 | 1 | 0.02 | 75,000 | **Run (150k steps, 593 episodes). Best 12, final-20 mean 4.5.** Quintile means 1.46 → 1.39 → 1.42 → 2.75 → 4.45 (rising learning curve). |
| 8  | 0.00005 | 0.99 | 32 | 0.5 | 0.05 | 100,000 | _(fill after run)_ |
| 9  | 0.00005 | 0.97 | 32 | 1 | 0.05 | 50,000 | _(fill after run)_ |
| 10 | 0.00005 | 0.995 | 16 | 0.8 | 0.01 | 150,000 | _(fill after run)_ |

## Metrics to report per run (from `summary.csv`)

- `episodes` — number of completed episodes in the run.
- `best_reward` — highest single-episode reward reached.
- `final20_mean_reward` — mean reward over the last 20 episodes (settled performance).

## Summary of my runs

I ran **two** experiments from set C, **experiment 1** and **experiment 3**, each for the full 150,000 steps. (Experiment 2 uses batch size 128, which was prohibitively slow to train on this CPU-only machine, so I used experiment 3 as my second configuration instead.)

| Exp | γ | Batch | Episodes | Best reward | Final-20 mean | Plot |
|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1 | 0.99 | 16 | 642 | **13** | **4.0** | `plots/enock/exp01_rewards.png` |
| 3 | 0.90 | 32 | 678 | 12 | 2.0 | `plots/enock/exp03_rewards.png` |

**Best configuration — Experiment 1** (γ = 0.99, batch 16). It gave the highest settled performance (final-20 mean 4.0) with a stable, steadily rising reward curve and no divergence.

**Worst configuration — Experiment 3** (γ = 0.90, batch 32). It still learned but plateaued at about half the settled reward of exp 1.

**Which hyperparameter mattered most — the discount factor γ.** Breakout only rewards the agent when the ball breaks a brick, which happens several frames *after* the paddle action that aimed the bounce. A high γ (0.99) gives an effective reward horizon of ~100 steps, so the agent can connect its actions to those delayed rewards. Dropping to γ = 0.90 shortens the horizon to ~10 steps, making the agent short-sighted and slowing its learning — the clear cause of exp 3's weaker result.

**Final chosen configuration:** learning rate 5e-5, **γ = 0.99**, batch size 16, ε from 1.0 → 0.05 decayed over 100k steps (Experiment 1). Its saved model is `models/enock/exp01.pth`.

*Note: all rewards are modest because each run was limited to 150k steps on CPU; Atari Breakout typically needs millions of steps to reach high scores. The experiments are still directly comparable because they used identical training budgets.*
