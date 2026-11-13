# Task 3: Hyperparameter Tuning — Experiment Results

**Member Name:** Eldrige

These 10 experiments are defined in [`experiments.py`](experiments.py). Run them with:

```bash
uv run python experiments.py            # all 10 (long)
uv run python experiments.py --only 3   # a single experiment
```

Each run saves a model to `models/eldrige/expNN.pth`, an episode log to
`plots/eldrige/expNN_log.csv`, and a reward curve to
`plots/eldrige/expNN_rewards.png`. Final metrics are collected in
`models/eldrige/summary.csv`. Fill in the **Observed behaviour** column from
those logs/plots after each run (e.g. best reward, final 20-episode mean,
whether learning was stable or diverged).

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1  | 0.0001 | 0.99  | 32 | 1.0 | 0.05 | 100,000 | Stable baseline; settled ~11.2, best 26. |
| 2  | 0.0005 | 0.99  | 32 | 1.0 | 0.05 | 100,000 | Diverged — LR too high; reward collapsed to ~0.85, agent died instantly (most episodes, 641). |
| 3  | 0.00005| 0.99  | 32 | 1.0 | 0.05 | 100,000 | Learned but slowly (LR too low); only reached ~4.25. |
| 4  | 0.0001 | 0.95  | 32 | 1.0 | 0.05 | 100,000 | Short-sighted (low γ); underperformed baseline at ~7.3. |
| 5  | 0.0001 | 0.999 | 32 | 1.0 | 0.05 | 100,000 | Slow credit assignment (high γ); ~6.35, below baseline. |
| 6  | 0.0001 | 0.99  | 64 | 1.0 | 0.05 | 100,000 | Matched baseline (~11.6); larger batch gave stable gradients. |
| 7  | 0.0001 | 0.99  | 16 | 1.0 | 0.05 | 100,000 | Noisier small-batch gradients; weaker at ~5.75. |
| 8  | 0.0001 | 0.99  | 32 | 1.0 | 0.05 | 50,000  | Best run — fast ε-decay let it exploit sooner; settled ~14.75, best 40. |
| 9  | 0.0001 | 0.99  | 32 | 1.0 | 0.10 | 200,000 | Over-explored (slow decay + 0.1 floor); capped at ~6.85. |
| 10 | 0.0001 | 0.99  | 32 | 0.5 | 0.01 | 75,000  | Low ε-start aided early exploitation; solid ~8.15. |

## How each experiment isolates a hyperparameter

- **1** is the baseline (the defaults from `train.py`).
- **2–3** change only the **learning rate** (higher / lower) to see its effect on stability and speed.
- **4–5** change only **gamma** to compare short- vs long-horizon credit assignment.
- **6–7** change only the **batch size** (larger / smaller).
- **8–10** vary the **epsilon schedule** (decay speed, final floor, and start value) to study the exploration/exploitation trade-off.

## Metrics to report per run (from `summary.csv`)

- `episodes` — number of completed episodes in the run.
- `best_reward` — highest single-episode reward reached.
- `final20_mean_reward` — mean reward over the last 20 episodes (a measure of settled performance).
