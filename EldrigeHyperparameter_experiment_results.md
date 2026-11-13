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
| 1  | 0.0001 | 0.99  | 32 | 1.0 | 0.05 | 100,000 | _Baseline — fill in after run_ |
| 2  | 0.0005 | 0.99  | 32 | 1.0 | 0.05 | 100,000 | _Higher learning rate — fill in after run_ |
| 3  | 0.00005| 0.99  | 32 | 1.0 | 0.05 | 100,000 | _Lower learning rate — fill in after run_ |
| 4  | 0.0001 | 0.95  | 32 | 1.0 | 0.05 | 100,000 | _Lower discount (shorter horizon) — fill in after run_ |
| 5  | 0.0001 | 0.999 | 32 | 1.0 | 0.05 | 100,000 | _Higher discount (longer horizon) — fill in after run_ |
| 6  | 0.0001 | 0.99  | 64 | 1.0 | 0.05 | 100,000 | _Larger batch — fill in after run_ |
| 7  | 0.0001 | 0.99  | 16 | 1.0 | 0.05 | 100,000 | _Smaller batch — fill in after run_ |
| 8  | 0.0001 | 0.99  | 32 | 1.0 | 0.05 | 50,000  | _Faster epsilon decay — fill in after run_ |
| 9  | 0.0001 | 0.99  | 32 | 1.0 | 0.10 | 200,000 | _Slower decay, higher exploration floor — fill in after run_ |
| 10 | 0.0001 | 0.99  | 32 | 0.5 | 0.01 | 75,000  | _Less early exploration — fill in after run_ |

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
