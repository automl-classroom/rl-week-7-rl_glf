import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve

n_seeds = 2

# Read baseline DQN data
dqn_s0 = pd.read_csv("demo_data_seed_0.csv")
dqn_s1 = pd.read_csv("demo_data_seed_1.csv")

# Read new RND-DQN data
rnd_s0 = pd.read_csv("rnd_dqn_seed_0.csv")
rnd_s1 = pd.read_csv("rnd_dqn_seed_1.csv")

# Convert rewards to numpy and truncate to same length
dqn_0 = dqn_s0["rewards"].to_numpy()
dqn_1 = dqn_s1["rewards"].to_numpy()
rnd_0 = rnd_s0["rewards"].to_numpy()
rnd_1 = rnd_s1["rewards"].to_numpy()

min_len = min(len(dqn_0), len(dqn_1), len(rnd_0), len(rnd_1))

dqn_rewards = np.stack([dqn_0[:min_len], dqn_1[:min_len]])
rnd_rewards = np.stack([rnd_0[:min_len], rnd_1[:min_len]])
steps = dqn_s0["steps"].to_numpy()[:min_len]

# You can add other algorithms here
train_scores = {
    "dqn": dqn_rewards,
    "rnd_dqn": rnd_rewards,
}

# This aggregates only IQM, but other options include mean and median
iqm = lambda scores: np.array(
    [metrics.aggregate_iqm(scores[:, i]) for i in range(scores.shape[1])]
)
iqm_scores, iqm_cis = get_interval_estimates(
    train_scores,
    iqm,
    reps=2000,
)

# This is a utility function, but you can also just use a normal line plot with the IQM and CI scores
plot_sample_efficiency_curve(
    steps + 1,
    iqm_scores,
    iqm_cis,
    algorithms=["dqn", "rnd_dqn"],
    xlabel=r"Number of Evaluations",
    ylabel="IQM Normalized Score",
)
plt.gcf().canvas.manager.set_window_title(
    "IQM Normalized Score - DQN vs RND-DQN"
)
plt.legend()
plt.tight_layout()
plt.savefig("comparison_plot_rnd_vs_dqn.png")
plt.show()
