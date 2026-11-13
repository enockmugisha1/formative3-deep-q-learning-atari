"""Train a DQN agent to play Atari Breakout (Task 2).

The agent sees 4 stacked grayscale frames of 84x84 pixels and must choose
one of 4 actions (NOOP, FIRE, RIGHT, LEFT). It learns by storing its
experiences in a replay buffer and using them to update a convolutional
Q-network with the Bellman equation.

Run with:  uv run python train.py
"""

import csv
import os
import random
from collections import deque

import ale_py
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")  # save plots to a file, no display window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

# ---------------------------------------------------------------------------
# Hyperparameters (edit these for each experiment in Task 3)
# ---------------------------------------------------------------------------
LEARNING_RATE = 0.0001       # step size for the optimizer
GAMMA = 0.99                 # discount factor for future rewards
BATCH_SIZE = 32              # experiences sampled per training update
EPSILON_START = 1.0          # starting exploration rate (100% random actions)
EPSILON_END = 0.05           # final exploration rate
EPSILON_DECAY_STEPS = 100_000  # steps to decay epsilon from start to end

# Fixed training settings

TOTAL_STEPS = 150_000        # total environment steps to train for
BUFFER_SIZE = 20_000         # maximum experiences kept in the replay buffer
LEARNING_STARTS = 5_000      # collect this many steps before training begins
TRAIN_EVERY = 4              # do one training update every 4 environment steps
TARGET_UPDATE_EVERY = 1_000  # copy online weights to target net every N steps
MODEL_PATH = "dqn_model.pth"
PLOT_PATH = "plots/training_rewards.png"
LOG_PATH = "plots/training_log.csv"


def make_env():
    """Create the Breakout environment with the same preprocessing as Task 1."""
    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0.0)
    env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
    env = FrameStackObservation(env, stack_size=4)
    return env


class QNetwork(nn.Module):
    """CNN that maps a stack of 4 game frames to one Q-value per action."""

    def __init__(self, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),   # (4,84,84) -> (32,20,20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> (64,9,9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> (64,7,7)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        # Pixel values are 0-255, scale them to 0-1 before the conv layers
        return self.network(x / 255.0)


class ReplayBuffer:
    """Stores past experiences and hands back random batches for training."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def compute_epsilon(step):
    """Linearly decay epsilon from EPSILON_START to EPSILON_END."""
    fraction = min(step / EPSILON_DECAY_STEPS, 1.0)
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)


def select_action(online_net, state, epsilon, num_actions, device):
    """Epsilon-greedy: random action with probability epsilon, else best Q."""
    if random.random() < epsilon:
        return random.randrange(num_actions)
    state_tensor = torch.tensor(np.array(state), device=device).unsqueeze(0).float()
    with torch.no_grad():
        q_values = online_net(state_tensor)
    return int(q_values.argmax(dim=1).item())


def train_step(online_net, target_net, buffer, optimizer, loss_fn, device):
    """Sample a batch and do one gradient update on the online network."""
    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

    states = torch.tensor(states, device=device).float()
    actions = torch.tensor(actions, device=device).long()
    rewards = torch.tensor(rewards, device=device)
    next_states = torch.tensor(next_states, device=device).float()
    dones = torch.tensor(dones, device=device)

    # Q-values the online network currently predicts for the actions we took
    predicted_q = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Bellman target: reward + gamma * best Q-value of the next state.
    # The target network is used here so the target stays stable,
    # and (1 - done) zeroes out the future term for terminal states.
    with torch.no_grad():
        max_next_q = target_net(next_states).max(dim=1).values
        target_q = rewards + GAMMA * max_next_q * (1.0 - dones)

    loss = loss_fn(predicted_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def save_plot(episode_rewards):
    """Plot the reward of every episode plus a moving average."""
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.4, label="Episode reward")
    if len(episode_rewards) >= 20:
        window = 20
        moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(episode_rewards)), moving_avg,
                 label=f"{window}-episode moving average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN training on Breakout")
    plt.legend()
    plt.savefig(PLOT_PATH)
    plt.close()


def main():
    # Use a GPU if one is available (cuda on Nvidia, mps on Apple), else CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")

    env = make_env()
    num_actions = env.action_space.n

    # Online network learns every step; target network is a frozen copy
    # that provides stable Bellman targets and is refreshed periodically.
    online_net = QNetwork(num_actions).to(device)
    target_net = QNetwork(num_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.Adam(online_net.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.SmoothL1Loss()  # Huber loss, the standard choice for DQN
    buffer = ReplayBuffer(BUFFER_SIZE)

    os.makedirs("plots", exist_ok=True)
    log_file = open(LOG_PATH, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["episode", "episode_length", "total_steps", "reward", "epsilon"])

    state, _ = env.reset(seed=42)
    episode_reward = 0.0
    episode_length = 0
    episode_rewards = []

    for step in range(1, TOTAL_STEPS + 1):
        # 1. Pick an action (epsilon-greedy) and take it in the environment
        epsilon = compute_epsilon(step)
        action = select_action(online_net, state, epsilon, num_actions, device)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 2. Store the experience in the replay buffer
        buffer.add(np.array(state), action, reward, np.array(next_state), done)
        state = next_state
        episode_reward += reward
        episode_length += 1

        # 3. Train the online network from a random batch of past experiences
        if step > LEARNING_STARTS and step % TRAIN_EVERY == 0:
            train_step(online_net, target_net, buffer, optimizer, loss_fn, device)

        # 4. Periodically copy the online weights into the target network
        if step % TARGET_UPDATE_EVERY == 0:
            target_net.load_state_dict(online_net.state_dict())

        # 5. When an episode ends, log it and start a new one
        if done:
            episode_rewards.append(episode_reward)
            episode_number = len(episode_rewards)
            log_writer.writerow([episode_number, episode_length, step,
                                 episode_reward, round(epsilon, 3)])
            print(f"Episode {episode_number:4d} | steps {step:7d} | "
                  f"length {episode_length:5d} | reward {episode_reward:5.1f} | "
                  f"epsilon {epsilon:.3f}")
            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0

        # Save progress every 10,000 steps in case training is interrupted
        if step % 10_000 == 0:
            torch.save(online_net.state_dict(), MODEL_PATH)
            save_plot(episode_rewards)
            log_file.flush()

    # Final save
    torch.save(online_net.state_dict(), MODEL_PATH)
    save_plot(episode_rewards)
    log_file.close()
    env.close()
    print(f"Done. Model saved to {MODEL_PATH}, plot saved to {PLOT_PATH}")


if __name__ == "__main__":
    main()
