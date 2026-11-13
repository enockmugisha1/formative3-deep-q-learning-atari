"""Watch a trained DQN agent play Atari Breakout (Task 4).

This script loads the weights saved by ``train.py`` into the exact same
Q-network, then runs the agent in evaluation mode: it always takes the
action with the highest predicted Q-value (greedy, no exploration) and
renders the game window in real time.

Run with:  uv run python play.py
Optional:  uv run python play.py --model dqn_model.pth --episodes 5
"""

import argparse
import time

import ale_py
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

# Reuse the exact network definition from training so the architecture
# we load the weights into is guaranteed to match the one that saved them.
from train import QNetwork

# Breakout action indices (see README action table)
FIRE_ACTION = 1


def make_eval_env(render_mode="human"):
    """Breakout with the same preprocessing as training, but rendered live.

    ``render_mode="human"`` opens a window and draws the full-colour game in
    real time. The observation the agent receives is still the 4x84x84 stack
    of grayscale frames it was trained on; rendering only changes what is
    shown on screen, not what the network sees.
    """
    gym.register_envs(ale_py)
    env = gym.make(
        "ALE/Breakout-v5",
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode=render_mode,
    )
    env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
    env = FrameStackObservation(env, stack_size=4)
    return env


def greedy_action(online_net, state, device):
    """Always pick the action with the highest Q-value (no exploration)."""
    state_tensor = torch.tensor(np.array(state), device=device).unsqueeze(0).float()
    with torch.no_grad():
        q_values = online_net(state_tensor)
    return int(q_values.argmax(dim=1).item())


def play(model_path, episodes, device, fps):
    env = make_eval_env(render_mode="human")
    num_actions = env.action_space.n

    # Rebuild the network and load the trained weights.
    net = QNetwork(num_actions).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()  # evaluation mode: no dropout/batchnorm updates (none here, but good practice)
    print(f"Loaded model from {model_path} onto {device}")

    frame_delay = 1.0 / fps if fps > 0 else 0.0

    for episode in range(1, episodes + 1):
        state, info = env.reset()
        # Breakout only launches the ball after FIRE is pressed, so the game
        # would stall if the agent never happens to choose FIRE. We press it
        # once at the start and again whenever a life is lost.
        state, _, _, _, info = env.step(FIRE_ACTION)
        lives = info.get("lives", 0)

        done = False
        total_reward = 0.0
        while not done:
            action = greedy_action(net, state, device)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Re-fire after losing a life so play continues.
            if info.get("lives", lives) < lives:
                lives = info["lives"]
                state, _, terminated, truncated, info = env.step(FIRE_ACTION)

            done = terminated or truncated
            if frame_delay:
                time.sleep(frame_delay)

        print(f"Episode {episode:2d} | reward {total_reward:5.1f}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Watch a trained DQN play Breakout.")
    parser.add_argument("--model", default="dqn_model.pth", help="path to the saved weights")
    parser.add_argument("--episodes", type=int, default=5, help="number of games to play")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="frames per second cap for watchable playback (0 = no cap)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    play(args.model, args.episodes, device, args.fps)


if __name__ == "__main__":
    main()
