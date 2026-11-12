import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)

# Raw environment: frameskip=1 and no sticky actions, matching the README
env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0.0)
env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
env = FrameStackObservation(env, stack_size=4)

print("Observation shape:", env.observation_space.shape)  # Expect (4, 84, 84)
print("Actions:", env.action_space)                       # Expect Discrete(4)

obs, info = env.reset(seed=42)
total_reward = 0.0
for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        obs, info = env.reset()
env.close()
print("Random agent reward over 500 steps:", total_reward)
