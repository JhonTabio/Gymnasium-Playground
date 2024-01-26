import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.envs.toy_text.frozen_lake import generate_random_map as grm

#env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
env = gym.make("FrozenLake-v1", desc=grm(size=4, p=0.7), is_slippery=False, render_mode="rgb_array")

# Load the previously trained model
model = PPO.load("./log/model/training/frozenlake_training", env)

for i in range(5):
    mean_reward = evaluate_policy(model, env, n_eval_episodes=10)[0]
    print(f"Mean Reward: {mean_reward}")

env = RecordVideo(env, video_folder="recordings/testing")

# Test the agent
obs, info = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action.item())

    done = terminated or truncated

env.close()
