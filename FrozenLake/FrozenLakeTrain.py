import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
from gymnasium.envs.toy_text.frozen_lake import generate_random_map as grm

# Create the FrozenLake environment (non-slippery)
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")

# Initialize the PPO model
model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate = 0.007, 
        ent_coef=0.09, 
        gamma=0.99, 
        n_steps=2048, 
        batch_size=64, 
        n_epochs=10, 
        tensorboard_log="./log/tb_logs/training")

# Store learning rates during training
learning_rates = []

# Train the agent
model.learn(total_timesteps=2000)
learning_rates.append(model.learning_rate)

# Train the agent on randomzied enviornments (To prevent overfitting to single env)
for i in range(20):
    env = gym.make("FrozenLake-v1", desc=grm(size=4, p=0.7), is_slippery=False, render_mode="rgb_array")
    model.learn(total_timesteps=5000)
    learning_rates.append(model.learning_rate)

# Save the model
model.save("./log/model/training/frozenlake_training")

env = RecordVideo(env, video_folder="recordings/training")

# Reset back to the default for video export
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
obs, info = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action.item())
    
    done = terminated or truncated

env.close()
