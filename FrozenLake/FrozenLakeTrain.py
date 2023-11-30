import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

# Create the FrozenLake environment (non-slippery)
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")

# Initialize the PPO model
model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate = 0.003, 
        ent_coef=0.07, 
        gamma=0.99, 
        n_steps=2048, 
        batch_size=64, 
        n_epochs=10, 
        tensorboard_log="./log/tb_logs/training")

# Train the agent
model.learn(total_timesteps=100000)

for i in range(10):
    env = gym.make("FrozenLake-v1", desc=grm(size=4, p=0.7), is_slippery=False, render_mode="rgb_array")
    model.learn(total_timesteps=10000)

# Save the model
model.save("./log/model/training/frozenlake_training")

env = RecordVideo(env, video_folder="recordings/training")

# Remake environment with human mode to visualize the agent
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
obs, info = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action.item())
    
    done = terminated or truncated

env.close()
