import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO

# Create the FrozenLake-V1 environment
env = gym.make("LunarLander-v2", render_mode="human")

"""
# Initialize the A2C model
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.005, tensorboard_log="./log/tb_logs/")

# Train the model
model.learn(total_timesteps=500_000)  

# Save the best model
model.save("./log/model/lunar_model")
"""

# Load the best model
model = PPO.load("./log/model/lunar_model.zip")

# Visualize the agent solving the puzzle
obs, info = env.reset()
done = False

while not done:
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action.item())
    env.render()
    
    done = terminated or truncated
"""
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render(mode="human")
"""

# Close the environment
env.close()
