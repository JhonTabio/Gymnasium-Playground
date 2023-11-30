import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

# Continue with transfer learning on the slippery environment
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="rgb_array")

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
        tensorboard_log="./log/tb_logs/testing")


# Load the previously trained model
model = PPO.load("./log/model/training/frozenlake_training", env)

# Update the model's environment
#model.set_env(env)

# Continue training on the new environment
model.learn(total_timesteps=100000)

# Save the trained model
model.save("./log/model/testing/frozenlake_testing")

env = RecordVideo(env, video_folder="recordings/testing")

# Test the agent
#env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
obs, info = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action.item())

    done = terminated or truncated

env.close()
