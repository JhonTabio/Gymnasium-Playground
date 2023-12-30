import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
#from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.envs.toy_text.frozen_lake import generate_random_map as grm
import matplotlib.pyplot as plt

# Create a custom callback
# Since internal data is not easily accessible during training
class DataLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(DataLogger, self).__init__(verbose)
        self.rewards = []
        self.current_reward = 0
        self.inital_coef = 0.1
        #self.inital_coef = self.locals["self"].ent_coef

    def _on_step(self) -> bool:
        #print(self.locals["self"].ent_coef)
        #self.rewards.append(self.locals["rewards"])
        self.current_reward += self.locals["rewards"][0]

        if self.locals["dones"][0]:
            self.rewards.append(self.current_reward)
            self.current_reward = 0

        """
        reward_avg = sum(self.rewards[-3:]) / 3

        if reward_avg:
            self.locals["self"].ent_coef = max(self.inital_coef, self.locals["self"].ent_coef * 0.001)
        else:
            self.locals["self"].ent_coef *= 1.001
        
        #print(self.model.policy.optimizer.param_groups)
        """

        return super()._on_step()

num_envs = 5

# Create the FrozenLake environment (non-slippery)
# Create a list of envs, each with different maps to train on
envs = [gym.make("FrozenLake-v1", desc=grm(size=4, p=0.7), is_slippery=False, render_mode="rgb_array") for _ in range(num_envs)]

log_dirs = ["./log/tb_logs/training/parallel/env_{}".format(i) for i in range(num_envs)]
#env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
#env = gym.make("FrozenLake-v1", desc=grm(size=4, p=0.7), is_slippery=False, render_mode="rgb_array")

# Create a vectorized envrionment, each with different maps to train on
vec_env = DummyVecEnv([(lambda env: lambda: env) (env) for env in envs])
#vec_env = DummyVecEnv(lambda: test)

# Initialize the PPO model
""""
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
        tensorboard_log="./log/tb_logs/training")"""

model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./log/tb_logs/training/")

# Store rewards during training
#data_callback = DataLogger()
rewards_eval = []

# Train the agent
model.learn(total_timesteps=20000)
mean_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)[0]
rewards_eval.append(mean_reward)
#rewards_eval.append(evaluate_policy(model, vec_env, n_eval_episodes=10)[0])

""""
i = 0
j = 0
# Train the agent on randomzied enviornments (To prevent overfitting to single env)
while True:
    if i >= 5:
        break

    if j >= 100:
        break

    # If agent passed, make new env. If not, repeat training
    if mean_reward:
        env = gym.make("FrozenLake-v1", desc=grm(size=4, p=0.7), is_slippery=False, render_mode="rgb_array")
        i += 1
        print("New env")

    model.learn(total_timesteps=25, callback=data_callback)
    mean_reward = evaluate_policy(model, env, n_eval_episodes=10)[0]
    print(mean_reward, " is the mean reward")
    rewards_eval.append(mean_reward)
    j +=1
    #rewards_eval.append(evaluate_policy(model, env, n_eval_episodes=10)[0])
"""

# Save the model
model.save("./log/model/training/frozenlake_training")

# Plot the rewards throughout training
plt.plot(rewards_eval)
plt.xlabel("Training Step")
plt.ylabel("Reward")
plt.title("Agent's rewards during training")
plt.savefig("recordings/training/reward.png")

# Reset back to the default for video export
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")

# Start a video to show agent after training
env = RecordVideo(env, video_folder="recordings/training/")

obs, info = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action.item())
    
    done = terminated or truncated

env.close()
