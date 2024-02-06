import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
#from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.envs.toy_text.frozen_lake import generate_random_map as grm
import matplotlib.pyplot as plt
import numpy as np

# Create a custom callback
# Since internal data is not easily accessible during training
class DataLogger(BaseCallback):
    def __init__(self, check_freq, save_path="./log/model/training/frozenlake_training", verbose=0):
        super(DataLogger, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.episode_rewards = []
        self.episode_steps = []
        self.reward_avg = []
        self.steps_avg = []

        self.best_reward_avg = -np.inf

    def _on_step(self) -> bool:
        print("----------sTATS---------")
        print(self.locals,"\n")
        print(self.locals["n_steps"], "\n")
        print(self.locals["rewards"])
        print("---------------END STATS-----------")

       # Once an episode ends, new information is available 
        for info in self.locals['infos']:
            if 'episode' in info.keys():
                self.episode_rewards.append(info['episode']['r'])
                self.episode_steps.append(info['episode']['l'])
                #self.episode_successes.append(info['is_success'] if 'is_success' in info else False)

        # Check if we reached our frequency mark
        if self.n_calls % self.check_freq == 0:
            self.reward_avg.append(np.mean(self.episode_rewards[-self.check_freq:]))
            self.steps_avg.append(np.mean(self.episode_steps[-self.check_freq:]))

            if self.reward_avg[-1] > self.best_reward_avg:
                self.model.save(self.save_path)

        return super()._on_step()

    """
    def _on_rollout_end(self) -> None:
        self.episode_rewards.extend(self.locals["rewards"])
        self.episode_steps.append(self.locals["n_steps"])

        print("----------ROLLO UT END sTATS---------")
        print(self.locals,"\n")
        print(self.locals["n_steps"], "\n")
        print(self.locals["rewards"])
        print("---------------ROLL OUT END STATS-----------")

        return super()._on_rollout_end()
    """

def make_env(index: int, seed: int=0):
    def __init__():
        #env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
        env = gym.make("FrozenLake-v1", desc=grm(size=4, p=0.7), is_slippery=False, render_mode="rgb_array")
        env.reset(seed=seed + index)
        return env

    set_random_seed(seed)
    return __init__

num_envs = 6

# Create the FrozenLake environment (non-slippery)
# Create a list of envs, each with different maps to train on
#envs = [gym.make("FrozenLake-v1", desc=grm(size=4, p=0.7), is_slippery=False, render_mode="rgb_array") for _ in range(num_envs)]
envs = DummyVecEnv([make_env(i) for i in range(num_envs)])
log_dirs = ["./log/tb_logs/training/parallel/env_{}".format(i) for i in range(num_envs)]
#env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
#env = gym.make("FrozenLake-v1", desc=grm(size=4, p=0.7), is_slippery=False, render_mode="rgb_array")

# Create a vectorized envrionment, each with different maps to train on
#vec_env = VecMonitor([(lambda env: lambda: env) (env) for env in envs])
vec_env = VecMonitor(envs)

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
data_callback = DataLogger(1000)
#rewards_eval = []

# Train the agent
model.learn(total_timesteps=100000, callback=data_callback)
#mean_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)[0]
#rewards_eval.append(mean_reward)
#rewards_eval.append(evaluate_policy(model, vec_env, n_eval_episodes=10)[0])

#envs.close()

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
#model.save("./log/model/training/frozenlake_training")

# Plot the rewards throughout training
plt.plot(data_callback.reward_avg)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Agent's rewards during training")
plt.savefig("recordings/training/reward.png")

# Clear the data
plt.clf()

# Plot the Steps per episode throughout training
plt.plot(data_callback.steps_avg)
plt.xlabel("Episodes")
plt.ylabel("Steps")
plt.title("Agent's steps during training")
plt.savefig("recordings/training/steps.png")

plt.clf()

plt.plot(np.interp(np.array(data_callback.reward_avg), [0, 1], [0, 8]))
plt.plot(data_callback.steps_avg)
plt.xlabel("Episodes")
plt.ylabel("Rewards / Steps")
plt.title("Agent's rewards and steps during training")
plt.savefig("recordings/training/overlap.png")


#envs.close()
#vec_env.close()

# Reset back to the default for video export
#env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
env = gym.make("FrozenLake-v1", desc=grm(size=4, p=0.7), is_slippery=False, render_mode="rgb_array")

# Start a video to show agent after training
env = RecordVideo(env, video_folder="recordings/training/")

obs, info = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action.item())
    
    done = terminated or truncated

env.close()
