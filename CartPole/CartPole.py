"""
    The basics of Stable_Baseline3 and Gymnasium.

    Code was inspired and a modified version of:
    https://stable-baselines3.readthedocs.io/en/master/guide/examples.html

    By: Jhon Tabio
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from matplotlib.animation import FuncAnimation # TENSORBOARD


# Create a custom callback to log learning rate and episode rewards
class PlotCallback(BaseCallback):
    def __init__(self, figure, axs, verbose=0):
        super(PlotCallback, self).__init__(verbose)
        self.figure = figure
        self.axs = axs
        self.learning_rates = []
        self.episode_rewards = []
    
    # Log learning rate
    def _on_step(self):
        current_lr = self.model.policy.optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)

        return True
    
    # Log episode rewards
    def _on_episode_end(self):
        self.episode_rewards.append(self.episode_reward)

        update_data(self.axs, self)

def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, render_mode="human")
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

# Function to update the plot in real-time
def update_data(axs, pcb):
    # Perform a step in the environment
    # obs, _, _, _ = env.step([model.predict(obs)[0] for obs in obs])  # Agent's actions

    # Clear the previous plots and update the new data
    axs[0].clear()
    axs[1].clear()

    axs[0].plot(pcb.learning_rates)
    axs[0].set_xlabel('Training Steps')
    axs[0].set_ylabel('Learning Rate')

    axs[1].plot(pcb.episode_rewards)
    axs[1].set_xlabel('Training Episodes')
    axs[1].set_ylabel('Episode Reward')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    env_id = "CartPole-v1"
    resetModel = 'y'

    if os.path.exists("CartPole_Data.zip"):
        while True:
            resetModel = input("Would you like to reset the model (y/n): ")[0]

            if resetModel == 'y' or resetModel == 'n':
                break
    
    num_cpu = int(input("How many virtual environments would you like: "))  # Number of processes to use

    # Create the vectorized environment
    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    
    model = None

    if resetModel == 'y':
        model = PPO("MlpPolicy", vec_env, verbose=1)

        figure, axs = plt.subplots(2, 1, figsize=(8, 6))

        pcb = PlotCallback(figure, axs)

        obs = vec_env.reset()

        # Create an animation that updates the plot every 100 milliseconds
        # graph = FuncAnimation(figure, update_data, fargs=(vec_env, axs, pcb, obs), interval=100)

        model.learn(total_timesteps=5_000, callback=pcb)

        """
        plt.tight_layout()
        plt.show()
        """

        print("\nTRAINING COMPLETE!")

        # Close the plot when training is done
        plt.ioff()
        plt.show()

        model.save("CartPole_Data")
    
    elif resetModel == 'n':
        model = PPO.load("CartPole_Data")
        print(model.policy.optimizer.param_groups[0])
    
    #mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    #print("Mean Reward: ", mean_reward, "\n", "Standard Deveation Reward: ", std_reward)

    obs = vec_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render()
