import matplotlib.pyplot as plt
from OrbitDecayEnv import make_env
from stable_baselines import PPO2
import seaborn as sns
import numpy as np
import argparse


def target_distance(obs):
    return np.linalg.norm(obs[5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='logs/best_model.zip')
    args = parser.parse_args()
    model_path = args.model

    # Load model
    model = PPO2.load(model_path)
    # Create environment
    env = make_env()

    episode_reward_list = []
    for _ in range(100):
        obs = env.reset()
        episode_reward = 0
        while True:
            action, _state = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break
        episode_reward_list.append(episode_reward)
    sns.histplot(x=episode_reward_list)
    plt.savefig('../Plots/ModelRewardDistribution.png')
    plt.show()


if __name__ == '__main__':
    main()
