from OrbitDecayEnv import make_env
from stable_baselines import PPO2
import matplotlib.pyplot as plt
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

    obs = env.reset()
    distances = [target_distance(obs)]
    thrust = []
    episode_reward = 0
    while True:
        action, _state = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        distances.append(target_distance(obs))
        thrust.append(obs[-1])
        episode_reward += reward
        if done:
            break

    plt.plot(distances)
    plt.plot(thrust)
    plt.show()


if __name__ == '__main__':
    main()
