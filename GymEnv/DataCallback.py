from stable_baselines.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class DataCallback(BaseCallback):
    def __init__(self, eval_env, n_steps, file_name, plot_name=None, n_episodes=100, plot=True, verbose=0):
        super(DataCallback, self).__init__(verbose)
        self.data = []
        self.model = None
        self.plot_name = plot_name
        self.step_count = 0
        self.n_steps = n_steps
        self.eval_env = eval_env
        self.file_name = file_name
        self.n_episodes = n_episodes
        self.plot = plot

    def _on_step(self) -> bool:
        if self.step_count % self.n_steps == 0:
            rewards = []
            for i in range(self.n_episodes):
                episode_reward = 0
                obs = self.eval_env.reset()
                while True:
                    action, _state, = self.model.predict(obs)
                    obs, reward, done, _ = self.eval_env.step(action)
                    episode_reward += reward
                    if done:
                        break
                rewards.append(episode_reward)
            d = {'i': self.step_count, 'avg': np.average(rewards), 'std': np.std(rewards)}
            self.data.append(d)
            print(d)
        self.step_count += 1
        return True

    def _on_training_end(self) -> None:
        df = pd.DataFrame(self.data)
        if self.plot:
            df['avg'].plot()
            plt.show()
            if self.plot_name is not None:
                plt.savefig(self.plot_name)
        df.to_csv(self.file_name)
