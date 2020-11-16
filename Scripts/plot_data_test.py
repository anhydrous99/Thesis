import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
grid = plt.GridSpec(2, 3, wspace=0.5, hspace=0.6)


def plot(path, path2, i, j, lbl='a'):
    df = pd.read_csv(path)
    df2 = pd.read_csv(path2)
    df2['Step'] = df2['Step'] * df['i'].iloc[-1] / df2['Step'].iloc[-1]
    df.rename(columns={'i': 'step', 'avg': 'reward'}, inplace=True)
    ax = plt.subplot(grid[i, j])
    sns.lineplot(x='step', y='reward', data=df)
    ax.fill_between(df['step'], y1=df['reward'] - df['std'], y2=df['reward'] + df['std'], alpha=.5)

    # Smoothing
    new_x = np.linspace(df2['Step'].min(), df2['Step'].max(), len(df['step'])//2)
    x_tmp = np.linspace(df2['Step'].min(), df2['Step'].max(), len(df2['Step']))
    interp_f = scipy.interpolate.interp1d(x_tmp, df2['Value'], kind='slinear')
    new_y = interp_f(new_x)
    sns.lineplot(x=new_x, y=new_y, alpha=0.75)
    plt.title(lbl)
    ax.set_xticklabels([])


plot('../Data/data.csv', '../Data/run-PPO2_1-tag-episode_reward.csv', 0, 0, 'a)')
plot('../Data/data_1.csv', '../Data/run-PPO2_2-tag-episode_reward.csv', 0, 1, 'b)')
plot('../Data/data_2.csv', '../Data/run-PPO2_3-tag-episode_reward.csv', 0, 2, 'c)')
plot('../Data/data_3.csv', '../Data/run-PPO2_4-tag-episode_reward.csv', 1, 0, 'd)')
plot('../Data/data_4.csv', '../Data/run-PPO2_5-tag-episode_reward.csv', 1, 1, 'e)')
plot('../Data/data_5.csv', '../Data/run-PPO2_6-tag-episode_reward.csv', 1, 2, 'f)')
plt.figlegend(('Evaluation', 'Training'), loc='upper left')
plt.savefig('../Plots/TestPlot.png', bbox_inches='tight')
plt.show()