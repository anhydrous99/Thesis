import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
grid = plt.GridSpec(2, 3, wspace=0.5, hspace=0.6)


def plot(path, i, j):
    df = pd.read_csv(path)
    df.rename(columns={'i': 'step', 'avg': 'reward'}, inplace=True)
    plt.subplot(grid[i, j])
    ax = sns.lineplot(x='step', y='reward', data=df)
    ax.fill_between(df['step'], y1=df['reward'] - df['std'], y2=df['reward'] + df['std'], alpha=.5)


plot('../Data/data.csv', 0, 0)
plot('../Data/data_1.csv', 0, 1)
plot('../Data/data_2.csv', 0, 2)
plot('../Data/data_3.csv', 1, 0)
plot('../Data/data_4.csv', 1, 1)
plot('../Data/data_5.csv', 1, 2)
plt.savefig('../Plots/TestPlot.png')
plt.show()