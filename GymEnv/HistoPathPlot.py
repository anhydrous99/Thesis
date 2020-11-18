import matplotlib.pyplot as plt
from OrbitDecayEnv import make_env
from stable_baselines import PPO2
import seaborn as sns
import scipy.stats as st
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import warnings


def target_distance(obs):
    return np.linalg.norm(obs[5])


def best_fit_distribution(data, bins=200, ax=None):
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [
        st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi, st.chi2,
        st.cosine,
        st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f, st.fatiguelife,
        st.fisk,
        st.foldcauchy, st.foldnorm, st.frechet_r, st.frechet_l, st.genlogistic, st.genpareto, st.gennorm, st.genexpon,
        st.genextreme, st.gausshyper, st.gamma, st.gengamma, st.genhalflogistic, st.gilbrat, st.gompertz, st.gumbel_r,
        st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm, st.hypsecant, st.invgamma,
        st.invgauss,
        st.invweibull, st.johnsonsb, st.johnsonsu, st.ksone, st.kstwobign, st.laplace, st.levy, st.levy_l,
        st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke, st.nakagami, st.ncx2,
        st.ncf,
        st.nct, st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm, st.powernorm, st.rdist, st.reciprocal,
        st.rayleigh, st.rice, st.recipinvgauss, st.semicircular, st.t, st.triang, st.truncexpon, st.truncnorm,
        st.tukeylambda,
        st.uniform, st.vonmises, st.vonmises_line, st.wald, st.weibull_min, st.weibull_max, st.wrapcauchy
    ]
    print(len(DISTRIBUTIONS))

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for dist in tqdm(DISTRIBUTIONS):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = dist.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = dist.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                print(f'Dist: {dist.name} SSE: {sse}')

                # if axis pass in add to plot
                try:
                    if ax:
                        pdf.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = dist
                    best_params = params
                    best_sse = sse
        except Exception:
            pass
    return (best_distribution.name, best_params)


def make_pdf(dist, params, size):
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.95, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.95, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)
    return pdf


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
    size = 10
    for _ in tqdm(range(size)):
        obs = env.reset()
        episode_reward = 0
        while True:
            action, _state = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break
        episode_reward_list.append(episode_reward)
    data = np.array(episode_reward_list)
    best_fit_name, best_fit_params = best_fit_distribution(data)
    best_dist = getattr(st, best_fit_name)
    pdf = make_pdf(best_dist, best_fit_params, size)
    ax = pdf.plot()
    print(best_fit_name)
    print(best_fit_params)

    sns.histplot(data, ax=ax, stat='density')
    plt.xlabel('Reward')
    plt.title('Reward vs. Probability Density')

    plt.savefig('../Plots/ModelRewardDistribution.png')
    plt.show()


if __name__ == '__main__':
    main()
