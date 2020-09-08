from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines.common.callbacks import EvalCallback, CallbackList
from gym.wrappers.time_limit import TimeLimit
from DataCallback import DataCallback
from stable_baselines import PPO2
from gym.utils import seeding
from gym import spaces
from numba import njit
import numpy as np
import argparse
import optuna
import gym


@njit
def acc(r_input, v_input, thrust, angle, r_e, rm, GM, m, C_d, A, F_t):
    r2 = np.dot(r_input, r_input)
    r = np.sqrt(r2)
    v2 = np.dot(v_input, v_input)
    v = np.sqrt(v2)
    t1 = F_t * thrust
    t2 = np.array([t1 * np.cos(angle), t1 * np.sin(angle)])
    r_unit = r_input / r
    v_unit = v_input / v
    h = r - r_e
    perturbation = 0.05 * np.random.randn()
    rho = rm / ((7.8974E-24 + 8.89106E-31 * h) * (141.89 + 0.00299 * h) ** 11.388)
    drag = rho * v2 * C_d * A * v_unit
    force = - (GM * m / r2) * r_unit
    force -= drag + perturbation * drag
    force += t2
    a = force / m
    return a


@njit
def yoshida(r, v, dt, thrust, angle, r_e, rm, GM, m, C_d, A, F_t):
    c1 = 0.6756035959798288170238
    c2 = -0.1756035959798288170238
    c3 = -0.1756035959798288170238
    c4 = 0.6756035959798288170238
    d1 = 1.3512071919596576340476
    d2 = -1.7024143839193152680953
    d3 = 1.3512071919596576340476
    r1 = r + c1 * dt * v
    v1 = v + d1 * dt * acc(r1, v, thrust, angle, r_e, rm, GM, m, C_d, A, F_t)
    r2 = r1 + c2 * dt * v1
    v2 = v1 + d2 * dt * acc(r2, v1, thrust, angle, r_e, rm, GM, m, C_d, A, F_t)
    r3 = r2 + c3 * dt * v2
    v3 = v2 + d3 * dt * acc(r3, v2, thrust, angle, r_e, rm, GM, m, C_d, A, F_t)
    r = r3 + c4 * v3 * dt
    v = v3
    return r, v


@njit
def calculate_physics(r, v, thrust, angle, F_t, r_e, GM, m, C_d, A, steps, dt, rm):
    ndt = dt / steps
    for _ in range(steps):
        r, v = yoshida(r, v, ndt, thrust, angle, r_e, rm, GM, m, C_d, A, F_t)
    return r, v


class OrbitDecayEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self):
        self.h = 5.5E5                # Height of satellite 550 km in meters
        self.r_e = 6.371E6            # Radius of earth in meters
        self.r_s = self.h + self.r_e  # Radius from center of earth
        self.m = 100.0                # Mass of satellite
        self.mp = 75.0                # Mass of propellant
        self.dt = 1.0                 # Delta t
        self.GM = 3.986004418E14      # Earth's gravitational parameter
        self.C_d = 2.123              # Drag coefficient
        self.A = 1.0                  # Surface area normal to velocity
        self.F_t = 0.04               # Force of thrust
        self.steps = 1                # Step per dt
        self.threshold = 1            # Threshold to end episode
        self.rho_multiplier = 10000   # Rho is multiplied by this amount
        self.step_fuel = 125          # Number of steps for fuel to run out, when at full thrust
        self.orbit_v = np.sqrt(self.GM / self.r_s)
        # Some state vectors
        self.r = np.zeros(2)
        self.v = np.zeros(2)
        self.theta = 0
        self.thrust = 0.0
        high = np.array([1, 1, 1, 1, self.threshold, 1, np.pi, 1], dtype=np.float32)
        low = -high
        low[4] = 0
        low[5] = 0
        low[7] = 0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1.0, shape=(2,), dtype=np.float32)
        self.viewer = None
        self.current_step = 0
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.step_fuel_used = 0.0
        self.r = np.array([
            self.r_s,
            0
        ])
        self.v = np.array([
            0,
            self.orbit_v
        ])
        self.theta = 0
        self.thrust = 0.0
        self.current_step = 0
        return np.concatenate((self.r / self.r_s, self.v / self.orbit_v, np.zeros(2), [self.theta, self.thrust]))

    def step(self, action):
        assert self.action_space.contains(action)
        self.thrust += action[0] * 0.04 - 0.02
        self.thrust = np.clip(self.thrust, 0.0, 1.0)
        self.theta += action[1] * np.pi / 3 - np.pi / 6
        self.theta -= (2 * np.pi) * np.floor((self.theta + np.pi) * (1 / (2 * np.pi)))
        self.r, self.v = calculate_physics(self.r, self.v, self.thrust, self.theta, self.F_t, self.r_e, self.GM, self.m, self.C_d,
                                           self.A, self.steps, self.dt, self.rho_multiplier)
        # Distance from target
        hd = np.abs(np.linalg.norm(self.r) - self.r_s)
        # Vector distance from target velocity
        vel = np.abs(np.linalg.norm(self.v) - self.orbit_v)
        # Create output state and scale to between -1 and 1
        state = np.concatenate((self.r / self.r_s, self.v / self.orbit_v, [hd, vel, self.theta, self.thrust]))
        # Calculate fuel used
        self.step_fuel_used += self.thrust

        reward = self.current_step / 800.0 + 0.5

        done = False
        if hd > self.threshold:
            reward = 0.0
            done = True
        if self.step_fuel_used > self.step_fuel:
            reward = 0.0
            done = True

        self.current_step += 1
        return state, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass


def make_env():
    return TimeLimit(OrbitDecayEnv(), max_episode_steps=1000)


def make_venv(rank, seed=0):
    def _init():
        env = make_env()
        env.seed(seed + rank)
        return env
    return _init


def main():
    env = SubprocVecEnv([make_venv(i) for i in range(16)])
    env_eval1 = make_env()
    env_eval2 = make_env()
    env.seed(100)
    set_global_seeds(100)
    data_path = './training_result/'
    ppo_callback = DataCallback(env_eval1, 10000, data_path + 'data.csv')
    eval_callback = EvalCallback(env_eval2, best_model_save_path='./logs',
                                 log_path='./logs/', eval_freq=10000, n_eval_episodes=100,
                                 deterministic=True, render=False)
    callback = CallbackList([ppo_callback, eval_callback])

    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=data_path, ent_coef=0.0,
                 n_steps=256, nminibatches=8, lam=0.94, gamma=0.99, noptepochs=4)
    model.learn(total_timesteps=10000000, callback=callback, tb_log_name='PPO2')


def test(n):
    env = make_env()
    obs = env.reset()
    for i in range(n):
        print(f'i: {i} obs {obs}')
        obs, rewards, dones, info = env.step([0.0, 0.0])


def objective(trial: optuna.Trial):
    n_steps = 256
    nminibatches = 8
    noptepochs = 4
    lam = trial.suggest_uniform('lam', 0.8, 0.95)
    gamma = trial.suggest_uniform('gamma', 0.98, 0.999)
    learning_rate = trial.suggest_loguniform('learning_rate', 3e-4, 0.001)
    ent_coef = trial.suggest_uniform('ent_coef', 0.0,  0.01)
    cliprange = trial.suggest_uniform('cliprange', 0.01, 0.2)
    total_timesteps = 10000000

    set_global_seeds(100)
    env = SubprocVecEnv([make_venv(i) for i in range(16)])
    env_eval = make_env()

    model = PPO2(MlpPolicy, env, verbose=1, n_steps=n_steps, nminibatches=nminibatches, noptepochs=noptepochs,
                 lam=lam, gamma=gamma, learning_rate=learning_rate, ent_coef=ent_coef, cliprange=cliprange)
    model.learn(total_timesteps)

    return evaluate_policy(model, env_eval, 100)[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default='0', help='If not set to 0, runs a test for n steps')
    parser.add_argument('--optuna', action='store_true', help='Activate Optuna')
    args = parser.parse_args()

    if args.optuna:
        print('Optimizing PPO')
        study = optuna.load_study(study_name='PPO_study2', storage='mysql://anhydrous99:eeQstj5b9K@161.35.126.61/anhydrous99')
        study.optimize(objective, n_trials=100)
        exit(0)

    n = int(args.test)
    if n == 0:
        main()
    else:
        test(n)
