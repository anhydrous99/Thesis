from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from gym.wrappers.time_limit import TimeLimit
from stable_baselines import PPO2
from gym.utils import seeding
from gym import spaces
from numba import njit
import numpy as np
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
    rho = rm / ((7.8974E-24 + 8.89106E-31 * h) * (141.89 + 0.00299 * h) ** 11.388)
    force = - (GM * m / r2) * r_unit
    t = rho * v2 * C_d * A * v_unit
    force -= rho * v2 * C_d * A * v_unit
    force += t2
    a = force / m
    return a


@njit
def calculate_physics(x, v, thrust, angle, F_t, r_e, GM, m, C_d, A, steps, dt, rm):
    ndt = dt / steps
    for _ in range(steps):
        a = acc(x, v, thrust, angle, r_e, rm, GM, m, C_d, A, F_t)
        v = v + a * ndt / 2
        x = x + v * ndt
        v = v + a * ndt / 2
    return x, v


class OrbitDecayEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self):
        self.h = 5.95E5             # Height of satellite 550 km in meters
        self.r_e = 6.371E6          # Radius of earth in meters
        self.r_s = self.h + self.r_e
        self.m = 1000.0             # Mass of satellite
        self.dt = 1.0               # Delta t
        self.GM = 3.986004418E14    # Earth's gravitational parameter
        self.C_d = 2.123            # Drag coefficient
        self.A = 1.0                # Surface area normal to velocity
        self.F_t = 0.1              # Force of thrust
        self.steps = 5000           # Step per dt
        self.threshold = 3          # Threshold to end episode
        self.rho_multiplier = 10000 # Rho is multiplied by this amount
        self.orbit_v = None
        # Some state vectors
        self.r = np.zeros(2)
        self.v = np.zeros(2)
        high = np.array([1, 1, 1, 1, self.threshold], dtype=np.float32)
        low = -high
        low[4] = 0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1.0, shape=(2,), dtype=np.float32)
        self.viewer = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        theta = self.np_random.uniform(low=0, high=2 * np.pi)
        self.r = np.array([
            self.r_s,
            0
        ])
        self.orbit_v = np.sqrt(self.GM / self.r_s)
        self.v = np.array([
            0,
            self.orbit_v
        ])
        return np.concatenate((self.r, self.v, np.zeros(1)))

    def step(self, action):
        assert self.action_space.contains(action)
        thrust = action[0]
        angle = action[1] * 2 * np.pi
        self.r, self.v = calculate_physics(self.r, self.v, thrust, angle, self.F_t, self.r_e, self.GM, self.m, self.C_d,
                                           self.A, self.steps, self.dt, self.rho_multiplier)
        h = np.linalg.norm(self.r) - self.r_e
        hd = np.abs(self.h - h)  # Distance from intended height
        # Create output state and scale to between -1 and 1
        state = np.concatenate((self.r / self.r_s, self.v / self.orbit_v, [hd]))

        reward = 1.0
        done = False
        if hd > self.threshold:
            reward = 0.0
            done = True
        elif hd > self.threshold / 2:
            reward = 0.5

        return state, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass


def make_env():
    return TimeLimit(OrbitDecayEnv(), max_episode_steps=3000)


def make_venv(rank, seed=0):
    def _init():
        env = make_env()
        env.seed(seed + rank)
        return env
    return _init


def main():
    env = SubprocVecEnv([make_venv(i) for i in range(16)])
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='./training_result/')
    model.learn(total_timesteps=100000000)


def main2():
    env = make_env()
    obs = env.reset()
    for i in range(3000):
        print(f'i: {i} obs {obs}')
        obs, rewards, dones, info = env.step([0.0, 0.0])


if __name__ == '__main__':
    main()

