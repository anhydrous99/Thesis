from stable_baselines.common.policies import MlpPolicy
from gym.wrappers.time_limit import TimeLimit
from stable_baselines import PPO2
from gym.utils import seeding
from gym import spaces
import numpy as np
import gym


class OrbitDecayEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self):
        self.h = 5.5E5              # Height of satellite 300 km in meters
        self.r_e = 6.371E6        # Radius of earth in meters
        self.r_s = self.h + self.r_e
        self.m = 100              # Mass of satellite
        self.dt = 1               # Delta t
        self.GM = 3.986004418E14  # Earth's gravitational parameter
        self.C_d = 2.123          # Drag coefficient
        self.A = 1                # Surface area normal to velocity
        self.F_t = 0.2            # Force of thrust
        self.steps = 1000
        self.orbit_v = None
        # Some state vectors
        self.r = np.zeros(2)
        self.v = np.zeros(2)
        high = np.array([1, 1, 1, 1, 10], dtype=np.float32)
        low = -high
        low[4] = 0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
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
        dt = self.dt / self.steps
        thrust = action[0]
        angle = action[1] * np.pi
        for _ in range(self.steps):
            a = self._acceleration(self.r, self.v, thrust, angle)
            self.r += self.v * dt + 0.5 * a * dt ** 2
            self.v += a * dt
        h = np.linalg.norm(self.r) - self.r_e
        hd = np.abs(self.h - h)  # Distance from intended height
        # Create output state and scale to between -1 and 1
        state = np.concatenate((self.r / self.r_s, self.v / self.orbit_v, [hd]))

        reward = 1.0
        done = False
        if hd > 10:
            reward = 0.0
            done = True

        return state, reward, done, {}

    def _acceleration(self, r_input, v_input, thrust, angle):
        r2 = np.dot(r_input, r_input)
        r = np.sqrt(r2)
        v2 = np.dot(v_input, v_input)
        v = np.sqrt(v2)
        thrust = self.F_t = thrust
        thrust = np.array([thrust * np.cos(angle), thrust * np.sin(angle)])
        r_unit = r_input / r
        v_unit = v_input / v
        h = r - self.r_e
        rho = 1000 / ((7.8974E-24 + 8.89106E-31 * h) * (141.89 + 0.00299 * h) ** 11.388)
        force = - (self.GM * self.m / r2) * r_unit
        force -= rho * v2 * self.C_d * self.A * v_unit
        force += thrust
        a = force / self.m
        return a

    def render(self, mode='human'):
        pass

    def close(self):
        pass


def make_env():
    return TimeLimit(OrbitDecayEnv(), max_episode_steps=3000)


env = make_env()
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='./training_result/')
model.learn(total_timesteps=10000000)

