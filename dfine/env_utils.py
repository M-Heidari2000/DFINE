import gymnasium as gym
from dm_control import suite
from gymnasium.wrappers import RescaleAction
from dm_control.suite.wrappers import pixels
import numpy as np


class GymEnv(gym.Env):
    """
    Gymnasium interface wrapper for dm_control env wrapped by pixels.Wrapper.
    Converts observations to (C, H, W).
    """
    metadata = {'render.modes': ['rgb_array']}
    reward_range = (-np.inf, np.inf)

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        obs_spec = self._env.observation_spec()
        h, w, c = obs_spec['pixels'].shape
        # transpose shape for (C, H, W)
        return gym.spaces.Box(0, 255, (c, h, w), dtype=np.uint8)

    @property
    def action_space(self):
        action_spec = self._env.action_spec()
        return gym.spaces.Box(action_spec.minimum, action_spec.maximum, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None and hasattr(self._env, "task") and hasattr(self._env.task, "_random"):
            self._env.task._random = np.random.RandomState(seed)

        time_step = self._env.reset()
        obs = time_step.observation['pixels']
        obs = np.transpose(obs, (2, 0, 1))  # (C, H, W)
        info = {}
        return obs, info

    def step(self, action):
        time_step = self._env.step(action)
        obs = time_step.observation['pixels']
        obs = np.transpose(obs, (2, 0, 1))  # (C, H, W)
        reward = float(time_step.reward) if time_step.reward is not None else 0.0
        terminated = bool(time_step.last())
        truncated = False
        info = {'discount': time_step.discount}
        return obs, reward, terminated, truncated, info

    def render(self, mode='rgb_array', **kwargs):
        if mode != 'rgb_array':
            raise NotImplementedError("Only rgb_array mode supported")
        if not kwargs:
            kwargs = getattr(self._env, "_render_kwargs", {"height": 64, "width": 64, "camera_id": 0})
        img = self._env.physics.render(**kwargs)
        # transpose render output as well
        return np.transpose(img, (2, 0, 1))


class RepeatAction(gym.Wrapper):
    """Action repeat wrapper (Gymnasium API)."""
    def __init__(self, env, skip=4):
        super().__init__(env=env)
        self._skip = skip

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None
        for _ in range(self._skip):
            obs, r, term, trunc, info = self.env.step(action)
            total_reward += r
            terminated = terminated or term
            truncated = truncated or trunc
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def make_env(domain_name: str, task_name: str, action_repeat: int = 2):
    env = suite.load(domain_name=domain_name, task_name=task_name)
    env = pixels.Wrapper(env, render_kwargs={"height": 64, "width": 64, "camera_id": 0})
    env = GymEnv(env=env)
    env = RepeatAction(env=env, skip=action_repeat)
    env = RescaleAction(env=env, min_action=-1.0, max_action=1.0)
    return env