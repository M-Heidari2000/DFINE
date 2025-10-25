import numpy as np
import gymnasium as gym
from dm_control import suite
from gymnasium.wrappers import RescaleAction
from dm_control.suite.wrappers import pixels


class GymEnv(gym.Env):
    """
    dm_control -> Gymnasium wrapper.

    Returns:
        obs (np.float32): 1D state vector (concat of all non-'pixels' observations)
        info['pixels'] (np.uint8): HxWxC RGB image (for Minari; JPEG-safe)
        info['discount'] (np.float32): dm_control discount (1.0 mid-episode, 0.0 at terminal)
    """
    metadata = {'render.modes': ['rgb_array']}
    reward_range = (-np.inf, np.inf)

    def __init__(self, env):
        super().__init__()
        self._env = env

        # ----- infer state vector dimension (exclude 'pixels') -----
        obs_spec = self._env.observation_spec()
        flat_size = 0
        for key, spec in obs_spec.items():
            if key == 'pixels':
                continue
            shape = getattr(spec, "shape", ())
            flat_size += int(np.prod(shape) if len(shape) else 1)

        if flat_size == 0:
            raise RuntimeError(
                "No non-pixel observations found to form the state vector. "
                "Ensure pixels.Wrapper was created with pixels_only=False."
            )
        self._state_dim = flat_size

        # ----- spaces -----
        action_spec = self._env.action_spec()
        self.action_space = gym.spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            shape=action_spec.shape,
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._state_dim,), dtype=np.float32
        )

    def __getattr__(self, name):
        # delegate to underlying dm_control env
        return getattr(self._env, name)

    # ---------------- Helpers ----------------

    @staticmethod
    def _flatten_state(obs_dict):
        """Concatenate all non-'pixels' items into a single 1D float32 vector."""
        parts = []
        for k, v in obs_dict.items():
            if k == 'pixels':
                continue
            arr = np.asarray(v, dtype=np.float32)
            parts.append(arr.ravel())
        return (np.concatenate(parts, axis=0).astype(np.float32)
                if parts else np.zeros((0,), np.float32))

    @staticmethod
    def _safe_discount(d):
        # dm_control sometimes returns None -> use 1.0 mid-episode
        return np.float32(1.0 if d is None else float(d))

    # ---------------- Core API ----------------

    def reset(self, *, seed=None, options=None):
        # Make seeding robust to Minari passing large/np.int64 seeds
        if seed is not None and hasattr(self._env, "task") and hasattr(self._env.task, "_random"):
            try:
                safe_seed = int(seed) % (2**32 - 1)
                self._env.task._random = np.random.RandomState(safe_seed)
            except Exception:
                self._env.task._random = np.random.RandomState()

        ts = self._env.reset()
        state_vec = self._flatten_state(ts.observation)

        # Keep pixels as HWC uint8 for Minari (JPEG encoder expects this)
        pixels_hwc = ts.observation['pixels'].astype(np.uint8).copy()

        info = {
            'pixels': pixels_hwc,                           # H, W, C (uint8)
            'discount': self._safe_discount(ts.discount),   # float32
        }
        return state_vec, info

    def step(self, action):
        ts = self._env.step(np.asarray(action, dtype=np.float32))
        state_vec = self._flatten_state(ts.observation)
        pixels_hwc = ts.observation['pixels'].astype(np.uint8).copy()

        reward = float(ts.reward) if ts.reward is not None else 0.0
        terminated = bool(ts.last())
        truncated = False
        info = {
            'pixels': pixels_hwc,                           # H, W, C (uint8)
            'discount': self._safe_discount(ts.discount),   # float32
        }
        return state_vec, reward, terminated, truncated, info

    def render(self, mode='rgb_array', **kwargs):
        if mode != 'rgb_array':
            raise NotImplementedError("Only rgb_array mode supported")
        rk = getattr(self._env, "_render_kwargs", {"height": 64, "width": 64, "camera_id": 0})
        rk.update(kwargs or {})
        img = self._env.physics.render(**rk)  # H, W, 3 uint8
        return img  # HWC (keep as HWC for preview or convert as needed)


class RepeatAction(gym.Wrapper):
    """Action repeat wrapper (Gymnasium API). Accumulates reward, returns last obs/info."""
    def __init__(self, env, skip=4):
        super().__init__(env=env)
        self._skip = int(skip)

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
            terminated |= term
            truncated |= trunc
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def make_env(domain_name: str, task_name: str, action_repeat: int = 2,
             height: int = 64, width: int = 64, camera_id: int = 0):
    """
    Build dm_control suite environment compatible with Gymnasium.

    Observations: state vector (float32)
    info['pixels']: HWC uint8 image (Minari-friendly)
    """
    env = suite.load(domain_name=domain_name, task_name=task_name)

    # Keep low-dim observations AND add pixels
    env = pixels.Wrapper(
        env,
        render_kwargs={"height": height, "width": width, "camera_id": camera_id},
        pixels_only=False,            # keep non-pixel observations
        observation_key="pixels",
    )

    env = GymEnv(env=env)
    env = RepeatAction(env=env, skip=action_repeat)
    env = RescaleAction(env=env, min_action=-1.0, max_action=1.0)
    return env
