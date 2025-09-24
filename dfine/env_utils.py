import numpy as np
import gymnasium as gym
from gymnasium.utils import RecordConstructorArgs
from gymnasium.wrappers import RescaleAction


class ActionRepeatWrapper(gym.Wrapper, RecordConstructorArgs):
    """
        Execute same action several times
    """
    def __init__(
        self,
        env: gym.Env,
        repeat: int=4,
    ):
        RecordConstructorArgs.__init__(self, env=env, repeat=repeat)
        gym.Wrapper.__init__(self, env)
        self._repeat = repeat

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        return self.env.reset(seed=seed, options=options)
    
    def step(self, action):
        total_reward = 0.0
        for _ in range(self._repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        
        return obs, total_reward, terminated, truncated, info


def make_env(id: str, action_repeat: int=2):
    env = gym.make(id=id)
    env = ActionRepeatWrapper(env=env, repeat=action_repeat)
    env = RescaleAction(env=env, min_action=-1.0, max_action=1.0)
    return env