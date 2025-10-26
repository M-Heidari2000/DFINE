import numpy as np
from einops import rearrange


class ReplayBuffer:
    """
        Replay buffer holds sample trajectories
    """

    @staticmethod
    def preprocess_obs(obs, bit_depth=5):
        """
        reduces the bit depth of image for the ease of training and converts to [-0.5, 0.5]
        In addition, add uniform random noise same as original implementation
        """
        obs = obs.astype(np.float32)
        reduced_obs = np.floor(obs / 2 ** (8 - bit_depth))
        normalized_obs = reduced_obs / 2**bit_depth - 0.5
        normalized_obs += np.random.uniform(0.0, 1.0 / 2**bit_depth, normalized_obs.shape)
        # convert HWC -> CHW
        normalized_obs = rearrange(normalized_obs, "b h w c -> b c h w")
        return normalized_obs
    
    @staticmethod
    def postprocess_obs(proc_obs, bit_depth=5):
        """
        Approximate inverse of preprocess_obs:
        maps normalized float observations in [-0.5, 0.5] back to uint8 images in [0, 255].
        """
        restored = (proc_obs + 0.5) * (2 ** bit_depth)
        restored = np.clip(restored, 0, 2 ** bit_depth - 1)
        restored = restored * (2 ** (8 - bit_depth))
        restored = np.round(restored).astype(np.uint8)
        # convert CHW -> HWC
        restored = rearrange(restored, "b c h w -> b h w c")
        return restored

    def __init__(
        self,
        capacity: int,
        u_dim: int,
    ):
        self.capacity = capacity

        self.u_dim = u_dim

        self.ys = np.zeros((capacity, 3, 64, 64), dtype=np.uint8)
        self.us = np.zeros((capacity, u_dim), dtype=np.float32)
        self.cs = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=bool)

        self.index = 0
        self.is_filled = False

    def __len__(self):
        return self.capacity if self.is_filled else self.index

    def push(
        self,
        y,
        u,
        c,
        done,
    ):
        """
            Add experience (single step) to the replay buffer
        """
        self.ys[self.index] = y
        self.us[self.index] = u
        self.cs[self.index] = c
        self.done[self.index] = done

        self.index = (self.index + 1) % self.capacity
        self.is_filled = self.is_filled or self.index == 0

    def sample(
        self,
        batch_size: int,
        chunk_length: int
    ):
        done = self.done.copy()
        done[-1] = 1
        episode_ends = np.where(done)[0]

        all_indexes = np.arange(len(self))
        distances = episode_ends[np.searchsorted(episode_ends, all_indexes)] - all_indexes + 1
        valid_indexes = all_indexes[distances >= chunk_length]

        sampled_indexes = np.random.choice(valid_indexes, size=batch_size)
        sampled_ranges = np.vstack([
            np.arange(start, start + chunk_length) for start in sampled_indexes
        ])

        sampled_ys = self.ys[sampled_ranges].reshape(
            batch_size, chunk_length, 3, 64, 64
        )
        sampled_us = self.us[sampled_ranges].reshape(
            batch_size, chunk_length, self.us.shape[1]
        )
        sampled_cs = self.cs[sampled_ranges].reshape(
            batch_size, chunk_length, 1
        )
        sampled_done = self.done[sampled_ranges].reshape(
            batch_size, chunk_length, 1
        )

        return sampled_ys, sampled_us, sampled_cs, sampled_done