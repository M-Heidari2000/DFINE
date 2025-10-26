import minari
import numpy as np
from tqdm import tqdm
from io import BytesIO
from PIL import Image
from einops import rearrange


class ReplayBuffer:
    """
        Replay buffer holds sample trajectories
    """

    @staticmethod
    def load_from_minari(dataset: minari.MinariDataset):
        buffer = ReplayBuffer(
            capacity=dataset.total_steps,
            u_dim=dataset.action_space.shape[0],
        )
        print("loading the dataset")
        for episode in tqdm(dataset):
            steps = episode.actions.shape[0]
            for i in range(steps):
                y = episode.infos["pixels"][i].tobytes()
                y = Image.open(BytesIO(y)).convert("RGB")
                y = rearrange(np.array(y), "h w c -> c h w")
                buffer.push(
                    y=y,
                    u=episode.actions[i],
                    c=-episode.rewards[i],
                    done=episode.terminations[i] or episode.truncations[i],
                )
        return buffer

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