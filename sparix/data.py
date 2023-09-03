import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset


class FrameDataset(Dataset):
    def __init__(self, frame_source, frame_idx_list, num_frame_in_context, Hp, Wp, sample_size, pad = None):
        """
        frame_source points to data with the shape (B, 1, H, W)
        """
        self.frame_source         = frame_source
        self.frame_idx_list       = frame_idx_list
        self.num_frame_in_context = num_frame_in_context
        self.Hp, self.Wp          = Hp, Wp
        self.sample_size          = sample_size
        self.pad                  = pad

        self.idx_list = []
        self.update_random_dataset()


    def update_random_dataset(self):
        frame_source         = self.frame_source
        frame_idx_list       = self.frame_idx_list
        num_frame_in_context = self.num_frame_in_context
        sample_size          = self.sample_size

        # Create the dataset...
        # The rightmost context window is used as a target
        self.idx_list = random.choices(list(range(len(frame_idx_list) - num_frame_in_context)), k = sample_size)


    def __len__(self):
        return self.sample_size


    def __getitem__(self, idx):
        frame_source         = self.frame_source
        num_frame_in_context = self.num_frame_in_context
        Hp, Wp               = self.Hp, self.Wp
        pad                  = self.pad

        sample_idx = self.idx_list[idx]
        sample_frame_idx = self.frame_idx_list[sample_idx]

        # Get the context...
        context_min = sample_frame_idx
        context_max = sample_frame_idx + num_frame_in_context
        context = frame_source[context_min:context_max]    # (B, 1, H, W)

        # Flatten the context...
        B, C, H, W = context.shape
        context = pad(context.reshape(B * C, H, W))[:, None]
        B, C, H, W = context.shape
        num_patch = B * C * H * W // Hp // Wp
        context = context.reshape(B, C, H//Hp, Hp, W//Wp, Wp).swapaxes(3,4)
        context = context.reshape(B * C * H * W // Hp // Wp, Hp, Wp)

        # Get the target (context offset by 1)...
        target_min = sample_frame_idx
        target_max = sample_frame_idx + num_frame_in_context + 1
        target = frame_source[target_min:target_max]
        B, C, H, W = target.shape
        target = pad(target.reshape(B * C, H, W))[:, None]
        B, C, H, W = target.shape
        target = target.reshape(B, C, H//Hp, Hp, W//Wp, Wp).swapaxes(3,4)
        target = target.reshape(B * C * H * W // Hp // Wp, Hp, Wp)
        target = target[1:1 + num_patch]

        return context, target
