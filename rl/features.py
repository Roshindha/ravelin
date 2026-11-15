"""Custom SB3 features extractor: small CNN on the (C,H,W) grid + MLP on the state vector, concatenated.

Shape-generic (read from the observation space): prototype (8,16,16) + 9, v2 BreachEnv (9,14,14) + 11 -> features_dim 320."""
from __future__ import annotations

import gymnasium as gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class SiegeFeaturesExtractor(BaseFeaturesExtractor):
    """grid (B,8,16,16) -> conv stack -> 256-d;  state (B,9) -> MLP -> 64-d;  concat -> features_dim=320.

    Spatial resolution is kept for two conv layers (16x16, then 8x8) so border-tile / building positions
    stay distinguishable, then flattened (not global-pooled) because *where* things are matters for deploy.
    """

    GRID_OUT = 256
    STATE_OUT = 64

    def __init__(self, observation_space: gym.spaces.Dict) -> None:
        super().__init__(observation_space, features_dim=self.GRID_OUT + self.STATE_OUT)
        c, h, w = observation_space["grid"].shape
        (state_dim,) = observation_space["state"].shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 8x8
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),  # 4x4
            nn.Flatten(),
        )
        with th.no_grad():
            n_flat = self.cnn(th.zeros(1, c, h, w)).shape[1]
        self.grid_fc = nn.Sequential(nn.Linear(n_flat, self.GRID_OUT), nn.ReLU())
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, self.STATE_OUT), nn.ReLU(), nn.Linear(self.STATE_OUT, self.STATE_OUT), nn.ReLU())

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        g = self.grid_fc(self.cnn(observations["grid"]))
        s = self.state_mlp(observations["state"])
        return th.cat([g, s], dim=1)
