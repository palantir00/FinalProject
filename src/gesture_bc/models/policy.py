"""A trained network wrapped as a Policy (reset / act), plus checkpoint files."""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from gesture_bc.data.normalization import Normalizer
from gesture_bc.models.mlp import MLP, MOVE_DIM
from gesture_bc.sim.pick_place_env import flatten_observation
from gesture_bc.utils.device import resolve_device


class NeuralPolicy:
    """Runs the network on one observation at a time, like the expert did.

    observation dict -> flat vector -> normalize -> network -> denormalize movement,
    clip to [-1, 1], gripper = close if logit > 0.
    """

    def __init__(
        self,
        model: MLP,
        obs_keys: list[str],
        obs_normalizer: Normalizer,
        move_normalizer: Normalizer,
        device: torch.device,
    ):
        self.model = model.to(device).eval()  # eval(): dropout off
        self.obs_keys = obs_keys
        self.obs_normalizer = obs_normalizer
        self.move_normalizer = move_normalizer
        self.device = device

    def reset(self) -> None:
        pass  # the MLP has no memory between steps

    @torch.no_grad()
    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        x = self.obs_normalizer.normalize(flatten_observation(obs, self.obs_keys))
        output = self.model(torch.from_numpy(x).unsqueeze(0).to(self.device))[0].cpu().numpy()
        move = np.clip(self.move_normalizer.denormalize(output[:MOVE_DIM]), -1.0, 1.0)
        gripper = 1.0 if output[MOVE_DIM] > 0 else -1.0
        return np.append(move, gripper).astype(np.float32)


def save_checkpoint(
    path: str | Path,
    model: MLP,
    model_config: dict[str, Any],
    obs_keys: list[str],
    obs_normalizer: Normalizer,
    move_normalizer: Normalizer,
    metadata: dict[str, Any],
) -> None:
    """Everything needed to run the policy later: weights, architecture, normalization."""
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config,
            "obs_keys": obs_keys,
            "obs_normalizer": obs_normalizer.to_dict(),
            "move_normalizer": move_normalizer.to_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    # weights_only=True: only tensors and plain Python values are loaded, never arbitrary code.
    return torch.load(path, map_location="cpu", weights_only=True)


def load_policy(path: str | Path, device: str = "cpu") -> NeuralPolicy:
    checkpoint = load_checkpoint(path)
    config = checkpoint["model_config"]
    model = MLP(config["obs_dim"], tuple(config["hidden_sizes"]), config["dropout"])
    model.load_state_dict(checkpoint["model_state"])
    return NeuralPolicy(
        model,
        checkpoint["obs_keys"],
        Normalizer.from_dict(checkpoint["obs_normalizer"]),
        Normalizer.from_dict(checkpoint["move_normalizer"]),
        resolve_device(device),
    )
