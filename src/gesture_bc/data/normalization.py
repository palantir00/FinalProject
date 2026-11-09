"""Scale inputs and outputs of a network to mean 0 and standard deviation 1.

Positions are ~0.5 m, velocities ~0.1 m/s, opening ~0.05 m: without scaling, large inputs
would dominate the first layer and training would be slow and unstable. The statistics are
computed on the TRAINING split only, so no information from validation leaks into training.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, data: np.ndarray, min_std: float = 1e-3) -> "Normalizer":
        """Per-column statistics of ``data`` (rows = samples).

        A column that (almost) never changes, e.g. the target height, would be divided by ~0;
        min_std keeps such columns finite.
        """
        mean = data.mean(axis=0)
        std = np.maximum(data.std(axis=0), min_std)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def denormalize(self, y: np.ndarray) -> np.ndarray:
        return (y * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, values: dict[str, list[float]]) -> "Normalizer":
        return cls(np.array(values["mean"], np.float32), np.array(values["std"], np.float32))
