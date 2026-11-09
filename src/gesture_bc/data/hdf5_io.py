"""Save and load demonstrations as HDF5 files (layout inspired by robomimic).

    file attributes            format, format_version, created, git_commit, source, ...
    /data                      attributes: obs_keys, num_demos, num_success
    /data/demo_00000/          attributes: seed, success, num_steps
        obs/<key>              (T, size) float32, one dataset per observation entry
        actions                (T, 4) float32, what the demonstrator chose
        executed_actions       (T, 4) float32, what was sent to the environment
        tcp_targets            (T, 3) float32
        states/qpos            (T, nq) float64
        states/qvel            (T, nv) float64

HDF5 is a single binary file holding named arrays in folders ("groups") plus small
attributes; reading one demo does not require loading the whole file.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from gesture_bc.data.episode import Episode

FORMAT = "gesture_bc.demos"
FORMAT_VERSION = 1


def _encode(value: Any) -> Any:
    """HDF5 attributes hold numbers and strings; store dicts and lists as JSON text."""
    return json.dumps(value) if isinstance(value, dict | list | tuple) else value


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str) and value[:1] in "[{":
        return json.loads(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


class DemoWriter:
    """Write episodes one by one, so a crash during collection keeps the earlier ones.

    Usage:
        with DemoWriter("data/expert.hdf5", {"source": "scripted_expert", ...}) as writer:
            writer.add(episode)
    """

    def __init__(self, path: str | Path, metadata: dict[str, Any]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w")
        self._file.attrs["format"] = FORMAT
        self._file.attrs["format_version"] = FORMAT_VERSION
        self._file.attrs["created"] = datetime.now().isoformat(timespec="seconds")
        for key, value in metadata.items():
            self._file.attrs[key] = _encode(value)
        self._data = self._file.create_group("data")
        self.num_demos = 0
        self.num_success = 0

    def add(self, episode: Episode) -> None:
        demo = self._data.create_group(f"demo_{self.num_demos:05d}")
        demo.attrs["seed"] = episode.seed
        demo.attrs["success"] = episode.success
        demo.attrs["num_steps"] = episode.num_steps
        obs_group = demo.create_group("obs")
        for key, values in episode.obs.items():
            obs_group.create_dataset(key, data=values)
        demo.create_dataset("actions", data=episode.actions)
        demo.create_dataset("executed_actions", data=episode.executed_actions)
        demo.create_dataset("tcp_targets", data=episode.tcp_targets)
        demo.create_dataset("states/qpos", data=episode.qpos)
        demo.create_dataset("states/qvel", data=episode.qvel)
        if self.num_demos == 0:
            self._data.attrs["obs_keys"] = _encode(list(episode.obs))
        self.num_demos += 1
        self.num_success += int(episode.success)
        self._data.attrs["num_demos"] = self.num_demos
        self._data.attrs["num_success"] = self.num_success
        self._file.flush()  # write to disk now

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "DemoWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def read_metadata(path: str | Path) -> dict[str, Any]:
    """File attributes (format, git commit, configs, ...) plus the /data attributes."""
    with h5py.File(path, "r") as file:
        metadata = {key: _decode(value) for key, value in file.attrs.items()}
        metadata.update({key: _decode(value) for key, value in file["data"].attrs.items()})
    return metadata


def read_demos(path: str | Path) -> list[Episode]:
    """All episodes of a file, in the order they were written."""
    episodes = []
    with h5py.File(path, "r") as file:
        if file.attrs.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file")
        data = file["data"]
        obs_keys = _decode(data.attrs["obs_keys"])
        for name in sorted(data):
            demo = data[name]
            episodes.append(
                Episode(
                    obs={key: demo["obs"][key][()] for key in obs_keys},
                    actions=demo["actions"][()],
                    executed_actions=demo["executed_actions"][()],
                    tcp_targets=demo["tcp_targets"][()],
                    qpos=demo["states/qpos"][()],
                    qvel=demo["states/qvel"][()],
                    seed=int(demo.attrs["seed"]),
                    success=bool(demo.attrs["success"]),
                )
            )
    return episodes
