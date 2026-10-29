"""Load configuration (YAML files or saved dictionaries) into dataclasses."""

import dataclasses
from pathlib import Path
from typing import Any, TypeVar

import yaml

ConfigT = TypeVar("ConfigT")


def config_from_dict(config_class: type[ConfigT], values: dict[str, Any]) -> ConfigT:
    """Create ``config_class`` from a dictionary.

    Keys missing from the dictionary keep the dataclass defaults. An unknown key (e.g. a typo)
    raises a TypeError instead of being silently ignored. Lists become tuples, so a loaded
    config compares equal to one built in code and cannot be modified by accident.
    """
    values = {key: tuple(v) if isinstance(v, list) else v for key, v in values.items()}
    return config_class(**values)


def load_config(path: str | Path, config_class: type[ConfigT]) -> ConfigT:
    """Create ``config_class`` from a YAML file (see config_from_dict)."""
    with open(path, encoding="utf-8") as file:
        values = yaml.safe_load(file) or {}
    return config_from_dict(config_class, values)


def config_to_dict(config: Any) -> dict[str, Any]:
    """Plain dictionary of a dataclass config (for saving it next to results)."""
    return dataclasses.asdict(config)
