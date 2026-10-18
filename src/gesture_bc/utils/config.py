"""Load YAML configuration files into dataclasses."""

from pathlib import Path
from typing import TypeVar

import yaml

ConfigT = TypeVar("ConfigT")


def load_config(path: str | Path, config_class: type[ConfigT]) -> ConfigT:
    """Create ``config_class`` from a YAML file.

    Keys missing from the file keep the dataclass defaults. An unknown key (e.g. a typo)
    raises a TypeError instead of being silently ignored. YAML lists become tuples, so a
    loaded config compares equal to one built in code and cannot be modified by accident.
    """
    with open(path, encoding="utf-8") as file:
        values = yaml.safe_load(file) or {}
    values = {
        key: tuple(value) if isinstance(value, list) else value for key, value in values.items()
    }
    return config_class(**values)
