"""YAML configuration loader for the RSA project.

Provides helpers to load experiment, model, and steering configs from
the ``configs/`` directory at the project root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Return the absolute path to the RSA project root.

    The project root is determined by walking upward from this file until
    a ``configs/`` directory is found.  Falls back to the ``RSA``
    environment variable if set, or to ``C:/RSA`` as a last resort.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "configs").is_dir():
            return parent
    env_root = os.environ.get("RSA")
    if env_root:
        return Path(env_root).resolve()
    return Path("C:/RSA").resolve()


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    path:
        Absolute or relative path to a ``.yaml`` / ``.yml`` file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML contents.  Returns an empty dict if the file is
        empty.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    yaml.YAMLError
        If the file contains invalid YAML.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def get_config(config_name: str) -> dict[str, Any]:
    """Load a config from the project's ``configs/`` directory.

    Parameters
    ----------
    config_name:
        Name of the config file, with or without the ``.yaml`` extension.
        For example, ``"model"`` loads ``configs/model.yaml``.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.
    """
    if not config_name.endswith((".yaml", ".yml")):
        config_name = f"{config_name}.yaml"
    config_path = get_project_root() / "configs" / config_name
    return load_config(config_path)
