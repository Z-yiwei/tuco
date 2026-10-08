"""Load and validate the experiment protocols shipped with the release."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # Python 3.9/3.10
    import tomli as tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SETTINGS = {"single_sim", "sim2sim", "sim2real"}
TASKS = {
    "single_sim": {"robomimic"},
    "sim2sim": {"peg", "stackcube", "cupcake"},
    "sim2real": {"peg", "stackcube", "cupcake"},
}


def load_protocol(setting: str, task: str) -> dict[str, Any]:
    """Return a checked protocol dictionary for one experiment family/task."""
    if setting not in SETTINGS:
        raise ValueError(f"unknown setting {setting!r}; expected one of {sorted(SETTINGS)}")
    if task not in TASKS[setting]:
        raise ValueError(
            f"unknown {setting} task {task!r}; expected one of {sorted(TASKS[setting])}"
        )
    path = REPOSITORY_ROOT / "configs" / setting / f"{task}.toml"
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    _validate(setting, task, config)
    return config


def value(setting: str, task: str, key: str) -> Any:
    """Read a dot-separated key from a checked protocol."""
    current: Any = load_protocol(setting, task)
    for component in key.split("."):
        if not isinstance(current, dict) or component not in current:
            raise KeyError(f"{setting}/{task} has no protocol value {key!r}")
        current = current[component]
    return current


def _validate(setting: str, task: str, config: dict[str, Any]) -> None:
    if setting == "sim2sim":
        data = config["data"]
        training = config["training"]
        evaluation = config["evaluation"]
        if data["target_domain"] != "mujoco" or data["source_domain"] != "isaacsim":
            raise ValueError(f"{task}: expected MuJoCo target and IsaacSim source")
        if training["steps"] % training["checkpoint_every"]:
            raise ValueError(f"{task}: steps must be divisible by checkpoint_every")
        if evaluation["late_checkpoints"] > training["keep_last"]:
            raise ValueError(f"{task}: evaluation needs more checkpoints than training keeps")
        if evaluation["environment"] != "mujoco":
            raise ValueError(f"{task}: formal evaluation must run in MuJoCo")
    elif setting == "sim2real":
        data = config["data"]
        training = config["training"]
        if data["target_domain"] != "real_fr3" or data["source_domain"] != "isaacsim":
            raise ValueError(f"{task}: expected real FR3 target and IsaacSim source")
        if data["selected_states"] > data["physical_states"]:
            raise ValueError(f"{task}: selected state count exceeds candidate count")
        if training["terminal_epoch"] + 1 != training["num_epoch_indices"]:
            raise ValueError(f"{task}: terminal epoch and epoch count disagree")

