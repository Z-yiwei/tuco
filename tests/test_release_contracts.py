import os
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python 3.9/3.10 single-simulator environment
    import tomli as tomllib

from tuco.config import PAPER_AGGREGATION, TucoConfig


ROOT = Path(__file__).resolve().parents[1]


def _toml(path: str) -> dict:
    return tomllib.loads((ROOT / path).read_text(encoding="utf-8"))


def test_fixed_paper_method_contract():
    config = TucoConfig()
    assert config.tau == 0.01
    assert config.lambda_cov == 0.2
    assert config.rho == 0.1
    assert PAPER_AGGREGATION == "target_sum_candidate_sum"


def test_sim2sim_source_target_and_evaluation_contract():
    for task in ("peg", "stackcube", "cupcake"):
        config = _toml(f"configs/sim2sim/{task}.toml")
        assert config["data"]["source_domain"] == "isaacsim"
        assert config["data"]["target_domain"] == "mujoco"
        assert config["evaluation"]["environment"] == "mujoco"
        assert config["evaluation"]["late_checkpoints"] == 5
        assert config["evaluation"]["rollouts_per_checkpoint"] == 50


def test_sim2real_repeat_and_real_rollout_contract():
    expected = {
        "peg": (10, 1, "delta_joint_step_v1"),
        "stackcube": (10, 2, "delta_joint_step_v1"),
        "cupcake": (9, 8, "absolute_joint_target_binary_width_v1"),
    }
    for task, (rollouts, action_steps, representation) in expected.items():
        config = _toml(f"configs/sim2real/{task}.toml")
        assert config["data"]["source_domain"] == "isaacsim"
        assert config["data"]["target_domain"] == "real_fr3"
        assert config["data"]["real_rollouts"] == rollouts
        assert config["data"]["real_action_steps"] == action_steps
        assert config["data"]["attribution_visual_repeats"] == 1
        assert config["data"]["attribution_repeat_index"] == 0
        assert config["training"]["action_representation"] == representation
        assert config["training"]["normalization_population"] == (
            "selected_sim_training_split"
        )
        assert config["training"]["terminal_epoch"] == 150
        assert config["training"]["num_epoch_indices"] == 151


def test_state_generation_uses_vendored_runtime_for_every_task(tmp_path):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "BUNDLE_URL" not in readme
    assert "user-collected" in readme

    launch = ROOT / "configs/launch/data_generation.env.example"
    environment = os.environ.copy()
    environment.update(
        DRY_RUN="1",
        OMNIRESET_ROOT=str(ROOT),
        DATA_ROOT=str(tmp_path / "state"),
        WORK_ROOT=str(tmp_path / "work"),
        SIM2REAL_DATA_ROOT=str(tmp_path / "vision"),
        PY_ISAAC=sys.executable,
        PY_MUJOCO=sys.executable,
        PEG_TEACHER=str(tmp_path / "peg.pt"),
        STACKCUBE_TEACHER=str(tmp_path / "stackcube.pt"),
        CUPCAKE_TEACHER=str(tmp_path / "cupcake.pt"),
    )
    for setting in ("sim2sim",):
        for task in ("peg", "stackcube", "cupcake"):
            result = subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts/generate_data.sh"),
                    setting,
                    str(launch),
                    task,
                    "all",
                ],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            assert "[done]" in result.stdout
            assert str(tmp_path) in result.stdout


def test_unresolved_cupcake_expert_is_not_silently_substituted(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/replay_expert.py"), "--task", "cupcake",
         "--output", str(tmp_path / "output"), "--dry-run"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "No other expert is substituted" in result.stderr
    assert not (tmp_path / "output").exists()
