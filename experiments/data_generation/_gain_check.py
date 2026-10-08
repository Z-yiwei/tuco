"""Check Franka gripper gains after applying environment overrides."""
from __future__ import annotations

# Operational ranges and reference values for the gripper actuator.
EXPECTED_PANDA_HAND = {
    "stiffness": (500.0, 3000.0, 1000.0),       # (min, max, canonical)
    "damping":   (5.0, 50.0, 14.0),
    "effort_limit_sim": (30.0, 200.0, 60.0),
}


def check_franka_gains(env_cfg, task_name: str, strict: bool = True) -> None:
    """Validate the gripper stiffness, damping, and effort limit.

    Args:
        env_cfg: Environment configuration after Hydra overrides.
        task_name: Task identifier included in diagnostic messages.
        strict: Raise on invalid gains when True, otherwise print a warning.

    Raises:
        ValueError: A gain is outside its operational range.
    """
    try:
        actuators = env_cfg.scene.robot.actuators
        if hasattr(actuators, "__getitem__"):
            panda_hand = actuators["panda_hand"]
        else:
            panda_hand = getattr(actuators, "panda_hand", None)
        if panda_hand is None:
            print("[gain-check] panda_hand actuator not found (non-Franka task?), skip")
            return
    except Exception as e:
        print(f"[gain-check] skip (cannot read env_cfg.scene.robot.actuators: {e})")
        return

    violations = []
    snapshot = {}
    for attr, (lo, hi, canon) in EXPECTED_PANDA_HAND.items():
        v = getattr(panda_hand, attr, None)
        snapshot[attr] = v
        if v is None:
            continue
        if not (lo <= float(v) <= hi):
            violations.append(f"  - panda_hand.{attr}={v} OUT OF RANGE [{lo}, {hi}] (canonical={canon})")

    print(f"[gain-check] task={task_name}")
    print(f"[gain-check] panda_hand: stiffness={snapshot.get('stiffness')} "
          f"damping={snapshot.get('damping')} effort_limit_sim={snapshot.get('effort_limit_sim')}")

    if violations:
        msg = (
            "\n[FATAL gain-check] panda_hand gain values are outside the expected operational range.\n"
            + "\n".join(violations)
            + "\nCheck the task-specific gripper actuator overrides before collection.\n"
        )
        if strict:
            raise ValueError(msg)
        else:
            print("[WARN]" + msg)
