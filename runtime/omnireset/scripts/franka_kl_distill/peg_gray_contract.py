"""Fail-closed Peg/PegHole appearance contract; no Isaac imports required.

Validate the final resolved config, not the original USD's nominal material.
The appearance events are required because authored child materials can defeat
a spawn-level gray material. This is a config guard, not a pixel-level proof.
"""

import math

PEG_GRAY_RGB = (142 / 255, 144 / 255, 137 / 255)
OBJECTS = ("insertive_object", "receptive_object")


def is_peg_scene(env_cfg):
    for name in OBJECTS:
        spawn = getattr(getattr(env_cfg.scene, name, None), "spawn", None)
        path = str(getattr(spawn, "usd_path", "")).lower().replace("\\", "/")
        if "/peg/" in path or "/peghole/" in path or path.endswith("/peg.usd") or "peg_hole" in path:
            return True
    reset = getattr(env_cfg.events, "reset_from_reset_states", None)
    reset_types = getattr(reset, "params", {}).get("reset_types", [])
    return any(str(name).lower().startswith("peg") for name in reset_types)


def assert_peg_gray_contract(env_cfg, *, preserve_object_face_materials=False):
    """Return False for other tasks; reject unsafe Peg configs before gym.make."""
    if not is_peg_scene(env_cfg):
        return False
    prefix = "Peg gray-only contract (#8E9089)"
    if preserve_object_face_materials:
        raise ValueError(f"{prefix}: --preserve_object_face_materials is forbidden for Peg/PegHole")
    for name in OBJECTS:
        event_name = f"randomize_{name}_appearance"
        event = getattr(env_cfg.events, event_name, None)
        if event is None or getattr(event, "mode", None) not in {"startup", "reset", "interval"}:
            raise ValueError(f"{prefix}: {event_name} must remain active; asset colors are unsafe")
        params = event.params
        if params.get("texture_prob") != 0.0:
            raise ValueError(f"{prefix}: {event_name}.texture_prob must be 0")
        if getattr(params.get("asset_cfg"), "name", None) != name or params.get("mesh_names") != []:
            raise ValueError(f"{prefix}: {event_name} must cover all meshes of {name}")
        colors = params.get("colors", {})
        for channel, expected in zip("rgb", PEG_GRAY_RGB):
            bounds = colors.get(channel, ())
            if len(bounds) != 2 or not all(
                math.isclose(float(value), expected, rel_tol=0, abs_tol=1e-9) for value in bounds
            ):
                raise ValueError(f"{prefix}: {event_name}.colors[{channel}] must be {(expected, expected)}")
    return True
