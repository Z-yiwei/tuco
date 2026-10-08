#!/usr/bin/env python3
"""One-variable ablations for CupCake contact and tilted-release physics."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_cupcake_mujoco as Replay
import compare_peginsert_continuous_mujoco as B3
import cupcake_mujoco_model as CupCake


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ZARR = ROOT / "datasets/cupcake_sim2sim_t0_20260815/isaac_fit_b3center.zarr"
OLD_EXTRA_FRICTION = (0.05, 0.01, 0.01)


@dataclass(frozen=True)
class ContactProfile:
    name: str
    collision: str
    condim: int
    extra_friction: tuple[float, float, float]
    noslip_iterations: int
    friction_combine: str
    changed_from_old: str


PROFILES = (
    ContactProfile(
        "old_exact",
        "sdf",
        6,
        OLD_EXTRA_FRICTION,
        5,
        "legacy_b3",
        "none",
    ),
    ContactProfile(
        "collision_only",
        "compound",
        6,
        OLD_EXTRA_FRICTION,
        5,
        "legacy_b3",
        "SDF -> compound",
    ),
    ContactProfile(
        "condim_only",
        "sdf",
        3,
        OLD_EXTRA_FRICTION,
        5,
        "legacy_b3",
        "condim 6 -> 3",
    ),
    ContactProfile(
        "rolling_zero_only",
        "sdf",
        6,
        (0.0, 0.0, 0.0),
        5,
        "legacy_b3",
        "torsional/rolling coefficients -> 0",
    ),
    ContactProfile(
        "noslip_only",
        "sdf",
        6,
        OLD_EXTRA_FRICTION,
        0,
        "legacy_b3",
        "noslip iterations 5 -> 0",
    ),
    ContactProfile(
        "friction_combine_only",
        "sdf",
        6,
        OLD_EXTRA_FRICTION,
        5,
        "physx_average",
        "legacy max -> PhysX average",
    ),
    ContactProfile(
        "new_exact",
        "compound",
        3,
        (0.0, 0.0, 0.0),
        0,
        "physx_average",
        "all production changes",
    ),
)


def full_factorial_profiles() -> tuple[ContactProfile, ...]:
    profiles = []
    for collision, condim, rolling_zero, noslip, friction_combine in itertools.product(
        ("sdf", "compound"),
        (6, 3),
        (False, True),
        (5, 0),
        ("legacy_b3", "physx_average"),
    ):
        extra_friction = (0.0, 0.0, 0.0) if rolling_zero else OLD_EXTRA_FRICTION
        name = "_".join(
            (
                collision,
                f"dim{condim}",
                "roll0" if rolling_zero else "rollold",
                f"noslip{noslip}",
                "average" if friction_combine == "physx_average" else "legacy",
            )
        )
        profiles.append(
            ContactProfile(
                name,
                collision,
                condim,
                extra_friction,
                noslip,
                friction_combine,
                "full factorial cell",
            )
        )
    return tuple(profiles)


def x_rotation(degrees: float) -> np.ndarray:
    half_angle = math.radians(degrees) / 2.0
    return np.array([math.cos(half_angle), math.sin(half_angle), 0.0, 0.0])


def rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def tilt_degrees(quaternion: np.ndarray) -> float:
    local_z_world = rotation_matrix(quaternion)[:, 2]
    return math.degrees(math.acos(np.clip(local_z_world[2], -1.0, 1.0)))


def load_vertices(path: Path) -> np.ndarray:
    return np.asarray(
        [
            [float(value) for value in line.split()[1:4]]
            for line in path.read_text(encoding="ascii").splitlines()
            if line.startswith("v ")
        ],
        dtype=np.float64,
    )


def is_cupcake_table_pair(model: mujoco.MjModel, pair: int) -> bool:
    names = {
        mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom1[pair])
        ),
        mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom2[pair])
        ),
    }
    return "table_top" in names and any(
        name and name.startswith("cupcake_collision") for name in names
    )


def build_context(store, raw: np.ndarray, episode_end: int, profile: ContactProfile):
    model_profile = B3.b3_center_profile(physics_substeps=4)
    model_profile.update(
        cupcake_collision=profile.collision,
        hand_collision="disabled",
        finger_collision="menagerie_mesh_only",
        noslip_iterations=profile.noslip_iterations,
        plate_collision_enabled=False,
        contact_solref=[0.02, 1.0],
    )
    model = CupCake.build_model(raw, profile_cfg=model_profile)
    controller = CupCake.Controller(model)
    data = mujoco.MjData(model)
    Replay.configure_episode(
        model,
        controller,
        store,
        0,
        episode_end,
        raw,
        friction_combine=profile.friction_combine,
    )
    model.opt.noslip_iterations = profile.noslip_iterations
    for pair in range(model.npair):
        if not is_cupcake_table_pair(model, pair):
            continue
        model.pair_dim[pair] = profile.condim
        model.pair_friction[pair, 2:] = profile.extra_friction

    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if body in controller.robot_body_ids or body_name in {"plate", "floor"}:
            model.geom_contype[geom] = 0
            model.geom_conaffinity[geom] = 0
    mujoco.mj_forward(model, data)
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    table_xy = data.geom_xpos[table, :2].copy()
    table_top = float(data.geom_xpos[table, 2] + model.geom_size[table, 2])
    return model, data, controller, table_xy, table_top


def release(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: CupCake.Controller,
    vertices: np.ndarray,
    table_xy: np.ndarray,
    table_top: float,
    initial_tilt: float,
    duration: float,
) -> dict:
    mujoco.mj_resetData(model, data)
    quaternion = x_rotation(initial_tilt)
    mesh_min_z = float((vertices @ rotation_matrix(quaternion).T)[:, 2].min())
    qpos = controller.insertive_qpos
    dof = controller.insertive_dof
    data.qpos[qpos : qpos + 3] = [
        table_xy[0],
        table_xy[1],
        table_top - mesh_min_z + 0.0005,
    ]
    data.qpos[qpos + 3 : qpos + 7] = quaternion
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    minimum_distance = math.inf
    maximum_contacts = 0
    for _ in range(int(round(duration / model.opt.timestep))):
        mujoco.mj_step(model, data)
        maximum_contacts = max(maximum_contacts, int(data.ncon))
        if data.ncon:
            minimum_distance = min(
                minimum_distance,
                min(float(contact.dist) for contact in data.contact[: data.ncon]),
            )
    final_tilt = tilt_degrees(data.qpos[qpos + 3 : qpos + 7])
    return {
        "initial_tilt_deg": initial_tilt,
        "final_tilt_deg": final_tilt,
        "absolute_tilt_change_deg": abs(final_tilt - initial_tilt),
        "final_speed": float(np.linalg.norm(data.qvel[dof : dof + 6])),
        "minimum_contact_distance_m": (
            minimum_distance if math.isfinite(minimum_distance) else None
        ),
        "maximum_contact_count": maximum_contacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, default=DEFAULT_ZARR)
    parser.add_argument("--angles", type=float, nargs="+", default=[40.0, 60.0])
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument(
        "--matrix",
        choices=("one_at_a_time", "full_factorial"),
        default="one_at_a_time",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "log/active/cupcake_sim2sim_20260815/contact_factor_ablation.json",
    )
    args = parser.parse_args()
    store = zarr.open(str(args.zarr), mode="r")
    Replay.configure_timing(store)
    raw = np.asarray(store["data/raw_state"][0], dtype=np.float64)
    episode_end = int(np.asarray(store["meta/episode_ends"])[0])
    vertices = load_vertices(CupCake.CUPCAKE_MESH)

    profiles = PROFILES if args.matrix == "one_at_a_time" else full_factorial_profiles()
    results = []
    for profile in profiles:
        model, data, controller, table_xy, table_top = build_context(
            store, raw, episode_end, profile
        )
        releases = [
            release(
                model,
                data,
                controller,
                vertices,
                table_xy,
                table_top,
                angle,
                args.duration,
            )
            for angle in args.angles
        ]
        results.append({"profile": asdict(profile), "releases": releases})
        print(
            profile.name,
            " ".join(
                f"{row['initial_tilt_deg']:.0f}->{row['final_tilt_deg']:.2f}deg"
                for row in releases
            ),
            flush=True,
        )

    output = {
        "definition": "one-variable changes from the exact old tilted-release profile",
        "matrix": args.matrix,
        "duration_s": args.duration,
        "angles_deg": args.angles,
        "source_zarr": str(args.zarr.resolve()),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
