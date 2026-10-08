#!/usr/bin/env python3
"""Release the authored CupCake SDF on a plane in Isaac/PhysX."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--cupcake-usd",
    type=Path,
    default=Path(__file__).resolve().parents[5] / "artifacts/local_assets/Props/Custom/CupCake/cupcake.usd",
)
parser.add_argument(
    "--collision-obj",
    type=Path,
    default=Path(__file__).resolve().parents[1]
    / "assets_mjcf/cupcake/cupcake_collision.obj",
)
parser.add_argument("--angles", type=float, nargs="+", default=[20.0, 50.0, 60.0])
parser.add_argument("--duration", type=float, default=5.0)
parser.add_argument("--out", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext


def vertices_from_obj(path: Path) -> np.ndarray:
    return np.asarray(
        [
            [float(value) for value in line.split()[1:4]]
            for line in path.read_text(encoding="ascii").splitlines()
            if line.startswith("v ")
        ],
        dtype=np.float64,
    )


def x_quaternion(degrees: float) -> np.ndarray:
    half_angle = math.radians(degrees) / 2.0
    return np.asarray([math.cos(half_angle), math.sin(half_angle), 0.0, 0.0])


def rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def tilt_degrees(quaternion: np.ndarray) -> float:
    local_z_world = rotation_matrix(quaternion)[:, 2]
    return math.degrees(math.acos(np.clip(local_z_world[2], -1.0, 1.0)))


def main() -> None:
    cupcake_usd = args.cupcake_usd.resolve()
    collision_obj = args.collision_obj.resolve()
    if not cupcake_usd.is_file():
        raise FileNotFoundError(cupcake_usd)
    if not collision_obj.is_file():
        raise FileNotFoundError(collision_obj)

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 240.0, device=args.device))
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        )
    )
    ground_cfg.func("/World/Ground", ground_cfg)
    cupcake_cfg = RigidObjectCfg(
        prim_path="/World/CupCake",
        spawn=sim_utils.UsdFileCfg(usd_path=str(cupcake_usd)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.12)),
    )
    cupcake = RigidObject(cupcake_cfg)
    sim.reset()
    cupcake.reset()

    vertices = vertices_from_obj(collision_obj)
    dt = sim.get_physics_dt()
    releases = []
    for angle in args.angles:
        quaternion = x_quaternion(angle)
        minimum_z = float((vertices @ rotation_matrix(quaternion).T)[:, 2].min())
        pose = torch.tensor(
            [[0.0, 0.0, -minimum_z + 0.0005, *quaternion]],
            dtype=torch.float32,
            device=cupcake.device,
        )
        velocity = torch.zeros((1, 6), dtype=torch.float32, device=cupcake.device)
        cupcake.write_root_pose_to_sim(pose)
        cupcake.write_root_velocity_to_sim(velocity)
        cupcake.reset()
        for _ in range(int(round(args.duration / dt))):
            cupcake.write_data_to_sim()
            sim.step(render=False)
            cupcake.update(dt)

        final_quaternion = cupcake.data.root_quat_w[0].cpu().numpy().astype(np.float64)
        final_velocity = cupcake.data.root_vel_w[0].cpu().numpy().astype(np.float64)
        releases.append(
            {
                "initial_tilt_deg": angle,
                "final_tilt_deg": tilt_degrees(final_quaternion),
                "final_speed": float(np.linalg.norm(final_velocity)),
                "final_position_m": cupcake.data.root_pos_w[0].cpu().tolist(),
                "final_quaternion_wxyz": final_quaternion.tolist(),
                "final_velocity": final_velocity.tolist(),
            }
        )

    result = {
        "definition": "Authored CupCake USD SDF released on an Isaac/PhysX ground plane",
        "cupcake_usd": str(cupcake_usd),
        "collision_obj": str(collision_obj),
        "duration_s": args.duration,
        "physics_dt_s": dt,
        "device": args.device,
        "releases": releases,
    }
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="ascii")
    print(output, end="")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
