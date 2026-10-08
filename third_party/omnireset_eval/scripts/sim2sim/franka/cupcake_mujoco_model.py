"""CupCake-on-Plate MuJoCo scene adapter for the audited Franka B3 runtime."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_loop_eval as CL
import quat_utils as Q


ASSET_DIR = Path(__file__).resolve().parent.parent / "assets_mjcf" / "cupcake"
CUPCAKE_MESH = ASSET_DIR / "cupcake_collision.obj"
CUPCAKE_VISUAL_MESH = ASSET_DIR / "cupcake_visual.obj"
PLATE_MESH = ASSET_DIR / "plate_collision.obj"
COACD_DIR = ASSET_DIR / "coacd"
COACD_MANIFEST = COACD_DIR / "manifest.json"
CONVEX_DECOMPOSITION_DIR = ASSET_DIR / "coacd_fine"
CONVEX_DECOMPOSITION_MANIFEST = CONVEX_DECOMPOSITION_DIR / "manifest.json"
PLATE_DECOMPOSITION_DIR = ASSET_DIR / "convex_radial16"
PLATE_DECOMPOSITION_MANIFEST = PLATE_DECOMPOSITION_DIR / "manifest.json"
RADIAL32_DECOMPOSITION_DIR = ASSET_DIR / "convex_radial32"
RADIAL32_DECOMPOSITION_MANIFEST = RADIAL32_DECOMPOSITION_DIR / "manifest.json"
PLATE_COACD_DIR = ASSET_DIR / "plate_coacd"
PLATE_COACD_MANIFEST = PLATE_COACD_DIR / "manifest.json"
SOURCE_HAND_MESH = ASSET_DIR.parent / "franka_mimic_hand_col.obj"

CUPCAKE_ASSEMBLED_POS = np.array([0.0, 0.0, -0.003], dtype=np.float64)
CUPCAKE_ASSEMBLED_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
PLATE_ASSEMBLED_POS = np.zeros(3, dtype=np.float64)
PLATE_ASSEMBLED_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
SUCCESS_POSITION_M = 0.005
SUCCESS_ORIENTATION_XY_RAD = 0.025
RELAXED_SUCCESS_POSITION_M = 0.015
RELAXED_SUCCESS_ORIENTATION_XY_RAD = 0.15
CUPCAKE_CONTACT_SOLREF = [0.02, 1.0]
ROBOT_OBJECT_CONTACT_SOLREF = [0.005, 1.0]
CUPCAKE_CONTACT_SOLIMP = [0.95, 0.99, 0.001, 0.5, 2.0]
CUPCAKE_CONTACT_OFFSET = 0.0013625
ROBOT_CONTACT_OFFSET = 0.005
OBJECT_OBJECT_CONTACT_MARGIN = 2.0 * CUPCAKE_CONTACT_OFFSET
ROBOT_OBJECT_CONTACT_MARGIN = ROBOT_CONTACT_OFFSET + CUPCAKE_CONTACT_OFFSET
CUPCAKE_COMPOUND_GEOMS = (
    (
        "cupcake_collision_base",
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        [0.0, 0.0, 0.025],
        [0.039, 0.025, 0.0],
    ),
    (
        "cupcake_collision_mid",
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        [0.0, 0.0, 0.055],
        [0.043, 0.043, 0.025],
    ),
    (
        "cupcake_collision_top",
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        [0.0, 0.0, 0.080],
        [0.028, 0.028, 0.020],
    ),
)
OBJECT_CONTACT_FRICTION = [1.0, 1.0, 0.0, 0.0, 0.0]
FINGER_CONTACT_FRICTION = [2.0, 2.0, 0.0, 0.0, 0.0]
TABLE_WORLD_POS = np.array([0.4, 0.0, -0.881], dtype=np.float64)
TABLE_WORLD_QUAT = np.array(
    [0.7071067811865476, 0.0, 0.0, -0.7071067811865475], dtype=np.float64
)
TABLE_ASSET_QUAT = TABLE_WORLD_QUAT.copy()
# Exact collision cubes from pat_vention.usd in asset-local coordinates.
TABLE_COLLISION_BOXES = (
    ([0.0, 0.0, 0.835], [1.3648233219, 1.0865139076, 0.066]),
    ([0.5101859392, 0.6530382172, 0.9925399437], [0.04, 0.04, 2.0]),
    ([-0.5114538018, 0.6530382172, 0.9925399437], [0.04, 0.04, 2.0]),
    ([-0.5114538018, -0.6368841546, 0.9925399437], [0.04, 0.04, 2.0]),
    ([0.5111775908, -0.6368841546, 0.9925399437], [0.04, 0.04, 2.0]),
    ([0.0, 0.0, 1.9654242394], [1.3648233219, 1.0865139076, 0.0380806107]),
    ([0.0, 0.0, 0.1188022451], [1.3648233219, 1.0865139076, 0.0463817015]),
)
TABLE_TOP_WORLD_Z = -0.013


def collision_piece_records(asset_dir: Path, manifest_path: Path) -> list[dict]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    pieces = list(manifest.get("pieces", []))
    if len(pieces) != int(manifest.get("piece_count", -1)) or not pieces:
        raise ValueError(f"invalid convex decomposition manifest: {manifest_path}")
    resolved_pieces = []
    for piece in pieces:
        path = asset_dir / piece["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        piece = piece.copy()
        piece["path"] = str(path)
        resolved_pieces.append(piece)
    return resolved_pieces


def convex_decomposition_piece_records() -> list[dict]:
    return collision_piece_records(
        CONVEX_DECOMPOSITION_DIR, CONVEX_DECOMPOSITION_MANIFEST
    )


def plate_decomposition_piece_records() -> list[dict]:
    return collision_piece_records(
        PLATE_DECOMPOSITION_DIR, PLATE_DECOMPOSITION_MANIFEST
    )


def radial32_decomposition_piece_records() -> list[dict]:
    return collision_piece_records(
        RADIAL32_DECOMPOSITION_DIR, RADIAL32_DECOMPOSITION_MANIFEST
    )


def coacd_piece_records() -> list[dict]:
    return collision_piece_records(COACD_DIR, COACD_MANIFEST)


def plate_coacd_piece_records() -> list[dict]:
    return collision_piece_records(PLATE_COACD_DIR, PLATE_COACD_MANIFEST)


def pose_in_robot_root(raw: np.ndarray, position: slice, quaternion: slice):
    return Q.subtract_frame_transforms(
        raw[18:21], raw[21:25], raw[position], raw[quaternion]
    )


def _configure_robot(spec, profile_cfg) -> None:
    gripper_cfg = profile_cfg.get("gripper", CL.B3_GRIPPER)
    finger_collision = profile_cfg.get("finger_collision", "mimic")
    for body in spec.bodies:
        if body.name != "world":
            body.gravcomp = 1.0

    for actuator in spec.actuators:
        if actuator.name in {f"actuator{index}" for index in range(1, 8)}:
            actuator.gainprm = np.zeros_like(np.asarray(actuator.gainprm))
            actuator.biasprm = np.zeros_like(np.asarray(actuator.biasprm))
        if actuator.name == "actuator8":
            stiffness = float(gripper_cfg["tendon_stiffness"])
            damping = float(gripper_cfg["tendon_damping"])
            force_limit = float(gripper_cfg["tendon_force_limit"])
            control_gain = gripper_cfg.get("control_gain")
            if control_gain is None:
                control_gain = stiffness * 0.04 / 255.0
            actuator.gainprm = np.array(
                [control_gain, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            )
            actuator.biasprm = np.array(
                [0.0, -stiffness, -damping, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            )
            actuator.forcerange = [-force_limit, force_limit]

    if finger_collision == "mimic":
        for body_name in ("left_finger", "right_finger"):
            for geom in spec.body(body_name).geoms:
                if str(geom.type).endswith("MESH") and geom.contype != 0:
                    geom.contype = 0
                    geom.conaffinity = 0

        spec.add_mesh(name="mimic_lf", file=CL.MIMIC_LF)
        spec.add_mesh(name="mimic_rf", file=CL.MIMIC_RF)
        for body_name, mesh_name in (
            ("left_finger", "mimic_lf"),
            ("right_finger", "mimic_rf"),
        ):
            body = spec.body(body_name)
            for geom in body.geoms:
                if str(geom.type).endswith("BOX"):
                    geom.contype = 0
                    geom.conaffinity = 0
            body.add_geom(
                name=f"{body_name}_mimic",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                friction=[2.0, 0.1, 0.05],
                condim=6,
                rgba=[0.70, 0.70, 0.72, 1.0],
            )
    elif finger_collision == "menagerie_mesh_only":
        for body_name in ("left_finger", "right_finger"):
            for geom in spec.body(body_name).geoms:
                if not str(geom.type).endswith("MESH"):
                    geom.contype = 0
                    geom.conaffinity = 0

    hand_collision = profile_cfg.get("hand_collision", "menagerie")
    if hand_collision == "disabled":
        hand = spec.body("hand")
        for geom in hand.geoms:
            if geom.contype or geom.conaffinity:
                geom.contype = 0
                geom.conaffinity = 0
    elif hand_collision == "source_usd":
        if not SOURCE_HAND_MESH.is_file():
            raise FileNotFoundError(SOURCE_HAND_MESH)
        hand = spec.body("hand")
        for geom in hand.geoms:
            if geom.contype or geom.conaffinity:
                geom.contype = 0
                geom.conaffinity = 0
        spec.add_mesh(name="mimic_hand", file=str(SOURCE_HAND_MESH))
        hand.add_geom(
            name="hand_mimic",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="mimic_hand",
            friction=[1.0, 0.05, 0.01],
            condim=6,
            rgba=[0.70, 0.70, 0.72, 1.0],
        )


def build_model(
    raw_state: np.ndarray,
    profile_cfg: dict | None = None,
    insertive_mass: float = 0.11,
) -> mujoco.MjModel:
    """Build one episode's scene in the robot-root coordinate frame."""
    raw_state = np.asarray(raw_state, dtype=np.float64)
    if raw_state.shape != (57,):
        raise ValueError(f"expected 57-D raw state, got {raw_state.shape}")
    for path in (CUPCAKE_MESH, PLATE_MESH):
        if not path.is_file():
            raise FileNotFoundError(path)

    profile_cfg = profile_cfg or {"gripper": CL.B3_GRIPPER, "physics_substeps": 1}
    cupcake_collision = profile_cfg.get("cupcake_collision", "convex_decomposition")
    cupcake_plate_collision = profile_cfg.get(
        "cupcake_plate_collision", "base_cylinder"
    )
    plate_collision_enabled = bool(profile_cfg.get("plate_collision_enabled", True))
    contact_solref = np.asarray(
        profile_cfg.get("contact_solref", CUPCAKE_CONTACT_SOLREF), dtype=np.float64
    )
    robot_contact_solref = np.asarray(
        profile_cfg.get("robot_contact_solref", ROBOT_OBJECT_CONTACT_SOLREF),
        dtype=np.float64,
    )
    contact_solimp = np.asarray(
        profile_cfg.get("contact_solimp", CUPCAKE_CONTACT_SOLIMP), dtype=np.float64
    )
    robot_contact_solimp = np.asarray(
        profile_cfg.get("robot_contact_solimp", CUPCAKE_CONTACT_SOLIMP),
        dtype=np.float64,
    )
    if contact_solref.shape != (2,) or np.any(contact_solref <= 0.0):
        raise ValueError("contact_solref must contain positive timeconstant/dampratio")
    if robot_contact_solref.shape != (2,) or np.any(robot_contact_solref <= 0.0):
        raise ValueError(
            "robot_contact_solref must contain positive timeconstant/dampratio"
        )
    if contact_solimp.shape != (5,) or robot_contact_solimp.shape != (5,):
        raise ValueError("contact solimp profiles must contain five values")
    contact_solref = contact_solref.tolist()
    robot_contact_solref = robot_contact_solref.tolist()
    contact_solimp = contact_solimp.tolist()
    robot_contact_solimp = robot_contact_solimp.tolist()
    if cupcake_collision not in {
        "convex_decomposition",
        "radial32",
        "coacd",
        "sdf",
        "convex_mesh",
        "compound",
    }:
        raise ValueError(f"unknown CupCake collision representation: {cupcake_collision}")
    if cupcake_plate_collision not in {
        "convex_hull",
        "radial16",
        "base_cylinder",
    }:
        raise ValueError(
            "unknown CupCake-to-plate collision representation: "
            f"{cupcake_plate_collision}"
        )
    if profile_cfg.get("hand_collision", "menagerie") not in {
        "menagerie",
        "source_usd",
        "disabled",
    }:
        raise ValueError("unknown panda_hand collision representation")
    if profile_cfg.get("finger_collision", "mimic") not in {
        "mimic",
        "menagerie",
        "menagerie_mesh_only",
    }:
        raise ValueError("unknown finger collision representation")
    spec = mujoco.MjSpec.from_file(CL.PANDA)
    spec.memory = int(profile_cfg.get("memory_bytes", 64 << 20))
    spec.option.timestep = CL.SIM_DT / int(profile_cfg.get("physics_substeps", 1))
    integrators = {
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
        "rk4": mujoco.mjtIntegrator.mjINT_RK4,
    }
    integrator = profile_cfg.get("integrator", "implicitfast")
    if integrator not in integrators:
        raise ValueError(f"unknown integrator: {integrator}")
    spec.option.integrator = integrators[integrator]
    spec.option.impratio = 10.0
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    # PhysX's patch-friction contact has two tangential constraints.  MuJoCo's
    # noslip post-processing plus condim=6 adds torsional/rolling constraints
    # that can hold this object at an arbitrary release angle.
    spec.option.noslip_iterations = int(profile_cfg.get("noslip_iterations", 0))
    spec.option.iterations = 150
    _configure_robot(spec, profile_cfg)

    spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    world = spec.worldbody
    world.add_light(
        pos=[0.4, 0.0, 1.6],
        dir=[0.0, 0.0, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.7, 0.7, 0.7],
    )
    world.add_light(
        pos=[0.4, -0.8, 1.2],
        dir=[0.0, 0.6, -1.0],
        diffuse=[0.4, 0.4, 0.4],
    )
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[3.0, 3.0, 0.1],
        pos=[0.0, 0.0, -0.5],
        rgba=[0.30, 0.30, 0.32, 1.0],
    )

    table_pos, table_quat = Q.subtract_frame_transforms(
        raw_state[18:21],
        raw_state[21:25],
        TABLE_WORLD_POS,
        TABLE_WORLD_QUAT,
    )
    table = world.add_body(name="table", pos=table_pos.tolist(), quat=table_quat.tolist())
    for index, (position, full_size) in enumerate(TABLE_COLLISION_BOXES):
        table.add_geom(
            name="table_top" if index == 0 else f"table_collision_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=position,
            quat=TABLE_ASSET_QUAT,
            size=(0.5 * np.asarray(full_size, dtype=np.float64)).tolist(),
            rgba=[0.45, 0.45, 0.50, 1.0],
            condim=3,
            friction=[1.0, 0.05, 0.01],
        )

    spec.add_mesh(name="cupcake_mesh", file=str(CUPCAKE_VISUAL_MESH))
    spec.add_mesh(name="plate_mesh", file=str(PLATE_MESH))
    decomposition_pieces = (
        convex_decomposition_piece_records()
        if cupcake_collision == "convex_decomposition"
        else radial32_decomposition_piece_records()
        if cupcake_collision == "radial32"
        else coacd_piece_records()
        if cupcake_collision == "coacd"
        else []
    )
    plate_decomposition_pieces = (
        plate_decomposition_piece_records()
        if cupcake_collision == "convex_decomposition"
        and cupcake_plate_collision == "radial16"
        else []
    )
    plate_collision_pieces = plate_coacd_piece_records()
    for piece in decomposition_pieces:
        spec.add_mesh(name=piece["name"], file=piece["path"])
    for index, piece in enumerate(plate_decomposition_pieces):
        spec.add_mesh(
            name=f"cupcake_plate_piece_mesh_{index:02d}", file=piece["path"]
        )
    for index, piece in enumerate(plate_collision_pieces):
        spec.add_mesh(name=f"plate_collision_mesh_{index:02d}", file=piece["path"])
    cupcake_pos, cupcake_quat = pose_in_robot_root(
        raw_state, slice(31, 34), slice(34, 38)
    )
    cupcake = world.add_body(
        name="cupcake", pos=cupcake_pos.tolist(), quat=cupcake_quat.tolist()
    )
    cupcake.add_freejoint(name="cupcake_freejoint")
    cupcake.add_geom(
        name="cupcake_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="cupcake_mesh",
        mass=float(insertive_mass),
        contype=0,
        conaffinity=0,
        rgba=[0.55, 0.12, 0.05, 1.0],
    )
    if cupcake_collision in {"convex_decomposition", "radial32", "coacd"}:
        collision_geom_names = []
        representation = (
            "convex_piece"
            if cupcake_collision == "convex_decomposition"
            else "radial32_piece"
            if cupcake_collision == "radial32"
            else "coacd_piece"
        )
        for index, piece in enumerate(decomposition_pieces):
            geom_name = f"cupcake_collision_{representation}_{index:02d}"
            cupcake.add_geom(
                name=geom_name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=piece["name"],
                mass=0.0,
                # A separate bit keeps these pieces active for diagnostics while
                # preventing duplicate implicit contacts with the plate pieces;
                # all CupCake contacts are defined by explicit pair entries.
                contype=2,
                conaffinity=0,
                friction=[1.5, 0.05, 0.01],
                condim=3,
                solref=contact_solref,
                solimp=contact_solimp,
                rgba=[0.0, 0.0, 0.0, 0.0],
            )
            collision_geom_names.append(geom_name)
        if cupcake_collision in {"convex_decomposition", "radial32"}:
            plate_collision_geom_names = []
            if cupcake_plate_collision == "convex_hull":
                geom_name = "cupcake_collision_plate_hull"
                cupcake.add_geom(
                    name=geom_name,
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname="cupcake_mesh",
                    mass=0.0,
                    contype=0,
                    conaffinity=0,
                    friction=[1.5, 0.05, 0.01],
                    condim=3,
                    solref=contact_solref,
                    solimp=contact_solimp,
                    rgba=[0.0, 0.0, 0.0, 0.0],
                )
                plate_collision_geom_names.append(geom_name)
            elif cupcake_plate_collision == "radial16":
                for index, piece in enumerate(plate_decomposition_pieces):
                    geom_name = f"cupcake_collision_plate_piece_{index:02d}"
                    cupcake.add_geom(
                        name=geom_name,
                        type=mujoco.mjtGeom.mjGEOM_MESH,
                        meshname=f"cupcake_plate_piece_mesh_{index:02d}",
                        mass=0.0,
                        contype=0,
                        conaffinity=0,
                        friction=[1.5, 0.05, 0.01],
                        condim=3,
                        solref=contact_solref,
                        solimp=contact_solimp,
                        rgba=[0.0, 0.0, 0.0, 0.0],
                    )
                    plate_collision_geom_names.append(geom_name)
            else:
                geom_name = "cupcake_collision_plate_base"
                cupcake.add_geom(
                    name=geom_name,
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=[0.0, 0.0, 0.025],
                    size=[0.039, 0.025, 0.0],
                    mass=0.0,
                    contype=0,
                    conaffinity=0,
                    friction=[1.5, 0.05, 0.01],
                    condim=3,
                    solref=contact_solref,
                    solimp=contact_solimp,
                    rgba=[0.0, 0.0, 0.0, 0.0],
                )
                plate_collision_geom_names.append(geom_name)
        else:
            plate_collision_geom_names = collision_geom_names
    elif cupcake_collision == "compound":
        collision_geom_names = []
        for name, geom_type, position, size in CUPCAKE_COMPOUND_GEOMS:
            cupcake.add_geom(
                name=name,
                type=geom_type,
                pos=position,
                size=size,
                mass=0.0,
                friction=[1.5, 0.05, 0.01],
                condim=3,
                solref=contact_solref,
                solimp=contact_solimp,
                rgba=[0.0, 0.0, 0.0, 0.0],
            )
            collision_geom_names.append(name)
        plate_collision_geom_names = collision_geom_names
    else:
        cupcake.add_geom(
            name="cupcake_collision",
            type=(
                mujoco.mjtGeom.mjGEOM_SDF
                if cupcake_collision == "sdf"
                else mujoco.mjtGeom.mjGEOM_MESH
            ),
            meshname="cupcake_mesh",
            mass=0.0,
            friction=[1.5, 0.05, 0.01],
            condim=3,
            solref=contact_solref,
            solimp=contact_solimp,
            rgba=[0.0, 0.0, 0.0, 0.0],
        )
        collision_geom_names = ["cupcake_collision"]
        plate_collision_geom_names = collision_geom_names

    plate_pos, plate_quat = pose_in_robot_root(
        raw_state, slice(44, 47), slice(47, 51)
    )
    plate = world.add_body(name="plate", pos=plate_pos.tolist(), quat=plate_quat.tolist())
    plate.add_geom(
        name="plate_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="plate_mesh",
        contype=0,
        conaffinity=0,
        rgba=[0.92, 0.92, 0.92, 1.0],
    )
    # The source USD uses a 2 mm wide visual ring at z=6.5 mm. Two thin,
    # collision-free discs reproduce the annulus without adding a planar mesh.
    plate.add_geom(
        name="plate_target_ring_outer",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0, 0.0, 0.00645],
        size=[0.041327, 0.00005, 0.0],
        mass=0.0,
        contype=0,
        conaffinity=0,
        rgba=[0.95, 0.05, 0.05, 1.0],
    )
    plate.add_geom(
        name="plate_target_ring_inner_mask",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0, 0.0, 0.00651],
        size=[0.039327, 0.00004, 0.0],
        mass=0.0,
        contype=0,
        conaffinity=0,
        rgba=[0.92, 0.92, 0.92, 1.0],
    )
    plate_collision_names = []
    for index, _piece in enumerate(plate_collision_pieces):
        geom_name = "plate_collision" if index == 0 else f"plate_collision_piece_{index:02d}"
        plate.add_geom(
            name=geom_name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"plate_collision_mesh_{index:02d}",
            contype=int(plate_collision_enabled),
            conaffinity=int(plate_collision_enabled),
            friction=[0.35, 0.02, 0.01],
            condim=3,
            solref=contact_solref,
            solimp=contact_solimp,
            rgba=[0.0, 0.0, 0.0, 0.0],
        )
        plate_collision_names.append(geom_name)

    # Pair entries let the runtime adapter reproduce PhysX's per-material
    # friction combine rule instead of MuJoCo's implicit maximum.
    for cupcake_geom in collision_geom_names:
        spec.add_pair(
            geomname1=cupcake_geom,
            geomname2="table_top",
            condim=3,
            friction=OBJECT_CONTACT_FRICTION,
            solref=contact_solref,
            solimp=contact_solimp,
        )
    if plate_collision_enabled:
        for cupcake_geom in plate_collision_geom_names:
            for plate_geom in plate_collision_names:
                spec.add_pair(
                    geomname1=cupcake_geom,
                    geomname2=plate_geom,
                    condim=3,
                    friction=OBJECT_CONTACT_FRICTION,
                    solref=contact_solref,
                    solimp=contact_solimp,
                )

    gripper_cfg = profile_cfg.get("gripper", CL.B3_GRIPPER)
    if (
        gripper_cfg.get("finger_contact_solref") is not None
        and profile_cfg.get("finger_collision", "mimic") == "mimic"
    ):
        for cupcake_geom in collision_geom_names:
            for finger in ("left_finger", "right_finger"):
                spec.add_pair(
                    geomname1=cupcake_geom,
                    geomname2=f"{finger}_mimic",
                    condim=3,
                    friction=FINGER_CONTACT_FRICTION,
                    solref=robot_contact_solref,
                    solimp=robot_contact_solimp,
                )
            if profile_cfg.get("hand_collision") == "source_usd":
                spec.add_pair(
                    geomname1=cupcake_geom,
                    geomname2="hand_mimic",
                    condim=3,
                    friction=OBJECT_CONTACT_FRICTION,
                    solref=robot_contact_solref,
                    solimp=robot_contact_solimp,
                )

    world.add_camera(
        name="cam_front",
        pos=[1.05, -0.55, 0.62],
        xyaxes=[0.5, 0.86, 0.0, -0.28, 0.16, 0.95],
    )
    world.add_camera(
        name="cam_side",
        pos=[0.4, -0.9, 0.45],
        xyaxes=[1.0, 0.0, 0.0, 0.0, 0.35, 0.94],
    )
    world.add_camera(
        name="cam_task",
        pos=[0.72, 0.72, 0.34],
        xyaxes=[-0.778, 0.629, 0.0, -0.280, -0.346, 0.896],
    )
    world.add_camera(
        name="cam_top",
        pos=[0.34, 0.25, 0.72],
        xyaxes=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    # Match scripts/record_rl_episode.py's native Isaac observer camera.
    world.add_camera(
        name="cam_isaac_observer",
        pos=[1.45, -0.75, 0.80],
        xyaxes=[
            0.573558,
            0.819165,
            0.0,
            -0.376803,
            0.263762,
            0.887911,
        ],
        fovy=32.84,
    )
    model = spec.compile()

    sysid = profile_cfg.get("sysid")
    if sysid is not None:
        model.dof_armature[:7] = np.asarray(sysid["armature"], dtype=np.float64)
        model.dof_frictionloss[:7] = np.asarray(
            sysid["static_friction"], dtype=np.float64
        )
        model.dof_damping[:7] = np.asarray(sysid["viscous_friction"], dtype=np.float64)
    else:
        model.dof_damping[:7] = 0.0
    if gripper_cfg.get("joint_armature") is not None:
        model.dof_armature[7:9] = float(gripper_cfg["joint_armature"])
    if gripper_cfg.get("joint_friction") is not None:
        model.dof_frictionloss[7:9] = float(gripper_cfg["joint_friction"])
    if gripper_cfg.get("profile") == "b3_training_center":
        model.dof_damping[7:9] = 0.0
    return model


class Controller:
    """Task handles plus the interface required by the audited B3 executor."""

    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.l0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0")
        self.insertive = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "cupcake"
        )
        self.receptive = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        self.insertive_geom_ids = {
            geom
            for geom in range(model.ngeom)
            if int(model.geom_bodyid[geom]) == self.insertive
            and (model.geom_contype[geom] or model.geom_conaffinity[geom])
        }
        if not self.insertive_geom_ids:
            raise ValueError("CupCake body has no active collision geoms")
        self.insertive_geom = min(self.insertive_geom_ids)
        self.grip_act = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
        )
        joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "cupcake_freejoint"
        )
        self.insertive_qpos = int(model.jnt_qposadr[joint])
        self.insertive_dof = int(model.jnt_dofadr[joint])
        self.jacp = np.zeros((3, model.nv), dtype=np.float64)
        self.jacr = np.zeros((3, model.nv), dtype=np.float64)
        self.robot_body_ids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in [
                *(f"link{index}" for index in range(8)),
                "hand",
                "left_finger",
                "right_finger",
            ]
        }

        # Compatibility aliases let the existing audited B3 executor operate
        # on a generic insertive/receptive task without duplicating it.
        self.peg = self.insertive
        self.hole = self.receptive
        self.peg_geom = self.insertive_geom
        self.peg_qpos = self.insertive_qpos
        self.peg_dof = self.insertive_dof

    def ee_root(self, data: mujoco.MjData):
        return Q.subtract_frame_transforms(
            data.xpos[self.l0],
            data.xquat[self.l0],
            data.xpos[self.hand].copy(),
            data.xquat[self.hand].copy(),
        )

    def jac_arm(self, data: mujoco.MjData):
        mujoco.mj_jac(
            self.m,
            data,
            self.jacp,
            self.jacr,
            data.xpos[self.hand],
            self.hand,
        )
        return np.vstack([self.jacp[:, :7], self.jacr[:, :7]])

    def grasp_close(self, data: mujoco.MjData) -> bool:
        hand_pos = data.xpos[self.hand]
        hand_quat = data.xquat[self.hand]
        tcp = hand_pos + Q.quat_apply(hand_quat, CL.TCP_OFF)
        local = Q.quat_apply(
            Q.quat_inv(hand_quat), data.xpos[self.insertive] - tcp
        )
        return bool(
            np.hypot(local[0], local[1]) < CL.LAT_THRESH
            and abs(local[2]) < CL.VERT_THRESH
        )

    def insertive_robot_contact_bodies(self, data: mujoco.MjData) -> set[str]:
        bodies = set()
        for contact in data.contact[: data.ncon]:
            if contact.geom1 in self.insertive_geom_ids:
                other = contact.geom2
            elif contact.geom2 in self.insertive_geom_ids:
                other = contact.geom1
            else:
                continue
            body = int(self.m.geom_bodyid[other])
            if body in self.robot_body_ids:
                bodies.add(
                    mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, body)
                )
        return bodies

    def insertive_robot_contact(self, data: mujoco.MjData) -> bool:
        return bool(self.insertive_robot_contact_bodies(data))

    # Compatibility with the shared Peg B3 diagnostics.
    peg_robot_contact_bodies = insertive_robot_contact_bodies
    peg_robot_contact = insertive_robot_contact


def set_raw_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: Controller,
    raw_state: np.ndarray,
) -> None:
    raw_state = np.asarray(raw_state, dtype=np.float64)
    mujoco.mj_resetData(model, data)
    data.qpos[:9] = raw_state[:9]
    data.qvel[:9] = raw_state[9:18]
    insertive_pos, insertive_quat = pose_in_robot_root(
        raw_state, slice(31, 34), slice(34, 38)
    )
    qpos = controller.insertive_qpos
    dof = controller.insertive_dof
    data.qpos[qpos : qpos + 3] = insertive_pos
    data.qpos[qpos + 3 : qpos + 7] = insertive_quat
    root_quat_inv = Q.quat_inv(raw_state[21:25])
    data.qvel[dof : dof + 3] = Q.quat_apply(
        root_quat_inv, raw_state[38:41] - raw_state[25:28]
    )
    data.qvel[dof + 3 : dof + 6] = Q.quat_apply(
        root_quat_inv, raw_state[41:44] - raw_state[28:31]
    )
    mujoco.mj_forward(model, data)


def _euler_xy_from_quat(quaternion: np.ndarray) -> tuple[float, float]:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return float(roll), float(pitch)


def raw_pose_success_mask(
    raw_states: np.ndarray,
    position_threshold_m: float = SUCCESS_POSITION_M,
    orientation_xy_threshold_rad: float = SUCCESS_ORIENTATION_XY_RAD,
) -> np.ndarray:
    """Evaluate an Isaac-style pose threshold directly on exported raw states."""
    raw_states = np.asarray(raw_states, dtype=np.float64)
    if raw_states.ndim != 2 or raw_states.shape[1] != 57:
        raise ValueError(f"expected raw states with shape (N, 57), got {raw_states.shape}")
    cupcake_pos, cupcake_quat = Q.combine_frame_transforms(
        raw_states[:, 31:34],
        raw_states[:, 34:38],
        CUPCAKE_ASSEMBLED_POS,
        CUPCAKE_ASSEMBLED_QUAT,
    )
    plate_pos, plate_quat = Q.combine_frame_transforms(
        raw_states[:, 44:47],
        raw_states[:, 47:51],
        PLATE_ASSEMBLED_POS,
        PLATE_ASSEMBLED_QUAT,
    )
    relative_pos, relative_quat = Q.subtract_frame_transforms(
        plate_pos, plate_quat, cupcake_pos, cupcake_quat
    )
    w, x, y, z = np.moveaxis(relative_quat, -1, 0)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return (
        (np.linalg.norm(relative_pos, axis=1) < position_threshold_m)
        & (np.abs(roll) + np.abs(pitch) < orientation_xy_threshold_rad)
    )


def success_metrics(data: mujoco.MjData, controller: Controller) -> dict[str, float | bool]:
    cupcake_pos, cupcake_quat = Q.combine_frame_transforms(
        data.xpos[controller.insertive],
        data.xquat[controller.insertive],
        CUPCAKE_ASSEMBLED_POS,
        CUPCAKE_ASSEMBLED_QUAT,
    )
    plate_pos, plate_quat = Q.combine_frame_transforms(
        data.xpos[controller.receptive],
        data.xquat[controller.receptive],
        PLATE_ASSEMBLED_POS,
        PLATE_ASSEMBLED_QUAT,
    )
    relative_pos, relative_quat = Q.subtract_frame_transforms(
        plate_pos, plate_quat, cupcake_pos, cupcake_quat
    )
    roll, pitch = _euler_xy_from_quat(relative_quat)
    position_error = float(np.linalg.norm(relative_pos))
    orientation_error = float(abs(roll) + abs(pitch))
    return {
        "position_error_m": position_error,
        "orientation_xy_error_rad": orientation_error,
        "strict_success": bool(
            position_error < SUCCESS_POSITION_M
            and orientation_error < SUCCESS_ORIENTATION_XY_RAD
        ),
        "relaxed_success": bool(
            position_error < RELAXED_SUCCESS_POSITION_M
            and orientation_error < RELAXED_SUCCESS_ORIENTATION_XY_RAD
        ),
    }
