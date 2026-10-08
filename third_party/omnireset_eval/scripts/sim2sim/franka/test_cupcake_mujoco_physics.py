#!/usr/bin/env python3
"""Small deterministic physics regressions for the CupCake MuJoCo scene."""

from __future__ import annotations

import math
import os
import sys
import unittest
from collections import Counter
import json
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
SOURCE_ZARR = ROOT / "datasets/cupcake_sim2sim_t0_20260815/isaac_fit_b3center.zarr"


def _mesh_topology(path: Path, weld_tolerance: float = 1.0e-8) -> dict:
    vertices = []
    faces = []
    with path.open(encoding="ascii") as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v":
                vertices.append([float(value) for value in fields[1:4]])
            elif fields[0] == "f":
                faces.append([int(value.split("/")[0]) - 1 for value in fields[1:4]])
    vertices_array = np.asarray(vertices, dtype=np.float64)
    faces_array = np.asarray(faces, dtype=np.int64)
    quantized = np.round(vertices_array / weld_tolerance).astype(np.int64)
    _, welded_ids = np.unique(quantized, axis=0, return_inverse=True)
    edge_counts: Counter[tuple[int, int]] = Counter()
    for face in welded_ids[faces_array]:
        for first, second in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge_counts[tuple(sorted((int(first), int(second))))] += 1
    return {
        "vertices": vertices_array,
        "face_count": len(faces_array),
        "boundary_edges": sum(count == 1 for count in edge_counts.values()),
        "nonmanifold_edges": sum(count > 2 for count in edge_counts.values()),
    }


def _x_rotation(degrees: float) -> np.ndarray:
    half_angle = math.radians(degrees) / 2.0
    return np.array([math.cos(half_angle), math.sin(half_angle), 0.0, 0.0])


def _rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def _tilt_degrees(quaternion: np.ndarray) -> float:
    local_z_world = _rotation_matrix(quaternion)[:, 2]
    return math.degrees(math.acos(np.clip(local_z_world[2], -1.0, 1.0)))


def _build_hand_sweep_model(active_collision: bool) -> tuple[mujoco.MjModel, int, int]:
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.001
    spec.option.gravity = [0.0, 0.0, 0.0]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 150
    spec.add_mesh(name="hand_probe", file=str(CupCake.SOURCE_HAND_MESH))
    spec.add_mesh(name="cupcake_probe_visual", file=str(CupCake.CUPCAKE_MESH))
    pieces = CupCake.convex_decomposition_piece_records()
    for piece in pieces:
        spec.add_mesh(name=piece["name"], file=piece["path"])

    spec.worldbody.add_geom(
        name="hand_probe",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="hand_probe",
        contype=int(active_collision),
        conaffinity=int(active_collision),
        friction=[0.5, 0.01, 0.001],
        condim=3,
        solref=CupCake.ROBOT_OBJECT_CONTACT_SOLREF,
        rgba=[0.65, 0.65, 0.70, 1.0],
    )
    body = spec.worldbody.add_body(name="cupcake_probe", pos=[-0.14, 0.0, -0.03])
    body.add_freejoint(name="cupcake_probe_freejoint")
    body.add_geom(
        name="cupcake_probe_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="cupcake_probe_visual",
        mass=0.1268,
        contype=0,
        conaffinity=0,
        rgba=[0.80, 0.30, 0.12, 1.0],
    )
    for index, piece in enumerate(pieces):
        geom_name = f"cupcake_probe_collision_{index:02d}"
        body.add_geom(
            name=geom_name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=piece["name"],
            mass=0.0,
            friction=[0.5, 0.01, 0.001],
            condim=3,
            solref=CupCake.ROBOT_OBJECT_CONTACT_SOLREF,
            rgba=[0.0, 0.0, 0.0, 0.0],
        )
        if active_collision:
            spec.add_pair(
                geomname1=geom_name,
                geomname2="hand_probe",
                condim=3,
                friction=[0.5, 0.5, 0.0, 0.0, 0.0],
                solref=CupCake.ROBOT_OBJECT_CONTACT_SOLREF,
            )
    model = spec.compile()
    joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "cupcake_probe_freejoint"
    )
    return model, int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])


def _run_pointed_top_probe() -> dict:
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.001
    spec.option.gravity = [0.0, 0.0, 0.0]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 150
    spec.add_mesh(name="cupcake_tip_visual", file=str(CupCake.CUPCAKE_MESH))
    pieces = CupCake.convex_decomposition_piece_records()
    for piece in pieces:
        spec.add_mesh(name=piece["name"], file=piece["path"])
    spec.worldbody.add_geom(
        name="tip_stop",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.004915, -0.002025, 0.125],
        size=[0.006, 0.006, 0.002],
        contype=0,
        conaffinity=0,
        solref=CupCake.ROBOT_OBJECT_CONTACT_SOLREF,
    )
    body = spec.worldbody.add_body(name="cupcake_tip_probe")
    body.add_freejoint(name="cupcake_tip_probe_freejoint")
    body.add_geom(
        name="cupcake_tip_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="cupcake_tip_visual",
        mass=0.1268,
        contype=0,
        conaffinity=0,
    )
    for index, piece in enumerate(pieces):
        geom_name = f"cupcake_tip_collision_{index:03d}"
        body.add_geom(
            name=geom_name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=piece["name"],
            mass=0.0,
            contype=0,
            conaffinity=0,
            condim=3,
            solref=CupCake.ROBOT_OBJECT_CONTACT_SOLREF,
        )
        spec.add_pair(
            geomname1=geom_name,
            geomname2="tip_stop",
            condim=3,
            friction=[0.5, 0.5, 0.0, 0.0, 0.0],
            solref=CupCake.ROBOT_OBJECT_CONTACT_SOLREF,
        )
    model = spec.compile()
    data = mujoco.MjData(model)
    joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "cupcake_tip_probe_freejoint"
    )
    qpos = int(model.jnt_qposadr[joint])
    dof = int(model.jnt_dofadr[joint])
    data.qvel[dof + 2] = 0.15
    mujoco.mj_forward(model, data)
    first_contact_z = None
    minimum_distance = math.inf
    for _ in range(500):
        mujoco.mj_step(model, data)
        if data.ncon:
            if first_contact_z is None:
                first_contact_z = float(data.qpos[qpos + 2])
            minimum_distance = min(
                minimum_distance,
                min(float(contact.dist) for contact in data.contact[: data.ncon]),
            )
    return {
        "first_contact_z": first_contact_z,
        "final_origin_z": float(data.qpos[qpos + 2]),
        "maximum_penetration": max(0.0, -minimum_distance),
    }


class CupCakePhysicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not SOURCE_ZARR.is_dir():
            raise unittest.SkipTest(f"missing rich replay fixture: {SOURCE_ZARR}")
        cls.store = zarr.open(str(SOURCE_ZARR), mode="r")
        Replay.configure_timing(cls.store)
        cls.raw = np.asarray(cls.store["data/raw_state"][0], dtype=np.float64)
        cls.episode_end = int(np.asarray(cls.store["meta/episode_ends"])[0])
        cls.mesh_vertices = _mesh_topology(CupCake.CUPCAKE_MESH)["vertices"]

    def make_context(self):
        profile = B3.b3_center_profile(physics_substeps=4)
        profile.update(
            cupcake_collision="convex_decomposition",
            hand_collision="source_usd",
            finger_collision="mimic",
            noslip_iterations=0,
            plate_collision_enabled=False,
        )
        model = CupCake.build_model(self.raw, profile_cfg=profile)
        controller = CupCake.Controller(model)
        data = mujoco.MjData(model)
        Replay.configure_episode(
            model,
            controller,
            self.store,
            0,
            self.episode_end,
            self.raw,
            friction_combine="physx_average",
        )
        return model, data, controller

    def test_source_mesh_audit_explains_why_sdf_is_not_production_default(self):
        topology = _mesh_topology(CupCake.CUPCAKE_MESH)
        self.assertEqual(topology["nonmanifold_edges"], 0)
        self.assertEqual(topology["boundary_edges"], 3)

        profile = B3.b3_center_profile(physics_substeps=4)
        model = CupCake.build_model(self.raw, profile_cfg=profile)
        active_names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            for geom in range(model.ngeom)
            if (model.geom_contype[geom] or model.geom_conaffinity[geom])
            and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        }
        self.assertIn("cupcake_collision_convex_piece_00", active_names)
        manifest = json.loads(
            CupCake.CONVEX_DECOMPOSITION_MANIFEST.read_text(encoding="ascii")
        )
        self.assertEqual(
            len(
                [
                    name
                    for name in active_names
                    if name.startswith("cupcake_collision_convex_piece_")
                ]
            ),
            manifest["piece_count"],
        )
        self.assertNotIn("cupcake_collision", active_names)

    def test_visual_mesh_keeps_usd_shading_data_and_upright_axis(self):
        collision = _mesh_topology(CupCake.CUPCAKE_MESH)
        visual = _mesh_topology(CupCake.CUPCAKE_VISUAL_MESH)
        np.testing.assert_allclose(
            visual["vertices"].min(axis=0),
            collision["vertices"].min(axis=0),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            visual["vertices"].max(axis=0),
            collision["vertices"].max(axis=0),
            atol=1.0e-12,
        )
        with CupCake.CUPCAKE_VISUAL_MESH.open(encoding="ascii") as stream:
            prefixes = Counter(line.split(maxsplit=1)[0] for line in stream if line.strip())
        self.assertEqual(prefixes["vn"], 4005)
        self.assertEqual(prefixes["vt"], 4005)

        profile = B3.b3_center_profile(physics_substeps=4)
        model = CupCake.build_model(self.raw, profile_cfg=profile)
        visual_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "cupcake_visual"
        )
        cupcake_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "cupcake"
        )
        self.assertEqual(int(model.geom_bodyid[visual_geom]), cupcake_body)
        self.assertEqual(int(model.geom_contype[visual_geom]), 0)
        self.assertEqual(int(model.geom_conaffinity[visual_geom]), 0)
        visual_up = _rotation_matrix(model.geom_quat[visual_geom])[:, 2]
        self.assertLess(float(np.linalg.norm(visual_up[:2])), 2.0e-3)
        self.assertGreater(float(visual_up[2]), 0.99999)

        Replay.configure_episode(
            model,
            CupCake.Controller(model),
            self.store,
            0,
            self.episode_end,
            self.raw,
            friction_combine="physx_average",
        )
        self.assertEqual(
            int(model.geom_sameframe[visual_geom]),
            int(mujoco.mjtSameFrame.mjSAMEFRAME_NONE),
        )

    def test_fine_convex_decomposition_covers_the_pointed_top(self):
        manifest = json.loads(
            CupCake.CONVEX_DECOMPOSITION_MANIFEST.read_text(encoding="ascii")
        )
        coacd_pieces = [
            piece for piece in manifest["pieces"] if piece.get("kind") == "coacd"
        ]
        support_pieces = [
            piece
            for piece in manifest["pieces"]
            if piece.get("kind") == "support_pad"
        ]
        self.assertGreaterEqual(len(coacd_pieces), 100)
        self.assertGreaterEqual(manifest["piece_face_count"], 7000)
        self.assertLess(abs(manifest["top_z_error_m"]), 1.0e-6)
        self.assertEqual(len(support_pieces), 1)
        self.assertLessEqual(support_pieces[0]["bounds_max_m"][2], 0.0011)

        outcome = _run_pointed_top_probe()
        self.assertIsNotNone(outcome["first_contact_z"], outcome)
        self.assertLess(outcome["first_contact_z"], 0.027, outcome)
        self.assertLess(outcome["maximum_penetration"], 0.0007, outcome)

    def test_legacy_plate_contact_can_use_one_full_source_convex_hull(self):
        profile = B3.b3_center_profile(physics_substeps=4)
        profile.update(
            cupcake_collision="convex_decomposition",
            cupcake_plate_collision="convex_hull",
            hand_collision="source_usd",
            finger_collision="mimic",
            plate_collision_enabled=True,
        )
        model = CupCake.build_model(self.raw, profile_cfg=profile)
        hull = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "cupcake_collision_plate_hull"
        )
        self.assertGreaterEqual(hull, 0)
        self.assertEqual(int(model.geom_type[hull]), mujoco.mjtGeom.mjGEOM_MESH)
        self.assertFalse(
            any(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom).startswith(
                    "cupcake_collision_plate_piece_"
                )
                for geom in range(model.ngeom)
                if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            )
        )
        plate = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "plate_collision"
        )
        matching_pairs = {
            frozenset((int(model.pair_geom1[pair]), int(model.pair_geom2[pair])))
            for pair in range(model.npair)
        }
        self.assertIn(frozenset((hull, plate)), matching_pairs)

    def test_production_plate_collision_is_convex_decomposition_without_sdf(self):
        profile = B3.b3_center_profile(physics_substeps=4)
        profile.update(
            cupcake_collision="convex_decomposition",
            plate_collision_enabled=True,
        )
        model = CupCake.build_model(self.raw, profile_cfg=profile)
        manifest = json.loads(
            CupCake.PLATE_COACD_MANIFEST.read_text(encoding="ascii")
        )
        plate_geoms = [
            geom
            for geom in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "").startswith(
                "plate_collision"
            )
        ]
        self.assertEqual(len(plate_geoms), manifest["piece_count"])
        self.assertGreaterEqual(manifest["piece_face_count"], 3000)
        self.assertFalse(
            any(
                int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_SDF)
                for geom in range(model.ngeom)
            )
        )
        support = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "cupcake_collision_plate_base",
        )
        self.assertGreaterEqual(support, 0)
        self.assertEqual(
            int(model.geom_type[support]), int(mujoco.mjtGeom.mjGEOM_CYLINDER)
        )
        np.testing.assert_allclose(model.geom_size[support, :2], [0.039, 0.025])

        detailed = [
            geom
            for geom in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "").startswith(
                "cupcake_collision_convex_piece_"
            )
        ]
        self.assertEqual(
            len(detailed),
            json.loads(
                CupCake.CONVEX_DECOMPOSITION_MANIFEST.read_text(encoding="ascii")
            )["piece_count"],
        )

    def test_source_style_visuals_and_target_ring_are_collision_free(self):
        profile = B3.b3_center_profile(physics_substeps=4)
        model = CupCake.build_model(self.raw, profile_cfg=profile)
        expected = {
            "cupcake_visual": [0.55, 0.12, 0.05, 1.0],
            "plate_visual": [0.92, 0.92, 0.92, 1.0],
            "plate_target_ring_outer": [0.95, 0.05, 0.05, 1.0],
        }
        for name, rgba in expected.items():
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.assertGreaterEqual(geom, 0)
            self.assertEqual(int(model.geom_contype[geom]), 0)
            self.assertEqual(int(model.geom_conaffinity[geom]), 0)
            np.testing.assert_allclose(model.geom_rgba[geom], rgba, atol=1.0e-6)
        outer = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "plate_target_ring_outer"
        )
        inner = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "plate_target_ring_inner_mask"
        )
        self.assertAlmostEqual(model.geom_size[outer, 0], 0.041327, places=7)
        self.assertAlmostEqual(model.geom_size[inner, 0], 0.039327, places=7)

    def test_relaxed_success_uses_isaac_metric_with_looser_thresholds(self):
        strict = CupCake.raw_pose_success_mask(self.raw[None])
        relaxed = CupCake.raw_pose_success_mask(
            self.raw[None],
            position_threshold_m=CupCake.RELAXED_SUCCESS_POSITION_M,
            orientation_xy_threshold_rad=(
                CupCake.RELAXED_SUCCESS_ORIENTATION_XY_RAD
            ),
        )
        self.assertFalse(bool(strict[0]))
        self.assertFalse(bool(relaxed[0]))

        model, data, controller = self.make_context()
        CupCake.set_raw_state(model, data, controller, self.raw)
        plate_pos = data.xpos[controller.receptive].copy()
        plate_quat = data.xquat[controller.receptive].copy()
        qpos = controller.insertive_qpos
        data.qpos[qpos : qpos + 3] = plate_pos + [0.010, 0.0, 0.003]
        data.qpos[qpos + 3 : qpos + 7] = plate_quat
        mujoco.mj_forward(model, data)
        metrics = CupCake.success_metrics(data, controller)
        self.assertFalse(metrics["strict_success"], metrics)
        self.assertTrue(metrics["relaxed_success"], metrics)

    def test_plate_convex_decomposition_catches_and_settles_center_drop(self):
        profile = B3.b3_center_profile(physics_substeps=4)
        profile.update(
            cupcake_collision="convex_decomposition",
            plate_collision_enabled=True,
        )
        model = CupCake.build_model(self.raw, profile_cfg=profile)
        controller = CupCake.Controller(model)
        data = mujoco.MjData(model)
        CupCake.set_raw_state(model, data, controller, self.raw)
        plate_pos = data.xpos[controller.receptive].copy()
        qpos = controller.insertive_qpos
        dof = controller.insertive_dof
        data.qpos[qpos : qpos + 3] = plate_pos + [0.0, 0.0, 0.12]
        data.qpos[qpos + 3 : qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(model, data)

        final_window = []
        for step in range(int(2.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            if step >= int(1.5 / model.opt.timestep):
                final_window.append(data.xpos[controller.insertive, 2] - plate_pos[2])
        self.assertGreater(final_window[-1], 0.0)
        self.assertLess(np.ptp(final_window), 0.001)
        self.assertGreater(data.ncon, 0)
        metrics = CupCake.success_metrics(data, controller)
        self.assertTrue(metrics["strict_success"], metrics)
        self.assertTrue(metrics["relaxed_success"], metrics)

    def test_production_contacts_do_not_add_rolling_constraints(self):
        model, _, _ = self.make_context()
        self.assertEqual(model.opt.noslip_iterations, 0)
        checked = 0
        for pair in range(model.npair):
            names = {
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom1[pair])
                ),
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom2[pair])
                ),
            }
            if not any(name and name.startswith("cupcake_collision") for name in names):
                continue
            checked += 1
            self.assertEqual(model.pair_dim[pair], 3)
            np.testing.assert_allclose(model.pair_friction[pair, 2:], 0.0)
        self.assertGreater(checked, 0)

    def test_contact_offsets_do_not_inflate_the_rest_surface(self):
        model, _, controller = self.make_context()
        checked = 0
        for pair in range(model.npair):
            bodies = {
                int(model.geom_bodyid[int(model.pair_geom1[pair])]),
                int(model.geom_bodyid[int(model.pair_geom2[pair])]),
            }
            if controller.insertive not in bodies:
                continue
            checked += 1
            self.assertGreater(model.pair_margin[pair], 0.0)
            self.assertAlmostEqual(
                model.pair_margin[pair] - model.pair_gap[pair], 0.0, places=12
            )
        self.assertGreater(checked, 0)

    def test_recorded_mass_com_and_inertia_are_applied(self):
        model, _, controller = self.make_context()
        body = controller.insertive
        source_mass = float(
            np.asarray(self.store["data/insertive_object_body_masses"][0]).reshape(-1)[0]
        )
        source_com = np.asarray(
            self.store["data/insertive_object_body_coms"][0], dtype=np.float64
        )[:3]
        source_tensor = np.asarray(
            self.store["data/insertive_object_body_inertias"][0], dtype=np.float64
        ).reshape(3, 3, order="F")
        source_principal = np.linalg.eigvalsh(0.5 * (source_tensor + source_tensor.T))

        self.assertAlmostEqual(model.body_mass[body], source_mass, places=9)
        np.testing.assert_allclose(model.body_ipos[body], source_com, atol=1.0e-9)
        np.testing.assert_allclose(
            np.sort(model.body_inertia[body]), source_principal, atol=1.0e-10
        )

    def test_collision_free_drop_obeys_gravity(self):
        model, data, controller = self.make_context()
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        mujoco.mj_resetData(model, data)
        qpos = controller.insertive_qpos
        dof = controller.insertive_dof
        data.qpos[qpos : qpos + 3] = [0.0, 0.0, 0.5]
        data.qpos[qpos + 3 : qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        duration = 0.2
        steps = int(round(duration / model.opt.timestep))
        for _ in range(steps):
            mujoco.mj_step(model, data)
        elapsed = steps * model.opt.timestep
        gravity = float(model.opt.gravity[2])
        expected_velocity = gravity * elapsed
        expected_height = 0.5 + 0.5 * gravity * elapsed**2
        self.assertAlmostEqual(data.qvel[dof + 2], expected_velocity, delta=0.03)
        self.assertAlmostEqual(data.qpos[qpos + 2], expected_height, delta=0.003)

    def test_hand_and_fingers_have_active_collision_geometry(self):
        model, data, controller = self.make_context()
        CupCake.set_raw_state(model, data, controller, self.raw)
        for body_name in ("hand", "left_finger", "right_finger"):
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            active = [
                geom
                for geom in range(model.ngeom)
                if int(model.geom_bodyid[geom]) == body
                and (model.geom_contype[geom] or model.geom_conaffinity[geom])
            ]
            self.assertTrue(active, f"{body_name} has no active collision geom")

        qpos = controller.insertive_qpos
        data.qpos[qpos : qpos + 3] = data.xpos[controller.hand]
        data.qpos[qpos + 3 : qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[controller.insertive_dof : controller.insertive_dof + 6] = 0.0
        mujoco.mj_forward(model, data)
        contacted_bodies = controller.insertive_robot_contact_bodies(data)
        self.assertIn("hand", contacted_bodies)

    def test_hand_collision_blocks_swept_cupcake(self):
        outcomes = {}
        for active_collision in (False, True):
            model, qpos, dof = _build_hand_sweep_model(active_collision)
            data = mujoco.MjData(model)
            data.qvel[dof] = 0.35
            mujoco.mj_forward(model, data)
            maximum_x = float(data.qpos[qpos])
            minimum_distance = math.inf
            contact_steps = 0
            for _ in range(1000):
                mujoco.mj_step(model, data)
                maximum_x = max(maximum_x, float(data.qpos[qpos]))
                if data.ncon:
                    contact_steps += 1
                    minimum_distance = min(
                        minimum_distance,
                        min(float(contact.dist) for contact in data.contact[: data.ncon]),
                    )
            outcomes[active_collision] = {
                "maximum_x": maximum_x,
                "contact_steps": contact_steps,
                "maximum_penetration": max(0.0, -minimum_distance),
            }

        self.assertGreater(outcomes[False]["maximum_x"], 0.15, outcomes)
        self.assertEqual(outcomes[False]["contact_steps"], 0, outcomes)
        self.assertLess(outcomes[True]["maximum_x"], -0.05, outcomes)
        self.assertGreater(outcomes[True]["contact_steps"], 0, outcomes)
        self.assertLess(outcomes[True]["maximum_penetration"], 0.0007, outcomes)

    def test_tilted_release_crosses_tipping_boundary_and_settles(self):
        model, data, controller = self.make_context()
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom])
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            if body in controller.robot_body_ids or geom_name == "floor":
                model.geom_contype[geom] = 0
                model.geom_conaffinity[geom] = 0
        mujoco.mj_forward(model, data)
        table_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "table_top"
        )
        table_xy = data.geom_xpos[table_geom, :2].copy()
        table_top = float(data.geom_xpos[table_geom, 2] + model.geom_size[table_geom, 2])

        results = {}
        for initial_tilt in (20.0, 50.0, 60.0):
            mujoco.mj_resetData(model, data)
            quaternion = _x_rotation(initial_tilt)
            rotation = _rotation_matrix(quaternion)
            mesh_min_z = float((self.mesh_vertices @ rotation.T)[:, 2].min())
            qpos = controller.insertive_qpos
            data.qpos[qpos : qpos + 3] = [
                table_xy[0],
                table_xy[1],
                table_top - mesh_min_z + 0.0005,
            ]
            data.qpos[qpos + 3 : qpos + 7] = quaternion
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            for _ in range(int(round(5.0 / model.opt.timestep))):
                mujoco.mj_step(model, data)
            results[initial_tilt] = {
                "final_tilt": _tilt_degrees(data.qpos[qpos + 3 : qpos + 7]),
                "final_speed": float(
                    np.linalg.norm(
                        data.qvel[
                            controller.insertive_dof : controller.insertive_dof + 6
                        ]
                    )
                ),
            }

        self.assertLess(results[20.0]["final_tilt"], 5.0, results)
        for initial_tilt in (50.0, 60.0):
            self.assertGreater(results[initial_tilt]["final_tilt"], 70.0, results)
            self.assertGreater(
                abs(results[initial_tilt]["final_tilt"] - initial_tilt),
                10.0,
                results,
            )
            self.assertLess(results[initial_tilt]["final_speed"], 0.1, results)


if __name__ == "__main__":
    unittest.main(verbosity=2)
