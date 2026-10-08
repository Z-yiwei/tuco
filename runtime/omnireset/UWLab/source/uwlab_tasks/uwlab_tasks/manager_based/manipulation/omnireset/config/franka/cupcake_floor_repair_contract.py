"""Opt-in floor repair, layered over (never rewriting) the sealed grasp contract."""
import hashlib
import json
import math
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_floor_repair(env, env_ids):
    path = Path(os.environ['CUPCAKE_FLOOR_CONTRACT']).resolve()
    contract = json.loads(path.read_text())
    assert contract['version'] == 'cupcake_floor_repair_v1'
    assert contract['regression_passed'] is True
    assert contract['runtime_source_sha256'] == sha(__file__)
    root = Path(os.environ['CUPCAKE_RESET_DATASET_ROOT']).resolve()
    assert contract['parent_training_contract_sha256'] == sha(root/'training_contract.json')
    for name, digest in contract['evidence_sha256'].items():
        assert sha(name) == digest, name
    floor = env.cfg.scene.fall_catcher
    assert floor.prim_path == '/World/FallCatcher' and floor.collision_group == -1
    assert tuple(floor.spawn.size) == (1000., 1000., .5)
    assert tuple(floor.init_state.pos[:2]) == (0., 0.)
    assert math.isclose(floor.init_state.pos[2], -1.118, abs_tol=1e-10)
    assert floor.spawn.collision_props.collision_enabled is True
    assert floor.spawn.physics_material.static_friction == .5
    assert floor.spawn.physics_material.dynamic_friction == .5
    assert floor.spawn.physics_material.restitution == 0.
    # Inspect the actual spawned collider, not only intended config.
    from pxr import UsdPhysics
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath('/World/FallCatcher')
    assert prim.IsValid(), 'floor was not spawned'
    from pxr import Usd
    colliders = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert colliders and any(UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() for p in colliders)
    quota = Path(os.environ['CUPCAKE_T0_TRAJECTORY_ROOT']).resolve()
    marker = json.loads((quota/'COMPLETE.json').read_text())
    assert marker['count'] == 200
    assert marker['manifest_sha256'] == sha(quota/'manifest.jsonl') == contract['sealed_trajectory_manifest_sha256']
    from .cupcake_t0_trajectory import verify_repair_with_trajectories
    verify_repair_with_trajectories(env, env_ids)
    result = dict(version=contract['version'], contract_sha256=sha(path),
                  parent_training_contract_sha256=contract['parent_training_contract_sha256'],
                  actual_collider_present=True, top_z_m=-.868, thickness_m=.5,
                  full200_remains_sealed=True)
    output = Path(env.cfg.log_dir)/f'cupcake_floor_contract_rank{os.environ.get("RANK", "0")}.json'
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print('[CUPCAKE_FLOOR_CONTRACT] '+json.dumps(result), flush=True)
