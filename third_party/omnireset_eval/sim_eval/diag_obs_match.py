"""Diagnose: does the MuJoCo-reconstructed obs (what the policy SEES in closed
loop) match the recorded obs the policy was TRAINED on?

For each held-out snapshot: set qpos from initial_state, run ObsBuilder to get the
t=0 reconstructed 200-D obs, and compare block-by-block against the snapshot's own
rec_obs[0]. Large mismatch => closed-loop input is OOD => explains near-zero SR
regardless of training quality.
"""
import os, sys, glob, numpy as np, mujoco, torch
os.environ.setdefault("MUJOCO_GL", "")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import quat_utils as Q, closed_loop_eval as CL

ROOT = os.path.dirname(HERE)
ALL = sorted(glob.glob(ROOT + "/datasets/snapshots_pegHole_upright_yaw_3cm/seed*/demo_0000.pt"),
             key=lambda p: int(p.split('seed')[1].split('/')[0]))

BLOCKS = CL.BLOCKS if hasattr(CL, "BLOCKS") else None
from closed_loop_eval import BLOCKS  # (start, width) per block, HIST=5


def rec_obs0(f):
    d = torch.load(f, map_location="cpu", weights_only=False)
    st = d["initial_state"]
    q9 = np.asarray(st["articulation"]["robot"]["joint_position"]).flatten()
    pp = np.asarray(st["rigid_object"]["insertive_object"]["root_pose"]).flatten()
    hp = np.asarray(st["rigid_object"]["receptive_object"]["root_pose"]).flatten()
    ro = np.asarray(d["rec_obs"][0], dtype=np.float32).flatten()
    return q9, pp, hp, ro


def recon0(q9, pp, hp):
    m = CL.build_model(hp[:3], hp[3:7]); ctrl = CL.Controller(m); obb = CL.ObsBuilder(ctrl)
    d = mujoco.MjData(m)
    pj = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "peg")]
    d.qpos[:9] = q9; d.qpos[pj:pj+3] = pp[:3]; d.qpos[pj+3:pj+7] = pp[3:7]
    mujoco.mj_forward(m, d)
    obb.reset(d)
    return obb.step(d, np.zeros(7, dtype=np.float32))


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    files = ALL[190:190+N]
    HIST = 5
    agg = {b: [] for b in BLOCKS}
    for f in files:
        q9, pp, hp, ro = rec_obs0(f)
        rc = recon0(q9, pp, hp)
        seed = os.path.basename(os.path.dirname(f))
        print(f"\n=== {seed}  |ro|={ro.shape} ===")
        for b, (start, w) in BLOCKS.items():
            # compare the newest (current) sub-vector of each block (index HIST-1)
            i = start + (HIST - 1) * w
            v_rec = rc[i:i+w]
            v_dat = ro[i:i+w]
            err = np.abs(v_rec - v_dat)
            agg[b].append(err.mean())
            print(f"  {b:<13} data={np.round(v_dat,3)}")
            print(f"  {'':<13} recon={np.round(v_rec,3)}  maxerr={err.max():.3f}")
    print("\n=== mean|err| per block over", len(files), "snapshots ===")
    for b in BLOCKS:
        print(f"  {b:<13} {np.mean(agg[b]):.4f}")
