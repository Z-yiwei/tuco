"""Open-loop replay test: feed the EXPERT's own recorded actions (rec_actions)
through the exact eval_sr controller/physics, from each snapshot's own
initial_state. If success rate is high, the closed-loop harness is correct and
low policy SR is a weak-policy issue (not a harness/action-scale bug).
"""
import os, sys, glob, collections, numpy as np, mujoco, torch
os.environ.setdefault("MUJOCO_GL", "")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import quat_utils as Q, closed_loop_eval as CL

ROOT = os.path.dirname(HERE)
ALL = sorted(glob.glob(ROOT + "/datasets/snapshots_pegHole_upright_yaw_3cm/seed*/demo_0000.pt"),
             key=lambda p: int(p.split('seed')[1].split('/')[0]))
SEAT = np.array([0, 0, CL.ASSEMBLED_Z])

def load(f):
    d = torch.load(f, map_location="cpu", weights_only=False)
    st = d["initial_state"]
    q9 = np.asarray(st["articulation"]["robot"]["joint_position"]).flatten()
    pp = np.asarray(st["rigid_object"]["insertive_object"]["root_pose"]).flatten()
    hp = np.asarray(st["rigid_object"]["receptive_object"]["root_pose"]).flatten()
    acts = np.stack([np.asarray(x, dtype=np.float32) for x in d["rec_actions"]])
    return q9, pp, hp, acts, bool(d["collect_success"])

def replay(q9, pp, hp, acts):
    m = CL.build_model(hp[:3], hp[3:7]); ctrl = CL.Controller(m); d = mujoco.MjData(m)
    pj = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "peg")]
    d.qpos[:9] = q9; d.qpos[pj:pj+3] = pp[:3]; d.qpos[pj+3:pj+7] = pp[3:7]; mujoco.mj_forward(m, d)
    sx = ox = False
    n = min(len(acts), CL.EP_LEN)
    for t in range(n):
        act = acts[t]
        sc = act[:6] * CL.SCALE
        ee, eq = ctrl.ee_root(d); dp = ee + sc[:3]; dq = Q.quat_mul(CL.quat_from_aa(sc[3:6]), eq)
        d.ctrl[ctrl.grip_act] = 0.0 if ctrl.grasp_close(d) else 255.0
        for _ in range(CL.DECIM):
            ee, eq = ctrl.ee_root(d); J = ctrl.jac_arm(d); evv = J @ d.qvel[:7]
            pe = dp - ee; qe = Q.quat_mul(dq, Q.quat_inv(eq)); aae = Q.axis_angle_from_quat(qe)
            d.qfrc_applied[:7] = np.clip(J.T @ (CL.KP*np.concatenate([pe, aae]) + CL.KD*(-evv)), -CL.TAU_MAX, CL.TAU_MAX)
            mujoco.mj_step(m, d); np.clip(d.qvel[:7], -CL.VEL_MAX, CL.VEL_MAX, out=d.qvel[:7])
        ph = Q.pose_in_root(hp[:3], hp[3:7], d.qpos[pj:pj+3].copy(), d.qpos[pj+3:pj+7].copy())
        dd = np.linalg.norm(ph[:3]-SEAT); rr = np.linalg.norm(ph[3:])
        if dd < 0.0025 and rr < 0.025: sx = True
        if dd < 0.01 and rr < 0.1: ox = True
    return sx, ox

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    files = ALL[190:190+N]
    st = o1 = cs = 0
    for i, f in enumerate(files):
        q9, pp, hp, acts, ok = load(f)
        cs += ok
        s, o = replay(q9, pp, hp, acts)
        st += s; o1 += o
        print(f"  {os.path.basename(os.path.dirname(f))}: collect_success={ok} replay strict={s} 1cm={o}", flush=True)
    print(f"[replay] N={len(files)}  collect_success={cs}/{len(files)}  "
          f"replay strict={st}/{len(files)}  1cm={o1}/{len(files)}", flush=True)
