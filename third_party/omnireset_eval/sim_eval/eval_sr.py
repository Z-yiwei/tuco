"""Leak-clean eval for a cotrain ckpt (standalone sim2sim_cotrain).
Held-out = peg snapshots [190:361] minus init-state collisions with expert pool [40:190].
1cm success = position<1cm AND orientation<0.1rad (real task gate).
Usage: python eval/eval_cotrain.py <ckpt_path> [ckpt2 ...]
"""
import os; os.environ.setdefault("MUJOCO_GL","")
import sys,glob,collections,numpy as np,mujoco,torch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quat_utils as Q, closed_loop_eval as CL
from mlp_util import load_mlp
SEAT=np.array([0,0,CL.ASSEMBLED_Z])
ALL=sorted(glob.glob(ROOT+"/datasets/snapshots_pegHole_upright_yaw_3cm/seed*/demo_0000.pt"),key=lambda p:int(p.split('seed')[1].split('/')[0]))
def iv(f):
    d=torch.load(f,map_location="cpu",weights_only=False)["initial_state"]
    return (np.asarray(d["articulation"]["robot"]["joint_position"]).flatten(),np.asarray(d["rigid_object"]["insertive_object"]["root_pose"]).flatten(),np.asarray(d["rigid_object"]["receptive_object"]["root_pose"]).flatten())
def kv(q,p,h): return tuple(np.round(np.concatenate([q,p,h]),6).tolist())
pk=set(kv(*iv(f)) for f in ALL[40:190]); HELD=[x for x in (iv(f) for f in ALL[190:361]) if kv(*x) not in pk]; NC=len(HELD)
def ev(path):
    model,norm=load_mlp(path); sm,ss,ac,asc,nobs=norm["s_mean"],norm["s_std"],norm["a_center"],norm["a_scale"],norm["n_obs"]; st=o1=0
    for q9,pp,hp in HELD:
        m=CL.build_model(hp[:3],hp[3:7]); ctrl=CL.Controller(m); obb=CL.ObsBuilder(ctrl); d=mujoco.MjData(m)
        pj=m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,"peg")]
        d.qpos[:9]=q9; d.qpos[pj:pj+3]=pp[:3]; d.qpos[pj+3:pj+7]=pp[3:7]; mujoco.mj_forward(m,d); obb.reset(d)
        h=collections.deque(maxlen=nobs); pa=np.zeros(7,dtype=np.float32); sx=ox=False
        for t in range(CL.EP_LEN):
            o=obb.step(d,pa)
            if not h:
                for _ in range(nobs): h.append(o)
            else: h.append(o)
            with torch.no_grad(): mu=model(torch.from_numpy(((np.stack(h)-sm)/ss)).float()[None]).numpy()[0]
            act=(mu*asc+ac).astype(np.float32); pa=act
            sc=act[:6]*CL.SCALE; ee,eq=ctrl.ee_root(d); dp=ee+sc[:3]; dq=Q.quat_mul(CL.quat_from_aa(sc[3:6]),eq)
            d.ctrl[ctrl.grip_act]=0.0 if ctrl.grasp_close(d) else 255.0
            for _ in range(CL.DECIM):
                ee,eq=ctrl.ee_root(d); J=ctrl.jac_arm(d); evv=J@d.qvel[:7]; pe=dp-ee; qe=Q.quat_mul(dq,Q.quat_inv(eq)); aae=Q.axis_angle_from_quat(qe)
                d.qfrc_applied[:7]=np.clip(J.T@(CL.KP*np.concatenate([pe,aae])+CL.KD*(-evv)),-CL.TAU_MAX,CL.TAU_MAX); mujoco.mj_step(m,d); np.clip(d.qvel[:7],-CL.VEL_MAX,CL.VEL_MAX,out=d.qvel[:7])
            ph=Q.pose_in_root(hp[:3],hp[3:7],d.qpos[pj:pj+3].copy(),d.qpos[pj+3:pj+7].copy()); dd=np.linalg.norm(ph[:3]-SEAT); rr=np.linalg.norm(ph[3:])
            if dd<0.0025 and rr<0.025: sx=True
            if dd<0.01 and rr<0.1: ox=True
        st+=sx; o1+=ox
    return st,o1
if __name__=="__main__":
    print(f"[eval] leak-clean held-out N={NC}/seed")
    for p in sys.argv[1:]:
        st,o1=ev(p); print(f"  {p.split('/checkpoints/')[-1]:<45} strict {st}/{NC}={st/NC*100:.0f}%  1cm {o1}/{NC}={o1/NC*100:.0f}%",flush=True)
