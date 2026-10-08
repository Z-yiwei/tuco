#!/usr/bin/env python3
"""Generate RGB demonstrations by executing a frozen expert from frozen resets.

This is physics rollout with rendering, not bitwise replay of recorded trajectories.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from artifacts import sha256, verify
from runtime_env import ROOT, RUNTIME, environment

PROFILES = {
    'peg': dict(expert='experts/sim2real/peg/model_4800.pt',
        pair='Peg__PegHole',reset='PegT0HomeQCurr_xy5cmTotal_qpm0150_base0_y180h045_n4096_s20260902',
        objects=('peg','peghole_big'),seconds=32,image_size=84,seed=20260920,hand=(1000,14,60)),
    'stackcube': dict(expert='experts/sim2real/stackcube/model_4340.pt',
        pair='InsertiveCube__ReceptiveCube',reset='stackcube_xy5_t0_candidate4096_homejitter02_fr3_20260827',
        objects=('cube','cube'),seconds=20,image_size=224,seed=20260827,hand=(5000,50,400)),
}

def build(task, artifacts, output, python, gpu, num_envs, limit, repeats, dry_run, state_ids=None, variant=0):
    if task not in PROFILES:
        raise ValueError('CupCake sim2real requires a matching expert and selected-state panel. See README.md. No other expert is substituted.')
    cfg=PROFILES[task];artifacts=artifacts.resolve();output=output.resolve()
    reset=f"datasets/OmniReset/Resets/{cfg['pair']}/resets_{cfg['reset']}.pt"
    panel=f'panels/{task}/selected_1200.json'
    manifest=json.loads((ROOT/'configs/artifacts.json').read_text())
    needed={cfg['expert'],reset,panel}
    rows=[x for x in manifest['files'] if x['path'] in needed]
    if {x['path'] for x in rows}!=needed: raise ValueError('Artifact manifest is incomplete')
    verify(artifacts,rows)
    ids=json.loads((artifacts/panel).read_text())
    if len(ids)!=1200 or len(set(ids))!=1200 or any(type(i)!=int or not 0<=i<4096 for i in ids):
        raise ValueError('Invalid frozen physical-state panel')
    if variant:
        if task!='stackcube' or state_ids is None or limit:
            raise ValueError('Top-ups require stackcube, --state-ids and no integration subset')
        selected=json.loads(state_ids.read_text())
        if len(selected)!=600 or len(set(selected))!=600 or not set(selected).issubset(ids):
            raise ValueError('Top-up panel must contain 600 unique IDs from the frozen candidate panel')
        ids=selected; repeats=1
    if limit: ids=ids[:limit]
    if repeats!=5 and not limit and not variant: raise ValueError('Reduced repeats require --integration-states')
    if output.exists(): raise FileExistsError(output)
    env=environment(artifacts)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),VIS_FULL_RENDER='1',OMNIRESET_CAMERA_SETUP='real',
        OMNIRESET_REAL_BASE_CAMERAS='1',OMNIRESET_REAL_CAMERA_PROFILE='axis_3cam_side16x9_wristrear_handy_mirror_20260826',
        OMNIRESET_CAMERA_RENDER_PROFILE='policy_fast84_v1' if task=='peg' else 'native',
        OMNIRESET_EXACT_CAMERA_INTRINSICS='1',OMNIRESET_EXACT_WRIST_VERTICAL_FLIP='0',
        OMNIRESET_REAL_CAMERA_POSITION_JITTER_M='0.020',OMNIRESET_REAL_CAMERA_ROTATION_JITTER_DEG='10.0',
        OMNIRESET_REAL_OBJECT_TABLE_Z_OFFSET='0',OMNIRESET_REAL_TABLE_COVER_COLLISION='0',
        OMNIRESET_REAL_RECEPTIVE_OBJECT_Z_LIFT='0',OMNIRESET_FR3_VISUAL_PROFILE='white_arm_base_ring_black_finger_pads',
        TEACHER_SHA256=sha256(artifacts/cfg['expert']),RESET_ARTIFACT_SHA256=sha256(artifacts/reset))
    for camera in ('FRONT','SIDE','WRIST'):
        env[f'OMNIRESET_REAL_{camera}_CAMERA_POSITION_JITTER_M']='0.020' if camera!='WRIST' else ('0.010' if task=='peg' else '0.030')
        env[f'OMNIRESET_REAL_{camera}_CAMERA_ROTATION_JITTER_DEG']='15.0' if task=='stackcube' and camera=='WRIST' else '10.0'
    state_file=output/'state_ids.json'
    if task=='peg': env['OMNIRESET_SAMPLE_HDRI_CONFIG']=str(output/'hdri.yaml')
    command=[python,'-u',str(RUNTIME/'scripts/franka_kl_distill/collect_vision_kl.py'),
        '--task','OmniReset-FrankaResearch3MimicFingertip-RelCartesianOSC-RGB-Play-v0',
        '--teacher_ckpt',str(artifacts/cfg['expert']),'--joint_target_bridge','--stochastic_teacher_actions',
        '--skip_joint_home_check','--joint_ik_damping','0.05','--joint_ik_step_scale','1.0',
        '--joint_simulation_jacobian_point','link_origin','--joint_max_velocity','0.2',
        '--joint_position_stiffness','80','--joint_position_damping','4',
        '--joint_episode_length_s',str(cfg['seconds']),'--camera_setup','real',
        '--reset_types',cfg['reset'],'--reset_state_indices_file',str(state_file),
        '--num_envs',str(num_envs),'--num_demos',str(len(ids)*repeats),
        '--image_size',str(cfg['image_size']),'--post_reset_warmup_steps','1','--seed',str(cfg['seed'] + variant*1000),
        '--max_steps','500000','--diagnostic_done_metrics','--successful_repeats_per_state',str(repeats),
        '--record_placement_metrics','--output',str(output/'chunks/states.zarr' if variant else output/'states.zarr'),'--headless','--device','cuda:0']
    if task=='peg': command+=['--b3_nominal_fixed_scene','--zarr_compressor','default']
    else: command+=['--stackcube_fixed_scene','--preserve_object_face_materials','--success_position_override_m','0.020','--success_orientation_override_deg','3.0']
    for prop,value in zip(('stiffness','damping','effort_limit_sim'),cfg['hand']):
        command.append(f'env.scene.robot.actuators.panda_hand.{prop}={value}.0')
    command += [f'env.scene.insertive_object={cfg["objects"][0]}',f'env.scene.receptive_object={cfg["objects"][1]}',
        f'env.events.reset_from_reset_states.params.dataset_dir={artifacts}/datasets/OmniReset']
    if task=='peg': command+=['env.events.reset_from_reset_states.params.rigid_object_position_offsets={insertive_object:[0.0,0.0,0.013],receptive_object:[0.0,0.0,0.013]}']
    receipt={'task':task,'kind':'integration_check' if limit else 'expert_data_generation','physics_rollout':True,
        'bitwise_historical_replay':False,'states':len(ids),'repeats':repeats,'topup_variant':variant,'inputs':rows,'command':command,
        'environment':{k:v for k,v in env.items() if k not in os.environ or os.environ[k]!=v}}
    print(json.dumps(receipt,indent=2))
    if dry_run: return
    output.mkdir(parents=True)
    if variant: (output/'chunks').mkdir()
    state_file.write_text(json.dumps(ids)+'\n')
    if task=='peg':
        hdri=sorted((artifacts/'hdri/peg').glob('*.hdr'))
        if len(hdri)!=10: raise ValueError('Expected frozen ten-image Peg HDRI pool')
        (output/'hdri.yaml').write_text('local:\n'+''.join('  - '+json.dumps(str(p))+'\n' for p in hdri))
    (output/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with (output/'collect.log').open('w') as log:
        result=subprocess.run(command,cwd=RUNTIME,env=env,stdout=log,stderr=subprocess.STDOUT)
    (output/'completion.json').write_text(json.dumps({'returncode':result.returncode,'kind':receipt['kind']})+'\n')
    if result.returncode: raise SystemExit(f'Collection failed; see {output}/collect.log')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True,choices=['peg','stackcube','cupcake'])
    p.add_argument('--artifacts',type=Path,default=ROOT/'artifacts')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--python',default=sys.executable)
    p.add_argument('--gpu',default='0');p.add_argument('--num-envs',type=int,default=4)
    p.add_argument('--integration-states',type=int,default=0)
    p.add_argument('--state-ids',type=Path)
    p.add_argument('--topup-variant',type=int,choices=range(6,11),default=0)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if not 0<=a.integration_states<=1200 or a.repeats<1 or a.num_envs<1:p.error('Invalid counts')
    build(a.task,a.artifacts,a.output,a.python,a.gpu,a.num_envs,a.integration_states,a.repeats,a.dry_run,a.state_ids,a.topup_variant)
if __name__=='__main__':main()
