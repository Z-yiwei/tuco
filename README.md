<div align="center">

# TUCO

### Curating Simulation Demonstrations for Sim-to-Real Robot Policy Co-Training

<p>
  Ning Zhu<sup>1,*</sup> &nbsp;·&nbsp;
  Mengfei Zhao<sup>2,3,*</sup> &nbsp;·&nbsp;
  Yikai Tang<sup>4,*</sup> &nbsp;·&nbsp;
  Zhangyujie Sun<sup>5</sup><br>
  Peihao Li<sup>6</sup> &nbsp;·&nbsp;
  Dongyue Ni<sup>7</sup> &nbsp;·&nbsp;
  Jindou Jia<sup>2,†</sup> &nbsp;·&nbsp;
  Jianfei Yang<sup>2,†</sup>
</p>

<p>
  <sup>1</sup>Stanford University &nbsp;·&nbsp;
  <sup>2</sup>Nanyang Technological University &nbsp;·&nbsp;
  <sup>3</sup>AXIS Robotics<br>
  <sup>4</sup>Carnegie Mellon University &nbsp;·&nbsp;
  <sup>5</sup>Fudan University &nbsp;·&nbsp;
  <sup>6</sup>University of California, Berkeley &nbsp;·&nbsp;
  <sup>7</sup>Shanghai Jiao Tong University
</p>

<sub><sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Corresponding authors</sub>

<p>
  <a href="https://tuco-curation.github.io/">
    <img src="https://img.shields.io/badge/Project-Homepage-6A3D9A?style=for-the-badge" alt="Project homepage">
  </a>
  <a href="https://arxiv.org/abs/2610.05407">
    <img src="https://img.shields.io/badge/arXiv-2610.05407-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv paper">
  </a>
</p>

<p><strong>Official implementation of TUCO.</strong></p>

</div>

<p align="center">
  <a href="https://tuco-curation.github.io/">
    <img src="assets/tuco_main_figure.jpg" width="100%" alt="TUCO method overview">
  </a>
</p>

<p align="center"><em>Overview of TUCO.</em></p>

<p align="center">
  <a href="#environment">Environment</a> &nbsp;·&nbsp;
  <a href="#data">Data</a> &nbsp;·&nbsp;
  <a href="#experiments">Experiments</a> &nbsp;·&nbsp;
  <a href="https://tuco-curation.github.io/">Project Page</a> &nbsp;·&nbsp;
  <a href="https://arxiv.org/abs/2610.05407">Paper</a>
</p>

---

## Environment

Use Linux, Conda, and an NVIDIA GPU. Run commands from the repository root.

```bash
# Single-simulator experiments
bash scripts/setup_environment.sh cupid

# Sim-to-sim and sim-to-real experiments
bash scripts/setup_environment.sh omnireset
```

IsaacSim data generation requires IsaacSim 5.1. Environment specifications are
provided in `environments/`.

## Data

- **RoboMimic:** Download the public low-dimensional demonstrations:
  ```bash
  conda activate cupid
  bash scripts/generate_data.sh single-sim third_party/cupid/data
  ```
- **OmniReset:** Obtain public assets and pretrained policies from the
  [official resources](https://uw-lab.github.io/UWLab/main/source/publications/omnireset/index.html).
  Data-generation entry points are provided in `scripts/generate_data.sh`.
- **Real robot:** Sim-to-real experiments additionally use user-collected
  real demonstrations and scoring rollouts.

Set data and checkpoint paths in the launch configurations below.

## Experiments

Copy the relevant configuration template and edit its task, paths, and curation
budget before running.

### Single-simulator

```bash
conda activate cupid
cp configs/launch/single_sim.env.example configs/launch/single_sim.env
bash experiments/single_sim/run.sh configs/launch/single_sim.env
```

Set `SPLIT=filter` for filtering or `SPLIT=select` for selection.

### Sim-to-sim

```bash
conda activate omnireset_release
cp configs/launch/sim2sim.env.example configs/launch/sim2sim.env
bash experiments/sim2sim/run.sh configs/launch/sim2sim.env
bash experiments/sim2sim/run_eval.sh configs/launch/sim2sim.env
```

### Sim-to-real

```bash
conda activate omnireset_release
cp configs/launch/sim2real.env.example configs/launch/sim2real.env
bash experiments/sim2real/run.sh configs/launch/sim2real.env
```

This runs selection and co-training. Evaluation requires the physical robot setup.

### Baselines

Set `METHOD` and the required inputs in the corresponding configuration.

```bash
# Single-simulator
cp configs/launch/diffusion_baseline.env.example configs/launch/baseline.env
bash experiments/single_sim/run_baseline.sh configs/launch/baseline.env

# Cross-domain
bash experiments/sim2sim/run_baseline.sh configs/launch/sim2sim.env
bash experiments/sim2real/run_baseline.sh configs/launch/sim2real.env
```
