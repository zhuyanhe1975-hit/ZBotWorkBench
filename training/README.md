# Optional local mjlab training worker

The static workbench and replay remain standalone. Training runs in a separate,
optional local Python process; it never runs in the browser.

The included `tasks/zbot_rl_mjlab` snapshot supplies one real six-joint ZBot PPO
task. `SOURCE_PROVENANCE.json` records its origin and file hashes, and the asset
directory retains its original provenance. The sibling project is not needed by
the worker after this snapshot has been copied.

Use an existing Python environment with `mjlab==1.3.0` and
`rsl-rl-lib==5.2.0`, or install `requirements.txt` into a dedicated environment.
The verified environment uses MuJoCo/MuJoCo-Warp 3.8.1, Warp 1.12.1, Torch
2.9.1 and SciPy 1.15.3. CPU training is supported; CUDA training additionally needs a compatible
GPU and driver. `requirements.txt` pins those tested versions. Select the matching
Torch 2.9.1 wheel for the machine: the verified GPU environment uses `2.9.1+cu128`,
while a CPU-only environment can use the CPU wheel. No global environment is
changed by the workbench. First use compiles Warp kernels, which can take longer than a
small training run. Kernel caches are shared between jobs under their parent
directory's `.zbot-training-cache`, unless `XDG_CACHE_HOME` is supplied.

```bash
/path/to/python training/probe.py
/path/to/python training/worker.py --request /absolute/job/request.json --job-dir /absolute/job
```

The request uses the application's `TrainingConfig` fields. `taskId` is
`Mjlab-Zbot-6dof-Bipedal-Walking` (from scratch) or
`Mjlab-Zbot-6dof-Walking-Finetune` (packaged model_801 warm start); `device` is `cpu` or `cuda:0` (etc.).
CUDA indices are the physical indices reported by the service's `nvidia-smi`
scan. Before importing Torch, each GPU worker sets PCI bus enumeration and
`CUDA_VISIBLE_DEVICES` to that requested index, replacing any inherited mask;
inside that worker the selected GPU is always `cuda:0`. Ready events and saved
checkpoint provenance retain both requested and internal device names. CPU jobs
clear CUDA visibility and are limited to 64 environments.
On Linux, the worker also restricts its own CPU affinity to at most `cpuThreads`
CPUs from its inherited allowed mask, before importing Torch or Warp. This bounds
native thread pools as well as OMP/MKL/Torch threads; it never changes the service
or host affinity. Ready events and checkpoint provenance record the effective
mask and thread count. Platforms that do not permit affinity retain library
thread limits. The lightweight probe reports `cudaBuild` when it can identify
the Torch build from its version file, without importing Torch or using a GPU.
The local service may additionally supply a resolved `resumePath` belonging to
an existing job. The worker allows only the fixed task architecture (ordinary MLP or the
packaged QuatFeatureMLP), rejects recurrent/unknown models and foreign
resume checkpoints lacking its own format marker, restores the saved task/PPO settings,
and interprets `iterations` as additional updates.

Stdout contains JSON events: `ready`, `progress`, `checkpoint`, `bundle`,
`finished`, or `error`. Native library logs are routed to stderr. A `STOP` file
in the job directory or SIGTERM requests a cooperative stop: the current update
finishes, then its checkpoint is saved and exported. An external service can
enforce a hard grace-period limit if initialization or an update is unresponsive.

Outputs include `checkpoints/model_N.pt`, TensorBoard logs, `walking_config.json`
inside `logs/`, `versions.json`, the exported scene in `model/`, and `bundle.json`.
Checkpoints retain actor, critic, optimizer, task, PPO and curriculum state.
Bundles contain only actor MLP weights, an explicit feature-transform marker
when required, self-contained MJCF, flattened
`training_*.obj` meshes, reset pose, observation/control contract and runtime
versions. Physics/control timesteps come from the actual training environment;
the browser must not silently substitute its older replay model or timestep.

A one-update smoke run verifies the pipeline, not learned walking. Training
quality depends on the reward, settings, initialization and training duration.

The from-scratch task overrides the copied task-card defaults with the setup
that produced the verified model_299 baseline: 1/240 s physics, decimation 8
(30 Hz control), and 8 physics steps of contact history. Its GPU UI defaults
are 4096 environments and 300 updates. Older workbench checkpoints made with
60 Hz physics are rejected because their policy and optimizer state belong to
a different contact/control discretization.
The worker also calls mjlab's `configure_torch_backends()` before environment
creation, matching the official training entry point's TF32 configuration.

Live preview is opt-in. The service creates an owned `LIVE_PREVIEW` marker only
while the user requests the view. At update boundaries the worker then copies
at most nine environments' body poses to CPU. With preview off, no pose copy is
performed and the browser does not create the training WebGL scene.

Reward task-card updates use an atomically replaced `TASK_CARD.json`. The fixed
allowlist contains all stage-one and stage-two reward terms plus the termination
penalty; unknown, missing, non-finite, or out-of-range fields are rejected by
both the service and worker. The worker applies a new revision only at an PPO
update boundary and refreshes `runner.walking_config`, so subsequent checkpoints
restore the actual weights used after the change.
