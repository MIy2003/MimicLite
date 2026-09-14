# Lambda migration and single-GPU capacity probe

Destination: `/share/ml/yangmin/mimiclite-chip` on `ssh lambda`.
Copied active-adaptation (including mimic-lite), any4hdmi, the full accepted
loco-manip dataset, local checkpoints and public G1 XML cache. Excluded virtual
environments, Git internals, runtime logs, generated FK caches and credentials.
The source machine and 5090 training are not modified or stopped.

User-requested allocation:

```bash
salloc -N 1 -t 8:00:00 --cpus-per-task 64 --account=research --qos=lv0a --job-name dexhand --gres=gpu:1 -p HGX,DGX
```

Allocated job 7987723 on hgx-hyperplane04: one NVIDIA A100-SXM4-80GB,
driver 575.57.08, 64 CPUs and 110000 MiB host memory. This is a single-card
capacity test, not authorization to use unallocated physical GPUs.

Hypothesis: A100 80GB supports more than 7168 environments per rank with enough
memory headroom, but larger batches may not improve rollout throughput.
Probe stages: 64 environments / 50 updates, then 7168, 12288, 16384, 20480 /
10 updates each. Stop on errors or 80% PyTorch reserved-memory gate. Check total
GPU use and warmed-up timing separately; PyTorch counters exclude some simulator
allocations. No convergence claim is made by this throughput probe.

Scripts: `active-adaptation/projects/mimic-lite/scripts/profile_chip_lambda.{sh,py}`.
All probe jobs disable W&B and start from scratch; the original dbiroz1a run is
not resumed here. Four-card production requires its own allocation and validation.
The setup uses Python 3.11 and the existing MJLab environment lockfile, followed
by explicit project installs; no virtual environment is copied from another host.

Status: transfer and installation completed. Python 3.11.16, Torch 2.11.0+cu128,
MJLab 1.6.0, MuJoCo/MuJoCo-Warp 3.11.0. Ten CHIP regression tests passed.
Project discovery initially left mimic_lite disabled; `aa-project enable
mimic_lite` fixed registration. Failed logs are retained as
`records/profile-7987723-registration-failed*`.

Named tmux session `mimic-chip-profile-7987723` runs `srun --jobid=7987723`.
Log: `records/profile-7987723-launch.log`; metrics: `records/profile-7987723/`.
The 64-environment gate completed 50 finite PPO updates. All four larger probes
completed 10 updates each, with finite metrics and no OOM. Timing excludes the
first three iterations; these are short throughput/capacity probes, not evidence
of convergence or long-run stability.

| Environments/GPU | Peak Torch reserved GiB | Warm median seconds/iteration |
| --- | --- | --- |
| 7168 | 24.65 | 4.84 |
| 12288 | 41.96 | 6.07 |
| 16384 | 56.09 | 7.04 |
| 20480 | 69.82 | 8.44 |

Recommendation for a subsequent **A100 80GB four-GPU smoke test**: 16384 per GPU.
20480 completed, but total nvidia-smi memory reached about 74.6 GiB, and it
exceeded the conservative 80% Torch-reserved gate. Its sample throughput gain
over 16384 was only about 4.3%. Torch counters exclude simulator/driver allocations.
Four ranks at 16384 means 65536 total environments versus the old 28672:
this changes PPO batch size and samples per iteration, not just device placement.
DDP memory, communication, host memory and learning behavior remain unvalidated.

Probe log ended in `PROFILE_COMPLETE`; the one-GPU allocation is released after
the probe. Local metrics copy: `active-adaptation/records/lambda/profile-7987723/`.
No four-GPU training job has been submitted; 5090 training and W&B are untouched.
