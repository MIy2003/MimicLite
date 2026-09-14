# CHIP resume: 12,000 → 20,000 iterations

- Server: SSH alias `5090`.
- GPUs: `2,5,6,7`; 7,168 environments per GPU. User explicitly authorized
  sharing GPU 2 with its existing approximately 0.9 GiB process.
- W&B: `YM42/mimic_lite/dbiroz1a`, `WANDB_RESUME=must`.
- Source checkpoint: `active-adaptation/records/chip-5090/train-2374-7168-12000-ym42/wandb/run-20260908_142820-dbiroz1a/files/checkpoint_12000.pt`.
- Remote tmux: `chip-resume-20000`.
- Remote log: `active-adaptation/records/chip-5090/resume-dbiroz1a-20000/train.log`.
- Remote launcher: `active-adaptation/records/chip-5090/launch_chip_resume_20000.sh`.
- Independent remote entrypoint: `projects/mimic-lite/scripts/train_resume_20000.py`.
  It preserves the resumed W&B name and allows updating the run configuration.
  Original training entrypoint and source checkpoint are not overwritten.

The source checkpoint does not contain Adam optimizer state: weights,
normalization and saved iteration are restored, but optimizer moments restart.
The normal training schedule is evaluated against the new total of 20,000;
this is not bitwise continuation of the original 12,000-iteration schedule.

Startup verified: all four ranks loaded actor/critic/normalization state;
W&B explicitly reported resuming `dbiroz1a`; the loop completed 2/8,000
additional updates after saving checkpoint_12000 in the new output directory.
Observed GPU memory (2/5/6/7): 30,375 / 28,931 / 29,250 / 28,809 MiB.
No startup OOM was observed. This verifies launch, not completion or convergence.
