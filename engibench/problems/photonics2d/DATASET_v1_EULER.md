# Generating the Photonics2D v1 dataset on Euler

Runbook for producing `IDEALLab/photonics_2d_120_120_v1` on the ETH Euler cluster using
[`dataset_slurm_v1.py`](dataset_slurm_v1.py). Euler uses SLURM, so it works directly with the
EngiBench SLURM helpers (`engibench.utils.slurm`).

The job runs in two phases: **generate** submits one optimization per condition as a SLURM array and
collects the results into a pickle; **assemble** turns the pickle into a `DatasetDict` and pushes it
to the Hub. By default the condition grid is taken from the existing v0 dataset, so v1 covers the same
boundary conditions and is directly comparable.

## 1. One-time setup

```bash
# Build a Python env on Euler (a personal-scratch venv is simplest for the ceviche stack).
module load stack/2024-06 python/3.11.6        # check `module avail python` for the current name
python -m venv "$SCRATCH/engibench-venv"
source "$SCRATCH/engibench-venv/bin/activate"
pip install -U pip
pip install -e ".[photonics2d]"                # from the repo root, on the fix-photonics branch

# Authenticate to the Hub (needs write access to the IDEALLab org).
huggingface-cli login                          # or: export HF_TOKEN=hf_xxx
```

## 2. Resource sizing

A single 120x120 / 200-step optimization is ~2-3 min on a CPU core. Defaults are tuned for that:

| Parameter | Default | Notes |
|---|---|---|
| optimizations | ~2000 | one per v0 condition (20 λ1 x 20 λ2 x 5 blur) |
| `--group-size` | 2 | array tasks = `ceil(n/2)` ≈ 1000; keep ≤ Euler's `MaxArraySize` |
| `--runtime` | `00:20:00` | per array task (covers `group_size` optimizations + margin) |
| `--mem-per-cpu` | `4G` | headroom for ceviche's sparse FDFD solve |
| concurrency | 1000 | the helper caps the array at `%1000` |

Total compute is ~100 CPU-hours; wall time is typically well under an hour plus queue.

## 3. Generate (submit the array, collect results)

The `generate` driver blocks until the dependent collection job finishes, so run it in `tmux` (or as
the SLURM driver job in the appendix) rather than tying up a login shell.

```bash
tmux new -s photonics_v1
source "$SCRATCH/engibench-venv/bin/activate"
export OMP_NUM_THREADS=1                        # each task is single-core; avoid thread oversubscription

python -m engibench.problems.photonics2d.dataset_slurm_v1 generate \
    --from-v0 \
    --group-size 2 \
    --runtime 00:20:00 \
    --mem-per-cpu 4G \
    --out "$SCRATCH/photonics_v1_results.pkl"
# Euler shareholders: add `--account es_<group>`. Fair-share users omit `--account`.
```

Detach with `Ctrl-b d`; monitor with `squeue --me`. Per-task logs land in `./opt_logs_v1/`.

## 4. Assemble and push

```bash
python -m engibench.problems.photonics2d.dataset_slurm_v1 assemble \
    --results "$SCRATCH/photonics_v1_results.pkl" \
    --push
```

Failed array tasks are reported and skipped; without `--push` the dataset is written to disk for
inspection instead of uploaded.

## 5. Validate

```bash
# The v1 contract: the stored objective must reproduce under simulate.
python - <<'PY'
import numpy as np
from datasets import load_dataset
from engibench.problems.photonics2d import Photonics2D

ds = load_dataset("IDEALLab/photonics_2d_120_120_v1")
row = ds["test"][0]
p = Photonics2D(config={"lambda1": row["lambda1"], "lambda2": row["lambda2"], "blur_radius": row["blur_radius"]})
sim = float(p.simulate(np.array(row["optimal_design"]))[0])
print("stored:", row["total_overlap"], "simulate:", sim,
      "match:", bool(np.isclose(sim, row["total_overlap"], rtol=1e-4)))
PY

# The generic dataset-backed tests should now pass.
pytest tests/test_problem_implementations.py -k Photonics
```

## Euler notes

- **`$SCRATCH`** — the helper writes its per-job pickles under `$SCRATCH` (Euler sets it). Scratch is
  purged on a rolling basis, which is fine for these transient files; keep the final results pickle and
  the pushed dataset elsewhere if you need to retain them.
- **Account** — only shareholders pass `--account`; the fair-share queue needs none.
- **Array size** — if `ceil(n_jobs / group_size)` exceeds Euler's `MaxArraySize`, raise `--group-size`.
- **Environment propagation** — the helper submits with `--export=ALL` and uses the driver's
  interpreter, so workers inherit the activated venv automatically; just submit from inside it.

## Appendix: run the driver as a SLURM job instead of tmux

```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=photonics_v1_driver
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=photonics_v1_driver_%j.log
source "$SCRATCH/engibench-venv/bin/activate"
export OMP_NUM_THREADS=1
python -m engibench.problems.photonics2d.dataset_slurm_v1 generate \
    --from-v0 --group-size 2 --out "$SCRATCH/photonics_v1_results.pkl"
EOF
```

Give the driver job enough `--time` to outlast the whole array (it waits on the collection job).
