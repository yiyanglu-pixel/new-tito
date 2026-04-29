# TITO MDQM9 Reproduction Execution Plan

This branch records a reproducible MDQM9-only plan for running TITO on
`mmgpu11` and includes the small code fixes needed during reproduction.
Timewarp is intentionally out of scope.

## Server And Directory Layout

```bash
ssh mmgpu11

mkdir -p /localhome3/lyy/repro
cd /localhome3/lyy/repro

git clone https://github.com/yiyanglu-pixel/new-tito.git tito
cd /localhome3/lyy/repro/tito

mkdir -p logs datasets/mdqm9-nc results
```

Expected final layout:

```text
/localhome3/lyy/repro/tito/
  datasets/mdqm9-nc/
    mdqm9-nc.hdf5
    mdqm9-nc.sdf
    splits/
  inference_files/
    mdqm9.ckpt
    test_mol.pdb
  logs/
  results/
```

## Conda Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh

conda create -n tito python=3.11 -y
conda activate tito

python -m pip install -U pip uv
uv pip install -e .

uv pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

uv pip install lightning torch_geometric wandb h5py rdkit mdtraj deeptime scipy tqdm
```

Validate:

```bash
python - <<'PY'
import torch, torch_geometric, lightning
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.version.cuda)
print("pyg", torch_geometric.__version__)
print("lightning", lightning.__version__)
PY
```

Expected validated environment:

```text
torch 2.5.0+cu121
CUDA available
torch_geometric 2.7.0 or compatible
lightning 2.6.1 or compatible
```

## MDQM9 Data Preparation

Preferred command:

```bash
cd /localhome3/lyy/repro/tito
conda activate tito

bash scripts/preprocessing/download_mdqm9_nc.sh \
  /localhome3/lyy/repro/tito/datasets/mdqm9-nc
```

Manual fallback:

```bash
cd /localhome3/lyy/repro/tito
mkdir -p datasets/mdqm9-nc/parts datasets/_sources

cd datasets/_sources
wget -O mdqm9-nc-loaders-main.zip \
  https://codeload.github.com/olsson-group/mdqm9-nc-loaders/zip/refs/heads/main
unzip mdqm9-nc-loaders-main.zip
cp -a mdqm9-nc-loaders-main/splits \
  /localhome3/lyy/repro/tito/datasets/mdqm9-nc/

cd /localhome3/lyy/repro/tito/datasets/mdqm9-nc
wget -c -O mdqm9-nc.sdf \
  https://zenodo.org/records/10579242/files/mdqm9-nc.sdf?download=1

for i in 00 01 02 03 04 05 06 07 08 09; do
  wget -c -O parts/mdqm9-nc_${i} \
    https://zenodo.org/records/10579242/files/mdqm9-nc_${i}?download=1
done

cat parts/mdqm9-nc_* > mdqm9-nc.hdf5
```

Data validation:

```bash
cd /localhome3/lyy/repro/tito
python - <<'PY'
import h5py, numpy as np
base = "datasets/mdqm9-nc"
h5 = h5py.File(f"{base}/mdqm9-nc.hdf5", "r")
print("molecules", len(h5.keys()))
for split in ["train", "val", "test"]:
    idx = np.load(f"{base}/splits/{split}_indices.npy")
    print(split, len(idx), idx[:5].tolist())
k = sorted(h5.keys())[0]
print("first", k, h5[k]["trajectories"]["md_0"].shape, h5[k]["data"]["time_lag"][()])
PY
```

Expected MDQM9 facts:

```text
total molecules: 12506
train/val/test molecules: 8754 / 2501 / 1251
each MD trajectory: 16000 frames
test frame interval: molecule-dependent, 1.031-2.281 ps
```

## Sanity Inference With Pretrained Checkpoint

This matches the README demo. It validates the pipeline but is not a
paper-level evaluation.

```bash
cd /localhome3/lyy/repro/tito
conda activate tito

CUDA_VISIBLE_DEVICES=1 python -u scripts/sample.py \
  --pdb_path inference_files/test_mol.pdb \
  --model_path inference_files/mdqm9.ckpt \
  --data_set mdqm9 \
  --data_path /localhome3/lyy/repro/tito/datasets/mdqm9-nc/ \
  --lag 1000 \
  --nested_samples 10 \
  --batch_size 1 \
  --ode_steps 20 \
  --save_pdb \
  > logs/mdqm9_pretrained_sample_demo.log 2>&1
```

Expected output:

```text
results/mdqm9/version_0/mdqm9/test/mol_00000/
  random_init_lag_1000_nested_10_ode_steps_20.pkl
  random_init_lag_1000_nested_10_ode_steps_20.pdb
```

Expected `.pkl` trajectory shape:

```text
(1, 11, n_atoms, 3)
```

Run the demo analysis:

```bash
python -u scripts/analyse.py \
  --data_set mdqm9 \
  --data_path /localhome3/lyy/repro/tito/datasets/mdqm9-nc/ \
  --split test \
  --mol_indices 0 \
  --model mdqm9 \
  --lag 1000 \
  --nested_samples 10 \
  --ode_steps 20 \
  --lag_tica 1 \
  --lag_vamp 1 \
  > logs/mdqm9_pretrained_analyse_demo.log 2>&1
```

## Full MDQM9 Training

Use physical GPU 2-7 for DDP training. GPU 0 remains available for
monitoring and GPU 1 can run small inference jobs.

```bash
cd /localhome3/lyy/repro/tito
conda activate tito

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
WANDB_MODE=offline \
PYTHONUNBUFFERED=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
OMP_NUM_THREADS=4 \
nohup python -u scripts/train.py \
  --data_set mdqm9 \
  --data_path /localhome3/lyy/repro/tito/datasets/mdqm9-nc/ \
  --epochs 1 \
  --batch_size 208 \
  --max_lag 1000 \
  --num_workers 6 \
  --multigpu \
  --save_freq 10 \
  --progress_log_every 100 \
  > logs/mdqm9_train_full_epoch1_b208_w6.log 2>&1 < /dev/null &

echo $! > logs/mdqm9_train_ddp.pid
```

Monitor:

```bash
tail -f logs/mdqm9_train_full_epoch1_b208_w6.log
nvidia-smi
ps -u yiyanglu -o pid,stat,etime,pcpu,pmem,cmd | grep train.py
```

Fallbacks:

```text
if OOM: batch_size 208 -> 192 -> 160 -> 128
if CPU pressure: num_workers 6 -> 4
```

Locate the latest trained checkpoint:

```bash
find /localhome3/lyy/repro/tito -type f -name "*.ckpt" \
  ! -path "*/inference_files/*" \
  -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" | sort | tail -20
```

## Paper-Level Inference

Do not use `--pdb_path` for paper-level evaluation. Sample from the
MDQM9 test split so the generated molecules match the reference MD.

Create test split shards:

```bash
cd /localhome3/lyy/repro/tito
mkdir -p logs/eval_indices

python - <<'PY'
import numpy as np
indices = np.arange(1251)
for gpu, chunk in enumerate(np.array_split(indices, 8)):
    path = f"logs/eval_indices/test_gpu{gpu}.txt"
    with open(path, "w") as f:
        f.write(" ".join(map(str, chunk.tolist())))
    print(gpu, len(chunk), int(chunk[0]), int(chunk[-1]))
PY
```

Default full evaluation sampling:

```text
test molecules: 1251
trajectories per molecule: 32
frames per trajectory: 1001
model lag: 1000 ps
model time per trajectory: 1000 ns = 1 us
```

Run pretrained full inference on all GPUs:

```bash
cd /localhome3/lyy/repro/tito
conda activate tito

for gpu in 0 1 2 3 4 5 6 7; do
  ids=$(cat logs/eval_indices/test_gpu${gpu}.txt)
  CUDA_VISIBLE_DEVICES=${gpu} \
  PYTHONUNBUFFERED=1 \
  nohup python -u scripts/sample.py \
    --model_path inference_files/mdqm9.ckpt \
    --data_set mdqm9 \
    --data_path /localhome3/lyy/repro/tito/datasets/mdqm9-nc/ \
    --split test \
    --mol_indices ${ids} \
    --lag 1000 \
    --nested_samples 1000 \
    --batch_size 32 \
    --ode_steps 20 \
    > logs/mdqm9_pretrained_full_sample_gpu${gpu}.log 2>&1 < /dev/null &
  echo $! > logs/mdqm9_pretrained_full_sample_gpu${gpu}.pid
done
```

If 10GB GPU memory is insufficient, keep the effective sample count at
32 by using four jobs of `batch_size=8`:

```bash
for job in 0 1 2 3; do
  for gpu in 0 1 2 3 4 5 6 7; do
    ids=$(cat logs/eval_indices/test_gpu${gpu}.txt)
    CUDA_VISIBLE_DEVICES=${gpu} \
    PYTHONUNBUFFERED=1 \
    nohup python -u scripts/sample.py \
      --model_path inference_files/mdqm9.ckpt \
      --data_set mdqm9 \
      --data_path /localhome3/lyy/repro/tito/datasets/mdqm9-nc/ \
      --split test \
      --mol_indices ${ids} \
      --lag 1000 \
      --nested_samples 1000 \
      --batch_size 8 \
      --ode_steps 20 \
      --i_job ${job} \
      > logs/mdqm9_pretrained_full_sample_gpu${gpu}_job${job}.log 2>&1 < /dev/null &
  done
  wait
done
```

Repeat the same commands for a new trained checkpoint by replacing
`--model_path inference_files/mdqm9.ckpt` with the latest checkpoint path.

## Paper-Level Analysis

Run analysis in shards:

```bash
cd /localhome3/lyy/repro/tito
conda activate tito

for gpu in 0 1 2 3 4 5 6 7; do
  ids=$(cat logs/eval_indices/test_gpu${gpu}.txt)
  nohup python -u scripts/analyse.py \
    --data_set mdqm9 \
    --data_path /localhome3/lyy/repro/tito/datasets/mdqm9-nc/ \
    --split test \
    --mol_indices ${ids} \
    --model mdqm9 \
    --lag 1000 \
    --nested_samples 1000 \
    --ode_steps 20 \
    --lag_tica 1 \
    --lag_vamp 1 \
    > logs/mdqm9_pretrained_full_analyse_part${gpu}.log 2>&1 < /dev/null &
  echo $! > logs/mdqm9_pretrained_full_analyse_part${gpu}.pid
done
```

If inference used jobs, add:

```bash
--jobs 0 1 2 3
```

Validation:

```bash
find results/mdqm9/version_0/mdqm9/test \
  -name "dihedrals_random_init_lag_1000_nested_1000_ode_steps_20.npy" | wc -l

find results/mdqm9/version_0/mdqm9/test \
  -name "tica_projections_random_init_lag_1000_nested_1000_ode_steps_20.npy" | wc -l

find results/mdqm9/version_0/mdqm9/test \
  -name "vamp_gap_random_init_lag_1000_nested_1000_ode_steps_20.npy" | wc -l
```

Expected count is close to `1251` for each file type.

## Summary CSV

Generate a minimal summary after analysis:

```bash
cd /localhome3/lyy/repro/tito

python - <<'PY'
import csv, os
import h5py
import numpy as np

base = "datasets/mdqm9-nc"
h5 = h5py.File(f"{base}/mdqm9-nc.hdf5", "r")
test = np.load(f"{base}/splits/test_indices.npy")
models = ["mdqm9"]
rows = []

for model_name in models:
    for local_idx, global_idx in enumerate(test):
        key = str(int(global_idx)).zfill(5)
        model_dir = f"results/mdqm9/version_0/{model_name}/test/mol_{local_idx:05d}"
        gap_path = f"{model_dir}/vamp_gap_random_init_lag_1000_nested_1000_ode_steps_20.npy"
        tica_path = f"{model_dir}/tica_projections_random_init_lag_1000_nested_1000_ode_steps_20.npy"
        dihedral_path = f"{model_dir}/dihedrals_random_init_lag_1000_nested_1000_ode_steps_20.npy"
        status = "ok" if all(os.path.exists(p) for p in [gap_path, tica_path, dihedral_path]) else "missing"
        gap = float(np.load(gap_path)) if os.path.exists(gap_path) else np.nan
        dt = float(h5[key]["data"]["time_lag"][()])
        md_frames = h5[key]["trajectories"]["md_0"].shape[0]
        rows.append({
            "model": model_name,
            "local_idx": local_idx,
            "global_idx": key,
            "n_atoms": h5[key]["trajectories"]["md_0"].shape[1],
            "md_frames": md_frames,
            "md_dt_ps": dt,
            "md_total_ns": dt * (md_frames - 1) / 1000,
            "vamp_gap": gap,
            "status": status,
        })

os.makedirs("results/mdqm9/version_0/eval", exist_ok=True)
out = "results/mdqm9/version_0/eval/summary.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(out, len(rows))
PY
```

## Evaluation Interpretation

Use the generated artifacts for distribution and kinetics comparison:

```text
reference MD:
  results/mdqm9/version_0/md/test/mol_XXXXX/

model samples:
  results/mdqm9/version_0/<model>/test/mol_XXXXX/
```

Compare:

```text
dihedral distributions
TICA 2D densities
VAMP singular values
VAMP gap
outlier molecules with replica-exchange trajectories, when available
```

Do not perform frame-by-frame comparisons. TITO samples stochastic
trajectories; evaluation is distributional and kinetic.

## Assumptions

- Only MDQM9 is reproduced.
- `mol_indices` are split-local indices, so full test is `0..1250`.
- Reference MD for evaluation is test split MD, not training data.
- Paper-level default uses `32` trajectories per molecule and `1001`
  frames per trajectory.
- If GPU memory is insufficient, use multiple `--i_job` runs and aggregate
  with `analyse.py --jobs`.
- Current 3080 10GB GPUs should reserve one card for monitoring or small
  inference while training is active.
