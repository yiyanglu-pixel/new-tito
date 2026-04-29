# Transferable Generative Models Bridge Femtosecond to Nanosecond Time-Step Molecular Dynamics

## Inference
To sample a trajetory with 1 ns timesteps for a molecule in a pdb file with TITO, run:
```
python scripts/sample.py --pdb_path inference_files/test_mol.pdb --model_path inference_files/mdqm9.ckpt --data_set mdqm9 --data_path /proj/berzelius-2025-189/datasets/mdqm9/ --lag 1000 --nested_samples 10 --batch_size 1 
```

## Training
To train a TITO model, run:
```
python scripts/train.py --dataset <mdqm9 or timewarp> --data_path <path to dataset> --epochs <number_of_epochs>
```

## Installation
For the environment installation on GPUs supported by PyTorch 2.5/cu121, first create an environment with python 3.11:
 ```
mamba create -n tito python=3.11
  ```
Then run:
```
  mamba activate tito
  uv pip install -e .
  uv pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121
  uv pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
  uv pip install lightning torch_geometric
```

For Blackwell/sm_100 GPUs, do not use the PyTorch 2.5/cu121 commands above. Install a PyTorch build that supports the GPU, then install PyG extension wheels from the matching PyG wheel index. For example, with PyTorch 2.7 and CUDA 12.8:
```
pip install --force-reinstall --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install torch_geometric lightning
python scripts/check_env.py
```
If you use NVIDIA containers, prefer the NVIDIA PyG container for your driver/CUDA stack instead of a plain `nvcr.io/nvidia/pytorch` container. The plain PyTorch container can still be missing `torch_scatter`, `torch_sparse`, or `torch_cluster`.

## Dataset pre-processing
### Small molecules
Download the mdqm9-nc dataset from [this link](https://github.com/olsson-group/mdqm9-nc-loaders). The h5 file, sdf file and splits need to be in the same directory.
### Timewarp
Download the Timewarp dataset using [scripts/preprocessing/download_timewarp.py](scripts/preprocessing/download_timewarp.py) and pre-process it with [scripts/preprocessing/timewarp_to_h5.py](scripts/preprocessing/timewarp_to_h5.py).

## Paper

This repository contains code to reproduce the results from:

```
@misc{https://doi.org/10.48550/arxiv.2510.07589,
  doi = {Diez2025},
  url = {https://arxiv.org/abs/2510.07589},
  author = {Diez,  Juan Viguera and Schreiner,  Mathias and Olsson,  Simon},
  title = {Transferable Generative Models Bridge Femtosecond to Nanosecond Time-Step Molecular Dynamics},
  publisher = {arXiv},
  year = {2025},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
