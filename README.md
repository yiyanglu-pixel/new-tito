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
Create an isolated conda environment:
```bash
conda env create -f environment.yml
conda activate tito
```

Install the CUDA 12.1 PyTorch and PyG wheels, then install this package:
```bash
uv pip install -r requirements-cuda121.txt
uv pip install -e .
```

Validate the environment:
```bash
python -c "import torch, tito; print(torch.__version__, torch.cuda.is_available(), tito.DEVICE)"
python scripts/train.py --help
python scripts/sample.py --help
```

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
