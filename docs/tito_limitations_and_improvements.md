# TITO Limitations and First-Step Improvements

This note maps the paper's stated limitations to the current codebase and to the first engineering changes in this branch.

## Current Limits

- Solvent and scale: the paper reports vacuum small-molecule simulations and implicit-water peptide simulations. The current graph code builds fully connected or radius graphs from Cartesian coordinates, but it has no unit-cell, minimum-image, or periodic boundary condition path.
- Large-system extrapolation: the paper uses Flory-style base-density rescaling for larger peptides. Before this branch, the code exposed only a manual `--base_density_std`, which made the extrapolation prior hard to reproduce.
- Thermodynamic state: the paper targets NVT at room temperature. The model contained a partial `temperature` switch, but train/sample scripts and datasets did not consistently provide thermodynamic condition fields.
- Long-run stability: sampling used only forward Euler and had no optional recentering or per-step displacement guard beyond the model's own centering.
- Energy correction and Hybrid MC: the repository does not contain OpenMM or another MD engine interface, potential-energy evaluation, box information, or proposal-density accounting, so a physically valid Metropolis correction cannot be added as a small patch.

## Implemented First Step

- Added a reproducible conda entry point through `environment.yml` and moved CUDA 12.1 wheel installation to `requirements-cuda121.txt`.
- Generalized `PainnCondVelocity` from a hard-coded temperature flag to optional scalar condition names and scales.
- Added training and sampling CLI flags for temperature and pressure conditions. These fields are optional and absent by default, preserving existing runs.
- Added `euler` and `heun` sampling modes, optional per-step displacement clipping, and per-step recentering.
- Added `fixed` and `flory` base-density scaling so peptide-size extrapolation experiments can record the reference size and exponent in saved args.

## Deferred Work

- Hybrid MC should be implemented only after adding an MD backend abstraction with energy evaluation, short relaxation, temperature-aware acceptance, and clear reporting of acceptance rates.
- PBC support requires unit-cell data in the datasets, PBC-aware neighbor construction, and minimum-image displacements in edge feature construction.
- Energy regularization requires training data with reliable energies and a loss term that is benchmarked against the current flow-matching objective.
- True thermodynamic transfer requires training data at multiple temperatures or pressures; the new condition interface is only the plumbing needed to run those experiments.

## Suggested Next Experiments

- Train a baseline conditional model with `--condition_temperature --temperature 300 --temperature_reference 300` to verify compatibility with current data.
- Compare Euler and Heun sampling at identical `--ode_steps` and lag values using VAMP and JSD metrics.
- For peptide extrapolation, run sampling with `--base_density_scaling flory --base_density_reference_size <training_atom_count>` and compare radius-of-gyration drift against the fixed-std baseline.
