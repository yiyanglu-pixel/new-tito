import argparse
import csv
import os
from pathlib import Path

import numpy as np


def result_suffix(args):
    if args.initialization == "random":
        return f"random_init_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}"
    return f"init_{str(int(args.initialization)).zfill(6)}_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}"


def summary_suffix(args):
    suffix = result_suffix(args)
    if args.lag_vamp is not None:
        suffix = f"{suffix}_vamp_lag_{args.lag_vamp}"
    return suffix


def molecule_dir(args, mol_idx):
    return (
        Path(args.results_root)
        / args.data_set
        / args.sub_data_set
        / args.model
        / args.split
        / f"mol_{mol_idx:05d}"
    )


def md_dir(args, mol_idx):
    return (
        Path(args.results_root)
        / args.data_set
        / args.sub_data_set
        / "md"
        / args.split
        / f"mol_{mol_idx:05d}"
    )


def with_vamp_lag(path, args):
    if args.lag_vamp is None:
        return None
    return path.with_name(f"{path.stem}_vamp_lag_{args.lag_vamp}{path.suffix}")


def load_optional(*paths):
    for path in paths:
        if path is not None and path.exists():
            return np.load(path, allow_pickle=True)
    return None


def scalar_or_nan(value):
    if value is None:
        return np.nan
    arr = np.asarray(value)
    if arr.size == 0:
        return np.nan
    return float(arr.reshape(-1)[0])


def shape_string(value):
    if value is None:
        return ""
    return "x".join(str(dim) for dim in np.asarray(value).shape)


def summarize_molecule(args, mol_idx):
    suffix = result_suffix(args)
    mol_path = molecule_dir(args, mol_idx)
    md_path = md_dir(args, mol_idx)

    vamp_gap_path = mol_path / f"vamp_gap_{suffix}.npy"
    pred_svs_path = mol_path / f"vamp_singular_values_{suffix}.npy"
    ref_svs_path = md_path / f"vamp_singular_values_lag_{int(args.lag)}.npy"
    vamp_gap = load_optional(with_vamp_lag(vamp_gap_path, args), vamp_gap_path)
    pred_svs = load_optional(with_vamp_lag(pred_svs_path, args), pred_svs_path)
    ref_svs = load_optional(with_vamp_lag(ref_svs_path, args), ref_svs_path)
    tica = load_optional(mol_path / f"tica_projections_{suffix}.npy")
    dihedrals = load_optional(mol_path / f"dihedrals_{suffix}.npy")

    row = {
        "mol_idx": mol_idx,
        "complete": all(value is not None for value in [vamp_gap, pred_svs, ref_svs, tica, dihedrals]),
        "vamp_gap": scalar_or_nan(vamp_gap),
        "ref_sv1": scalar_or_nan(ref_svs),
        "pred_sv1": scalar_or_nan(pred_svs),
        "tica_shape": shape_string(tica),
        "dihedral_shape": shape_string(dihedrals),
    }
    if ref_svs is not None and pred_svs is not None:
        n = min(np.asarray(ref_svs).size, np.asarray(pred_svs).size)
        row["sv_l2"] = float(np.linalg.norm(np.asarray(ref_svs).reshape(-1)[:n] - np.asarray(pred_svs).reshape(-1)[:n]))
    else:
        row["sv_l2"] = np.nan
    return row


def print_stats(rows):
    complete = [row for row in rows if row["complete"]]
    gaps = np.array([row["vamp_gap"] for row in complete], dtype=float)
    gaps = gaps[np.isfinite(gaps)]

    print(f"molecules requested: {len(rows)}")
    print(f"complete molecules: {len(complete)}")
    if gaps.size == 0:
        print("no finite vamp_gap values found")
        return
    print(f"vamp_gap mean: {gaps.mean():.6g}")
    print(f"vamp_gap median: {np.median(gaps):.6g}")
    print(f"vamp_gap std: {gaps.std(ddof=0):.6g}")
    print(f"vamp_gap min: {gaps.min():.6g}")
    print(f"vamp_gap max: {gaps.max():.6g}")
    print("best molecules:", ", ".join(str(row["mol_idx"]) for row in sorted(complete, key=lambda r: r["vamp_gap"])[:5]))
    print("worst molecules:", ", ".join(str(row["mol_idx"]) for row in sorted(complete, key=lambda r: r["vamp_gap"], reverse=True)[:5]))


def main():
    parser = argparse.ArgumentParser(description="Summarize TITO analysis outputs into a CSV.")
    parser.add_argument("--data_set", type=str, default="mdqm9")
    parser.add_argument("--sub_data_set", type=str, default="version_0")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model", type=str, default="mdqm9")
    parser.add_argument("--lag", type=float, default=1000.0)
    parser.add_argument("--nested_samples", type=int, default=1000)
    parser.add_argument("--ode_steps", type=int, default=20)
    parser.add_argument("--lag_vamp", type=int, default=None)
    parser.add_argument("--initialization", type=str, default="random")
    parser.add_argument("--mol_indices", type=int, nargs="+", default=list(range(1251)))
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    rows = [summarize_molecule(args, mol_idx) for mol_idx in args.mol_indices]
    output = args.output
    if output is None:
        suffix = summary_suffix(args)
        output = os.path.join(args.results_root, args.data_set, args.sub_data_set, args.model, args.split, f"summary_{suffix}.csv")

    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {output}")
    print_stats(rows)


if __name__ == "__main__":
    main()
