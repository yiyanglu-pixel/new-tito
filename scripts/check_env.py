import subprocess
import sys
import textwrap


MODULES = [
    "torch",
    "torch_geometric",
    "torch_scatter",
    "torch_sparse",
    "torch_cluster",
    "pyg_lib",
    "lightning",
    "rdkit",
    "mdtraj",
    "deeptime",
    "h5py",
]


def run_python(title, code, timeout=60):
    print(f"\n== {title} ==", flush=True)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print("TIMEOUT", flush=True)
        return False

    if proc.stdout:
        print(proc.stdout.rstrip(), flush=True)
    if proc.stderr:
        print(proc.stderr.rstrip(), flush=True)

    if proc.returncode == 0:
        print("OK", flush=True)
        return True

    if proc.returncode < 0:
        print(f"FAILED: terminated by signal {-proc.returncode}", flush=True)
    else:
        print(f"FAILED: exit code {proc.returncode}", flush=True)
    return False


def main():
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.replace(chr(10), ' ')}")

    for module in MODULES:
        run_python(
            f"import {module}",
            f"""
            import importlib
            mod = importlib.import_module("{module}")
            version = getattr(mod, "__version__", "unknown")
            path = getattr(mod, "__file__", "built-in")
            print("version:", version)
            print("path:", path)
            """,
        )

    run_python(
        "torch cuda smoke test",
        """
        import torch
        print("torch:", torch.__version__)
        print("torch.version.cuda:", torch.version.cuda)
        print("cuda available:", torch.cuda.is_available())
        print("compiled arch list:", getattr(torch.cuda, "get_arch_list", lambda: [])())
        if torch.cuda.is_available():
            print("device count:", torch.cuda.device_count())
            for idx in range(torch.cuda.device_count()):
                print(f"device[{idx}]:", torch.cuda.get_device_name(idx), torch.cuda.get_device_capability(idx))
            x = torch.ones(8, device="cuda")
            print("cuda tensor sum:", float((x * x).sum().cpu()))
        """,
    )

    run_python(
        "torch_scatter smoke test",
        """
        import torch
        from torch_scatter import scatter
        device = "cuda" if torch.cuda.is_available() else "cpu"
        src = torch.tensor([1.0, 2.0, 3.0], device=device)
        index = torch.tensor([0, 0, 1], device=device)
        out = scatter(src, index, dim=0, reduce="sum")
        print("device:", device)
        print("scatter:", out.cpu().tolist())
        """,
    )

    run_python(
        "torch_geometric radius_graph smoke test",
        """
        import torch
        import torch_geometric as geom
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [3.0, 0.0, 0.0]], device=device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        edge_index = geom.nn.radius_graph(x, r=1.0, batch=batch, max_num_neighbors=8)
        print("device:", device)
        print("edges:", edge_index.cpu().tolist())
        """,
    )

    run_python(
        "PyG wheel hint",
        """
        import torch
        base_version = torch.__version__.split("+", 1)[0]
        major, minor, *_ = base_version.split(".")
        torch_key = f"{major}.{minor}.0"
        if torch.version.cuda is None:
            cuda_key = "cpu"
        else:
            cuda_key = "cu" + torch.version.cuda.replace(".", "")
        print("Use the wheel index matching torch and torch.version.cuda:")
        print(f"https://data.pyg.org/whl/torch-{torch_key}+{cuda_key}.html")
        print("Example:")
        print(f"pip install --force-reinstall --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-{torch_key}+{cuda_key}.html")
        """,
    )


if __name__ == "__main__":
    main()
