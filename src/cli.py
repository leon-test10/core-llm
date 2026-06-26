"""
CLI entry point for core-solver-demo.

Usage:
    python -m src.cli run <workflow.yaml>
    python -m src.cli validate <workflow.yaml>
    python -m src.cli download [--source=tendl|jendl] [--dir=data/nuclear_library]
"""

import sys
import argparse
from pathlib import Path

from .workflows.engine import WorkflowEngine
from .workflows.parser import parse_workflow_yaml, validate_workflow


def cmd_run(args):
    """Execute a workflow YAML file."""
    workflow_path = Path(args.workflow)
    if not workflow_path.exists():
        print(f"Error: workflow file not found: {workflow_path}")
        sys.exit(1)

    engine = WorkflowEngine()
    try:
        results = engine.run(workflow_path)
        print(f"\nDone. Results stored in outputs/")
    except Exception as e:
        print(f"\nError executing workflow: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_validate(args):
    """Validate a workflow YAML file without executing it."""
    workflow_path = Path(args.workflow)
    if not workflow_path.exists():
        print(f"Error: workflow file not found: {workflow_path}")
        sys.exit(1)

    spec = parse_workflow_yaml(workflow_path)
    issues = validate_workflow(spec)
    if issues:
        print(f"Validation found {len(issues)} issue(s):")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("Workflow is valid.")


def cmd_download(args):
    """Download nuclear data files."""
    from .core.xs import download_common_nuclides

    data_dir = args.dir or "data/nuclear_library"
    source = args.source or "tendl"

    print(f"Downloading common nuclides from {source}...")
    print(f"Target directory: {data_dir}")
    print()

    results = download_common_nuclides(data_dir=data_dir, source=source)

    print(f"\nDownloaded {len(results)} files:")
    for nid, path in sorted(results.items()):
        size_kb = path.stat().st_size / 1024
        print(f"  {nid}: {path.name} ({size_kb:.1f} KB)")


def cmd_gpu_status(args):
    """Display GPU availability and installation guidance."""
    from .core.backends import gpu_available, gpu_vendor, gpu_info

    print("GPU Acceleration Status")
    print("=" * 60)
    print(f"  Available: {gpu_available()}")
    print(f"  Vendor:    {gpu_vendor() or 'none'}")
    print(f"  Info:      {gpu_info()}")
    print()

    if not gpu_available():
        print("Installation Instructions:")
        print()
        print("  NVIDIA GPU:")
        print("    pip install cupy-cuda12x")
        print()
        print("  AMD GPU:")
        print("    # First install ROCm (see: https://rocm.docs.amd.com)")
        print("    # Then install CuPy for ROCm:")
        print("    pip install cupy-rocm-6-0  # match your ROCm version")
        print()
        print("  Then set backend: gpu in your workflow YAML:")
        print("    backend: gpu    # or: cuda / rocm / auto")
        print()
    else:
        print(f"  Ready to use. Set backend: gpu in workflow YAML.")
        print(f"  Or use Python: MultiGroupDiffusionSolver(..., backend='gpu')")


def main():
    parser = argparse.ArgumentParser(
        description="Core Solver Demo — modular reactor physics framework",
        prog="core-solver-demo",
    )
    sub = parser.add_subparsers(dest="command", help="Commands")

    # run
    p_run = sub.add_parser("run", help="Execute a workflow YAML")
    p_run.add_argument("workflow", help="Path to workflow YAML file")

    # validate
    p_val = sub.add_parser("validate", help="Validate a workflow YAML")
    p_val.add_argument("workflow", help="Path to workflow YAML file")

    # download
    p_dl = sub.add_parser("download", help="Download nuclear data (TENDL/JENDL)")
    p_dl.add_argument("--source", default="tendl",
                      choices=["tendl", "jendl"],
                      help="Data source (default: tendl)")
    p_dl.add_argument("--dir", default="data/nuclear_library",
                      help="Target directory")

    # gpu-status
    p_gpu = sub.add_parser("gpu-status", help="Check GPU acceleration status")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "gpu-status":
        cmd_gpu_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
