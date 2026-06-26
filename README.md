# Core Solver Demo

A modular reactor physics framework demonstrating:

- **Arbitrary material registration** — materials are decoupled from solvers
- **Multiple cross-section providers** — constant (toy), tabulated, nuclear data library (TENDL/JENDL)
- **Multiple solvers** — 1-group and 2-group diffusion eigenvalue
- **Skill-based workflow** — compose YAML workflows manually or via LLM
- **Nuclear data adapter** — download and parse real ENDF-6 data

> ⚠️ **Disclaimer**: This is a demo for algorithm validation and education.
> Cross sections are toy/approximate unless real ENDF data is downloaded.
> Not for engineering safety analysis.

## Quick Start

```bash
# Install dependencies
pip install numpy scipy pyyaml

# Run the toy demo
cd core-solver-demo
python -m src.cli run examples/workflow_steady_diffusion.yaml

# Run with control rod inserted
python -m src.cli run examples/workflow_control_rod.yaml

# Run with nuclear data library (builtin approximate XS)
python -m src.cli run examples/workflow_nuclear_library.yaml

# Download real nuclear data (TENDL-2023)
python -m src.cli download --source=tendl --dir=data/nuclear_library
```

## Output

After running, check `outputs/`:
- `*.json` — results (keff, iterations, warnings)
- `*_flux.npy` — neutron flux distribution
- `*_power.npy` — power distribution  
- `*_report.md` — human-readable report

## Architecture

```
examples/workflow.yaml
        │
        ▼
WorkflowEngine (src/workflows/engine.py)
        │
   ┌────┼────┬──────────┐
   ▼    ▼    ▼          ▼
 Mesh  Mat  XSProvider  Solver
       │       │          │
       │   ┌───┼────┐     │
       │   │   │    │     │
       │ constant tabulated nuclear
       │   │   │    │     │
       │  (toy) (table) (ENDF)
       │                    │
       └────────────────────┘
                   │
              Postprocess
                   │
              Report
```

## Project Structure

```
core-solver-demo/
├── examples/          # YAML cases and workflows
├── src/
│   ├── core/          # Data structures: mesh, material, xs, result
│   ├── solvers/       # Diffusion solvers (1g, 2g), postprocess
│   ├── workflows/     # YAML parser, validator, engine
│   └── cli.py         # CLI: run, validate, download
├── data/
│   └── nuclear_library/  # Downloaded ENDF files (TENDL/JENDL)
├── tests/
└── outputs/           # Results (*.json, *.npy, *.md)
```

## Physics Models

### 1-Group Diffusion

```
-∇·D∇φ + Σa φ = (1/k) νΣf φ
```

Finite-volume discretization with power iteration eigenvalue solver.

### 2-Group Diffusion

```
-∇·D₀∇φ₀ + (Σa₀+Σs₀→₁)φ₀ = (1/k) χ₀ (νΣf₀φ₀+νΣf₁φ₁)
-∇·D₁∇φ₁ + Σa₁φ₁ = Σs₀→₁φ₀ + (1/k) χ₁ (...)
```

Block Gauss-Seidel power iteration.

## Cross-Section Sources

| Source | Accuracy | Use Case |
|--------|----------|----------|
| `constant` | Toy | Algorithm testing |
| `tabulated` | Interpolated | State-dependent demo |
| `nuclear_library` (builtin) | Rough (~20%) | Quick demo with real nuclide IDs |
| `nuclear_library` (TENDL download) | Engineering (~5-10%) | Better accuracy |
| `nuclear_library` (JENDL download) | Engineering (~5-10%) | Alternative source |

## License

MIT — for education and research.
