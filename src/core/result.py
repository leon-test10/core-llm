"""
Result containers and report generation.

Results carry metadata about how they were produced (solver, xs_source, etc.)
so that downstream consumers can distinguish toy from validated results.
"""

from dataclasses import dataclass, field
from typing import Any
import json
import numpy as np


@dataclass
class Result:
    """Container for solver results with provenance metadata."""
    case_name: str
    solver: str
    keff: float
    flux: np.ndarray
    converged: bool
    iterations: int
    residual_history: list[float] = field(default_factory=list)
    power: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)
    xs_source: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "case_name": self.case_name,
            "solver": self.solver,
            "keff": float(self.keff),
            "iterations": self.iterations,
            "converged": self.converged,
            "residual_final": float(self.residual_history[-1]) if self.residual_history else None,
            "max_power": float(self.power.max()) if self.power is not None else None,
            "min_power": float(self.power.min()) if self.power is not None else None,
            "xs_source": self.xs_source,
            "warnings": self.warnings,
        }

    def save(self, path: str):
        """Save results as JSON + numpy arrays."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        d = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        # Save arrays alongside
        base = path.replace(".json", "")
        np.save(f"{base}_flux.npy", self.flux)
        if self.power is not None:
            np.save(f"{base}_power.npy", self.power)
        if self.residual_history:
            np.save(f"{base}_residual.npy", np.array(self.residual_history))


@dataclass
class Report:
    """Markdown report generator."""
    result: Result
    mesh_info: dict = field(default_factory=dict)
    material_info: dict = field(default_factory=dict)

    def generate(self) -> str:
        lines = []
        lines.append("# Calculation Report")
        lines.append("")
        lines.append("## Case")
        lines.append(f"- **name**: {self.result.case_name}")
        lines.append(f"- **solver**: {self.result.solver}")
        lines.append(f"- **xs source**: {self.result.xs_source}")
        if self.mesh_info:
            lines.append(f"- **mesh**: {self.mesh_info.get('type', 'unknown')} "
                        f"{self.mesh_info.get('n_cells', '?')} cells")
        lines.append("")

        lines.append("## Results")
        lines.append(f"- **k_eff**: {self.result.keff:.6f}")
        lines.append(f"- **iterations**: {self.result.iterations}")
        lines.append(f"- **converged**: {self.result.converged}")
        if self.result.residual_history:
            lines.append(f"- **final residual**: {self.result.residual_history[-1]:.2e}")
        if self.result.power is not None:
            lines.append(f"- **max power**: {self.result.power.max():.4f}")
            lines.append(f"- **min power**: {self.result.power.min():.4f}")
        lines.append("")

        if self.result.warnings:
            lines.append("## Warnings")
            for w in self.result.warnings:
                lines.append(f"- {w}")
            lines.append("")

        # Disclaimer
        lines.append("## Notes")
        if "toy" in self.result.xs_source or "constant" in self.result.xs_source:
            lines.append("[WARNING] This result uses **toy fixed cross sections** "
                         "and is **not valid for engineering analysis**.")
        elif "builtin" in self.result.xs_source:
            lines.append("[WARNING] This result uses **builtin approximate cross sections**. "
                         "Accuracy is limited. Use downloaded ENDF data for better results.")
        elif "TENDL" in self.result.xs_source or "JENDL" in self.result.xs_source:
            lines.append("[INFO] Cross sections from evaluated nuclear data library. "
                         "Always validate against experimental benchmarks.")
        lines.append("")

        return "\n".join(lines)

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.generate())
