"""
Postprocessing: power normalization, peaking factors, report generation.
"""

import numpy as np
from ..core.mesh import Mesh, RegularGrid2D
from ..core.result import Result, Report


class PowerPostprocessor:
    """Compute derived quantities from flux and cross sections."""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh

    def compute_power(self, flux: np.ndarray,
                       kappaSigma_f_map: dict[int, float]) -> np.ndarray:
        """Compute cell power from flux and κΣf.

        P_i = κΣf_i · φ_i · V_i
        """
        n = self.mesh.n_cells()
        power = np.zeros(n)
        for i in range(n):
            kSf = kappaSigma_f_map.get(i, 0.0)
            if flux.ndim == 2:
                # 2-group: sum over groups
                phi_i = flux[:, i].sum() if flux.shape[0] == 2 else flux[i].sum()
            else:
                phi_i = flux[i]
            power[i] = kSf * phi_i * self.mesh.cells[i].volume
        return power

    def normalize_power(self, power: np.ndarray, total_power: float = 1.0) -> np.ndarray:
        """Normalize power to a target total."""
        s = power.sum()
        if s > 0:
            return power * total_power / s
        return power

    def compute_peaking_factor(self, power: np.ndarray) -> float:
        """F_Q = max(P_i) / avg(P_i)."""
        avg = power.mean()
        return float(power.max() / avg) if avg > 0 else 0.0

    def power_to_grid(self, power: np.ndarray) -> np.ndarray | None:
        """Reshape 1D power array to 2D grid if mesh is RegularGrid2D."""
        if isinstance(self.mesh, RegularGrid2D):
            return power.reshape((self.mesh.ny, self.mesh.nx))
        return None


class ReportWriter:
    """Write results to JSON, NPY, and Markdown report."""

    @staticmethod
    def write(result: Result, mesh: Mesh, output_dir: str = "outputs",
              case_name: str = "default"):
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save result
        result.case_name = case_name
        result.save(os.path.join(output_dir, f"{case_name}_results.json"))

        # Save flux and power as standalone npy
        np.save(os.path.join(output_dir, f"{case_name}_flux.npy"), result.flux)
        if result.power is not None:
            np.save(os.path.join(output_dir, f"{case_name}_power.npy"), result.power)

        # Generate report
        mesh_info = mesh.to_dict() if hasattr(mesh, 'to_dict') else {"type": "unknown"}
        report = Report(result=result, mesh_info=mesh_info)
        report.save(os.path.join(output_dir, f"{case_name}_report.md"))

        # Console summary
        print(f"\n{'='*60}")
        print(f"  Case: {case_name}")
        print(f"  Solver: {result.solver}")
        print(f"  k_eff = {result.keff:.6f}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Converged: {result.converged}")
        if result.power is not None:
            print(f"  Max power: {result.power.max():.4f}")
            print(f"  Min power: {result.power.min():.4f}")
        if result.warnings:
            print(f"  Warnings: {', '.join(result.warnings)}")
        print(f"  XS source: {result.xs_source}")
        print(f"{'='*60}")
        print(f"  Output: {output_dir}/{case_name}_*.json/npy/md")
