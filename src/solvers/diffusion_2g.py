"""
Two-group neutron diffusion eigenvalue solver.

Solves the coupled system:
    -∇·D₀∇φ₀ + (Σa₀ + Σs₀→₁) φ₀ = (1/k) χ₀ (νΣf₀ φ₀ + νΣf₁ φ₁)
    -∇·D₁∇φ₁ + Σa₁ φ₁ - Σs₀→₁ φ₀ = (1/k) χ₁ (νΣf₀ φ₀ + νΣf₁ φ₁)

using block Gauss-Seidel power iteration with CMFD-like acceleration.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Optional

from ..core.mesh import Mesh
from ..core.xs import XSProvider, TwoGroupXS, CrossSection
from ..core.result import Result


class TwoGroupDiffusionSolver:
    """Two-group diffusion eigenvalue solver.

    Uses a block Gauss-Seidel approach: solve fast group, then thermal group,
    then update eigenvalue (power iteration).
    """

    def __init__(
        self,
        mesh: Mesh,
        xs_provider: XSProvider,
        tolerance: float = 1e-8,
        max_iter: int = 500,
        initial_flux: np.ndarray | None = None,  # shape (2, n) or (n, 2)
        initial_keff: float = 1.0,
    ):
        self.mesh = mesh
        self.xs_provider = xs_provider
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.initial_flux = initial_flux
        self.initial_keff = initial_keff

        # Cross-section cache: cell_id → TwoGroupXS fields
        self._xs_map: dict[int, dict] = {}

    def _fetch_xs(self):
        """Retrieve 2-group cross sections for all cells."""
        self._xs_map.clear()
        for cell in self.mesh.cells:
            xs = self.xs_provider.get_xs(cell.material_id, cell.state)
            if isinstance(xs, TwoGroupXS):
                self._xs_map[cell.id] = {
                    "D": xs.D,
                    "Sa": xs.Sigma_a,
                    "nSf": xs.nuSigma_f,
                    "kSf": xs.kappaSigma_f,
                    "Ss": xs.Sigma_s,
                    "chi": xs.chi,
                }
            else:
                # 1-group → promote to 2-group (both groups same, no coupling)
                xs1 = xs
                self._xs_map[cell.id] = {
                    "D": [xs1.D, xs1.D],
                    "Sa": [xs1.Sigma_a, xs1.Sigma_a],
                    "nSf": [xs1.nuSigma_f, 0.0],
                    "kSf": [xs1.kappaSigma_f, 0.0],
                    "Ss": [[0.0, 0.0], [0.0, 0.0]],
                    "chi": [1.0, 0.0],
                }

    def _build_operator(self, group: int) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Build the leakage+absorption matrix for one group.

        Returns (A_g, M_g) where A_g contains leakage + absorption + downscatter removal,
        and M_g contains the coupling from the other group.
        """
        n = self.mesh.n_cells()
        A = sparse.lil_matrix((n, n))

        for cell in self.mesh.cells:
            i = cell.id
            D_g = self._xs_map[i]["D"][group]
            Sa_g = self._xs_map[i]["Sa"][group]
            Ss = self._xs_map[i]["Ss"]

            # Within-group removal
            removal = Sa_g * cell.volume
            if group == 0:
                # Fast group: add downscatter removal Σs₀→₁
                removal += Ss[0][1] * cell.volume
            A[i, i] += removal

        # Leakage
        for face in self.mesh.faces:
            L = face.left_cell
            R = face.right_cell
            area = face.area
            dx = face.distance

            if R is None and L is not None:
                if face.boundary_type == "vacuum":
                    D_L = self._xs_map[L]["D"][group]
                    coupling = 2.0 * D_L * area / (dx + 4.0 * D_L)
                    A[L, L] += coupling
            elif L is None and R is not None:
                if face.boundary_type == "vacuum":
                    D_R = self._xs_map[R]["D"][group]
                    coupling = 2.0 * D_R * area / (dx + 4.0 * D_R)
                    A[R, R] += coupling
            elif L is not None and R is not None:
                D_L = self._xs_map[L]["D"][group]
                D_R = self._xs_map[R]["D"][group]
                D_avg = 2.0 / (1.0 / D_L + 1.0 / D_R) if D_L > 0 and D_R > 0 else 0.0
                coupling = D_avg * area / dx
                A[L, L] += coupling
                A[R, R] += coupling
                A[L, R] -= coupling
                A[R, L] -= coupling

        # Build fission matrix
        F_g = sparse.lil_matrix((n, n))
        for cell in self.mesh.cells:
            i = cell.id
            F_g[i, i] = self._xs_map[i]["nSf"][group] * cell.volume

        return A.tocsr(), F_g.tocsr()

    def solve(self) -> Result:
        """Solve two-group eigenvalue problem."""
        self._fetch_xs()
        n = self.mesh.n_cells()

        A0, F0 = self._build_operator(0)
        A1, F1 = self._build_operator(1)

        # Build downscatter coupling matrix S (φ₀ → source for φ₁)
        S = sparse.lil_matrix((n, n))
        for cell in self.mesh.cells:
            i = cell.id
            Ss01 = self._xs_map[i]["Ss"][0][1]
            S[i, i] = Ss01 * cell.volume
        S = S.tocsr()

        # Initial flux
        if self.initial_flux is not None:
            flux_in = self.initial_flux
            if flux_in.ndim == 2 and flux_in.shape[0] == 2:
                phi0 = flux_in[0].copy()
                phi1 = flux_in[1].copy()
            elif flux_in.ndim == 2 and flux_in.shape[1] == 2:
                phi0 = flux_in[:, 0].copy()
                phi1 = flux_in[:, 1].copy()
            else:
                phi0 = flux_in.copy()
                phi1 = flux_in.copy()
        else:
            phi0 = np.ones(n, dtype=float)
            phi1 = np.ones(n, dtype=float)

        keff = self.initial_keff
        residual_history = []
        iterations = 0
        converged = False

        for iteration in range(self.max_iter):
            iterations += 1

            # --- Fission source (NOT divided by keff for Rayleigh quotient) ---
            fission_source = F0.dot(phi0) + F1.dot(phi1)

            # --- Fast group solve ---
            chi0 = np.array([self._xs_map[i]["chi"][0] for i in range(n)])
            source0 = chi0 * fission_source
            phi0_new = spsolve(A0, source0)

            # --- Thermal group solve ---
            chi1 = np.array([self._xs_map[i]["chi"][1] for i in range(n)])
            source1 = chi1 * fission_source + S.dot(phi0_new)
            phi1_new = spsolve(A1, source1)

            # --- Update keff (Rayleigh quotient: k = <Fφ_new> / <Fφ_old>) ---
            fission_source_new = F0.dot(phi0_new) + F1.dot(phi1_new)
            keff_new = fission_source_new.sum() / fission_source.sum()

            # --- Normalize ---
            total = phi0_new.sum() + phi1_new.sum()
            phi0_new = phi0_new / total
            phi1_new = phi1_new / total

            # --- Convergence check ---
            residual = (np.linalg.norm(phi0_new - phi0) +
                        np.linalg.norm(phi1_new - phi1)) / \
                       (np.linalg.norm(phi0_new) + np.linalg.norm(phi1_new) + 1e-16)
            residual_history.append(float(residual))

            phi0 = phi0_new
            phi1 = phi1_new
            keff = keff_new

            if residual < self.tolerance:
                converged = True
                break

        # Total flux and power
        total_flux = phi0 + phi1
        power = np.zeros(n)
        for i in range(n):
            kSf0 = self._xs_map[i]["kSf"][0]
            kSf1 = self._xs_map[i]["kSf"][1]
            power[i] = (kSf0 * phi0[i] + kSf1 * phi1[i]) * self.mesh.cells[i].volume
        if power.sum() > 0:
            power = power / power.sum()

        # Store 2-group flux as (2, n) array
        flux_2g = np.stack([phi0, phi1])

        warnings = []
        if not converged:
            warnings.append(f"Not converged after {iterations} iterations.")

        return Result(
            case_name="",
            solver="steady_diffusion_2g",
            keff=float(keff),
            flux=flux_2g,
            power=power,
            converged=converged,
            iterations=iterations,
            residual_history=residual_history,
            warnings=warnings,
            xs_source=self.xs_provider.get_source_info(),
        )
