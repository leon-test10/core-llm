"""
Single-group neutron diffusion eigenvalue solver.

Solves:
    -∇·D∇φ + Σa φ = (1/k) νΣf φ

using finite-volume discretization on a Mesh + power iteration.

Boundary conditions:
  - vacuum: φ(extrapolated) = 0 → φ_boundary = 0
  - reflective: J = 0 → zero-gradient (implemented as no leakage across face)

The discretized system is:
    A φ = (1/k) F φ

where A = M + diag(Σa_i · V_i), F = diag(νΣf_i · V_i),
and M is the leakage matrix built from face connections.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Optional

from ..core.mesh import Mesh, Face
from ..core.xs import XSProvider, OneGroupXS, CrossSection
from ..core.result import Result


class OneGroupDiffusionSolver:
    """Single-group diffusion eigenvalue solver with power iteration."""

    def __init__(
        self,
        mesh: Mesh,
        xs_provider: XSProvider,
        tolerance: float = 1e-8,
        max_iter: int = 500,
        initial_flux: np.ndarray | None = None,
        initial_keff: float = 1.0,
        boundary_condition: str = "vacuum",
    ):
        self.mesh = mesh
        self.xs_provider = xs_provider
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.initial_flux = initial_flux
        self.initial_keff = initial_keff
        self.boundary_condition = boundary_condition

        self._A: sparse.csr_matrix | None = None
        self._F: sparse.csr_matrix | None = None
        self._xs_map: dict[int, tuple[float, float, float, float, float]] = {}  # cell_id → (D, Σa, νΣf, κΣf, Σs)

    def _fetch_xs(self):
        """Retrieve cross sections for all cells."""
        self._xs_map.clear()
        for cell in self.mesh.cells:
            state = cell.state
            xs = self.xs_provider.get_xs(cell.material_id, state)
            if isinstance(xs, OneGroupXS):
                self._xs_map[cell.id] = (xs.D, xs.Sigma_a, xs.nuSigma_f,
                                          xs.kappaSigma_f, xs.Sigma_s)
            else:
                # 2-group → extract fast-group values as approximate 1-group
                xs2 = xs
                D = xs2.D[0]
                Sa = xs2.Sigma_a[0]
                nSf = xs2.nuSigma_f[0]
                kSf = xs2.kappaSigma_f[0]
                self._xs_map[cell.id] = (D, Sa, nSf, kSf, 0.0)

    def assemble_matrices(self):
        """Build A and F sparse matrices.

        Leakage term for face between cell L and R:
            J = -D_avg * (φ_R - φ_L) / dx
            where D_avg = 2 / (1/D_L + 1/D_R)  (harmonic mean)
        """
        n = self.mesh.n_cells()
        A = sparse.lil_matrix((n, n))
        F = sparse.lil_matrix((n, n))

        for cell in self.mesh.cells:
            i = cell.id
            D_i, Sa_i, nSf_i, kSf_i, Ss_i = self._xs_map[i]

            # Absorption + removal
            A[i, i] += Sa_i * cell.volume

            # Fission source
            F[i, i] = nSf_i * cell.volume

        # Leakage through faces
        for face in self.mesh.faces:
            L = face.left_cell
            R = face.right_cell
            area = face.area
            dx = face.distance

            if R is None:
                # Boundary face (left_cell is interior, right is boundary)
                # Vacuum: φ(R) = 0, extrapolation distance = 2.13*D typically
                # Simplified: φ(0) = 0 at extrapolated boundary, with D/dx coupling
                if face.boundary_type == "vacuum":
                    D_L = self._xs_map[L][0]
                    # Standard vacuum BC: φ(-2D) = 0 → leakage = D * φ / (dx/2 + 2D)
                    # Simplified for coarse mesh: J_out = 0.5 * φ_L / (dx/(4*D_L))
                    # Using Marshak boundary condition approximation:
                    coupling = 2.0 * D_L * area / (dx + 4.0 * D_L)
                    A[L, L] += coupling
                elif face.boundary_type == "reflective":
                    # Zero net current: no leakage term
                    pass
            elif L is None:
                # Boundary face (right_cell is interior, left is boundary)
                if face.boundary_type == "vacuum":
                    D_R = self._xs_map[R][0]
                    coupling = 2.0 * D_R * area / (dx + 4.0 * D_R)
                    A[R, R] += coupling
                elif face.boundary_type == "reflective":
                    pass
            else:
                # Interior face
                D_L = self._xs_map[L][0]
                D_R = self._xs_map[R][0]
                # Harmonic mean diffusion coefficient
                D_avg = 2.0 / (1.0 / D_L + 1.0 / D_R) if D_L > 0 and D_R > 0 else 0.0
                coupling = D_avg * area / dx
                A[L, L] += coupling
                A[R, R] += coupling
                A[L, R] -= coupling
                A[R, L] -= coupling

        self._A = A.tocsr()
        self._F = F.tocsr()

    def solve(self) -> Result:
        """Solve the eigenvalue problem using power iteration."""
        self._fetch_xs()
        self.assemble_matrices()

        n = self.mesh.n_cells()
        A = self._A
        F = self._F

        # Initial flux guess
        if self.initial_flux is not None:
            phi = self.initial_flux.copy().astype(float)
        else:
            phi = np.ones(n, dtype=float)

        keff = self.initial_keff
        residual_history = []
        iterations = 0
        converged = False

        for iteration in range(self.max_iter):
            iterations += 1

            # Source from previous iteration (F * phi)
            source = F.dot(phi)

            # Solve A φ_new = source
            phi_new = spsolve(A, source)

            # Compute new keff (Rayleigh quotient)
            keff_new = F.dot(phi_new).sum() / F.dot(phi).sum()

            # Normalize
            phi_new = phi_new / phi_new.sum()

            # Check convergence
            residual = np.linalg.norm(phi_new - phi) / np.linalg.norm(phi_new)
            residual_history.append(float(residual))

            if residual < self.tolerance:
                converged = True
                phi = phi_new
                keff = keff_new
                break

            phi = phi_new
            keff = keff_new

        # Post-process warnings
        warnings = []
        if not converged:
            warnings.append(f"Not converged after {iterations} iterations. "
                           f"Final residual: {residual_history[-1]:.2e}")
        n_fissile = sum(1 for cell in self.mesh.cells
                        if self._xs_map[cell.id][2] > 0)
        if n_fissile == 0:
            warnings.append("No fissile material detected — keff may be zero or meaningless.")
        if keff <= 0:
            warnings.append(f"Non-physical keff = {keff:.6f}")

        # Compute power
        power = np.zeros(n)
        for i in range(n):
            _, _, _, kSf, _ = self._xs_map[i]
            power[i] = kSf * phi[i] * self.mesh.cells[i].volume

        # Normalize power
        if power.sum() > 0:
            power = power / power.sum()

        return Result(
            case_name="",
            solver="steady_diffusion_1g",
            keff=float(keff),
            flux=phi,
            power=power,
            converged=converged,
            iterations=iterations,
            residual_history=residual_history,
            warnings=warnings,
            xs_source=self.xs_provider.get_source_info(),
        )
