"""
General N-group neutron diffusion eigenvalue solver.

Solves the G-group coupled system using block Gauss-Seidel power iteration:

  For each group g = 0 ... G-1:
    -∇·D_g∇φ_g + Σ_rem,g φ_g = χ_g S_f + Σ_{g'≠g} Σ_s(g'→g) φ_g'

  where:
    Σ_rem,g = Σ_a,g + Σ_{g'} Σ_s(g→g')   (total removal)
    S_f = Σ_{g'} νΣ_f,g' φ_g'            (fission source)

The discretization uses cell-centered finite volume on the provided Mesh.
"""

from typing import Optional
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ..core.mesh import Mesh
from ..core.xs_general import MultiGroupXS
from ..core.result import Result


class MultiGroupDiffusionSolver:
    """General N-group diffusion eigenvalue solver.

    Supports 1 to any number of groups. Uses block Gauss-Seidel
    within each power iteration.
    """

    def __init__(
        self,
        mesh: Mesh,
        xs_provider,  # XSProvider — duck-typed, must provide get_xs() returning MultiGroupXS
        tolerance: float = 1e-8,
        max_iter: int = 500,
        initial_flux: np.ndarray | None = None,  # shape: (G, n_cells) or (n_cells, G)
        initial_keff: float = 1.0,
        use_gs_acceleration: bool = True,
    ):
        self.mesh = mesh
        self.xs_provider = xs_provider
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.initial_flux = initial_flux
        self.initial_keff = initial_keff
        self.use_gs_acceleration = use_gs_acceleration

        # Cache: cell_id → MultiGroupXS
        self._xs_map: dict[int, MultiGroupXS] = {}
        self._n_groups: int = 0

        # Per-group operators (built once, same sparsity pattern for all groups)
        self._A_g: list[sparse.csr_matrix] = []   # removal + leakage per group

    def _fetch_xs(self):
        """Retrieve and validate multi-group cross sections for all cells."""
        self._xs_map.clear()

        # Get first cell's XS to determine n_groups
        first_xs = self.xs_provider.get_xs(self.mesh.cells[0].material_id,
                                            self.mesh.cells[0].state)
        if isinstance(first_xs, MultiGroupXS):
            self._n_groups = first_xs.n_groups
        else:
            # Legacy: OneGroupXS or TwoGroupXS — convert
            from ..core.xs import OneGroupXS, TwoGroupXS
            if isinstance(first_xs, OneGroupXS):
                first_xs = MultiGroupXS.from_1g(
                    first_xs.D, first_xs.Sigma_a, first_xs.nuSigma_f,
                    first_xs.kappaSigma_f, first_xs.Sigma_s)
            elif isinstance(first_xs, TwoGroupXS):
                Sigma_s = np.array(first_xs.Sigma_s)
                first_xs = MultiGroupXS.from_2g(
                    first_xs.D, first_xs.Sigma_a, first_xs.nuSigma_f,
                    first_xs.kappaSigma_f, Sigma_s, first_xs.chi)
            self._n_groups = first_xs.n_groups

        for cell in self.mesh.cells:
            raw = self.xs_provider.get_xs(cell.material_id, cell.state)
            if not isinstance(raw, MultiGroupXS):
                from ..core.xs import OneGroupXS, TwoGroupXS
                if isinstance(raw, OneGroupXS):
                    raw = MultiGroupXS.from_1g(
                        raw.D, raw.Sigma_a, raw.nuSigma_f,
                        raw.kappaSigma_f, raw.Sigma_s)
                elif isinstance(raw, TwoGroupXS):
                    Sigma_s = np.array(raw.Sigma_s)
                    raw = MultiGroupXS.from_2g(
                        raw.D, raw.Sigma_a, raw.nuSigma_f,
                        raw.kappaSigma_f, Sigma_s, raw.chi)
            self._xs_map[cell.id] = raw

    def _build_group_operator(self, group: int) -> sparse.csr_matrix:
        """Build A_g = leakage + removal for group g.

        Removal = Σ_a,g + Σ_{g'} Σ_s(g→g') — everything that takes
        neutrons OUT of group g.

        Returns sparse CSR matrix of size (n_cells, n_cells).
        """
        n = self.mesh.n_cells()
        A = sparse.lil_matrix((n, n))

        # Diagonal: absorption + total out-scattering removal (NOT self-scatter)
        for cell in self.mesh.cells:
            i = cell.id
            xs = self._xs_map[i]
            # Total removal from group g:
            #   absorption + sum of scattering out of g (excluding g→g)
            removal = xs.Sigma_a[group]
            removal += xs.Sigma_s[group, :].sum() - xs.Sigma_s[group, group]
            A[i, i] += removal * cell.volume

        # Leakage (face coupling)
        for face in self.mesh.faces:
            L = face.left_cell
            R = face.right_cell
            area = face.area
            dx = face.distance

            if R is None and L is not None:
                if face.boundary_type == "vacuum":
                    D_L = self._xs_map[L].D[group]
                    coupling = 2.0 * D_L * area / (dx + 4.0 * D_L)
                    A[L, L] += coupling
            elif L is None and R is not None:
                if face.boundary_type == "vacuum":
                    D_R = self._xs_map[R].D[group]
                    coupling = 2.0 * D_R * area / (dx + 4.0 * D_R)
                    A[R, R] += coupling
            elif L is not None and R is not None:
                D_L = self._xs_map[L].D[group]
                D_R = self._xs_map[R].D[group]
                D_avg = 2.0 / (1.0 / D_L + 1.0 / D_R) if D_L > 0 and D_R > 0 else 0.0
                coupling = D_avg * area / dx
                A[L, L] += coupling
                A[R, R] += coupling
                A[L, R] -= coupling
                A[R, L] -= coupling

        return A.tocsr()

    def solve(self) -> Result:
        """Solve the G-group eigenvalue problem."""
        self._fetch_xs()
        n = self.mesh.n_cells()
        G = self._n_groups

        # Build per-group operators (sparsity pattern same for all groups)
        self._A_g = [self._build_group_operator(g) for g in range(G)]

        # Build diagonal fission operator per group (F_g)_{ii} = νΣ_f,g_i * V_i
        F_g = []
        for g in range(G):
            Fg = sparse.lil_matrix((n, n))
            for cell in self.mesh.cells:
                i = cell.id
                Fg[i, i] = self._xs_map[i].nuSigma_f[g] * cell.volume
            F_g.append(Fg.tocsr())

        # Build inscattering source matrices S_{g'→g}: diag(Σ_s(g'→g)_i * V_i)
        S = {}
        for g_from in range(G):
            for g_to in range(G):
                if g_from == g_to:
                    continue
                s_val = self._xs_map[self.mesh.cells[0].id].Sigma_s[g_from, g_to]
                # Check if any cell has non-zero scattering between these groups
                has_scatter = any(
                    self._xs_map[c.id].Sigma_s[g_from, g_to] > 0
                    for c in self.mesh.cells
                )
                if has_scatter:
                    Sgg = sparse.lil_matrix((n, n))
                    for cell in self.mesh.cells:
                        i = cell.id
                        val = self._xs_map[i].Sigma_s[g_from, g_to] * cell.volume
                        if val > 0:
                            Sgg[i, i] = val
                    S[(g_from, g_to)] = Sgg.tocsr()

        # Initial flux: shape (G, n)
        if self.initial_flux is not None:
            if self.initial_flux.ndim == 2:
                if self.initial_flux.shape[0] == G:
                    phi = self.initial_flux.copy()
                elif self.initial_flux.shape[1] == G:
                    phi = self.initial_flux.T.copy()
                else:
                    raise ValueError(f"Initial flux shape {self.initial_flux.shape} "
                                     f"incompatible with G={G}")
            else:
                phi = np.tile(self.initial_flux, (G, 1))
        else:
            phi = np.ones((G, n), dtype=float)

        keff = self.initial_keff
        residual_history = []
        iterations = 0
        converged = False

        for iteration in range(self.max_iter):
            iterations += 1

            # Fission source: S_f = Σ_g νΣ_f,g φ_g (cell-wise)
            fission_source = np.zeros(n)
            for g in range(G):
                fission_source += F_g[g].dot(phi[g])

            phi_old = phi.copy()
            phi_new = np.zeros((G, n))

            # Block Gauss-Seidel: solve each group sequentially
            for g in range(G):
                # Source for group g:
                #   source_g = χ_g * S_f + Σ_{g'≠g} Σ_s(g'→g) φ_{g'}  (using latest φ)
                source = np.zeros(n)

                # Fission contribution
                for i in range(n):
                    xi = self._xs_map[i].chi[g]
                    if xi > 0:
                        source[i] += xi * fission_source[i]

                # Inscattering from other groups (use phi_new for already-solved groups)
                for g_from in range(G):
                    if g_from == g:
                        continue
                    key = (g_from, g)
                    if key in S:
                        # Use phi_new if g_from < g (already updated), else phi_old
                        phi_src = phi_new[g_from] if g_from < g else phi_old[g_from]
                        source += S[key].dot(phi_src)

                # Solve A_g φ_g = source
                phi_new[g] = spsolve(self._A_g[g], source)

            # Update keff: ratio of new to old fission source integral
            fission_new = np.zeros(n)
            for g in range(G):
                fission_new += F_g[g].dot(phi_new[g])
            keff_new = fission_new.sum() / fission_source.sum() if fission_source.sum() > 0 else 0.0

            # Normalize: total flux sum = 1
            total = phi_new.sum()
            if total > 0:
                phi_new = phi_new / total

            # Convergence: relative change in flux shape
            residual = np.linalg.norm(phi_new - phi_old) / (
                np.linalg.norm(phi_new) + 1e-16)
            residual_history.append(float(residual))

            phi = phi_new
            keff = keff_new

            if residual < self.tolerance:
                converged = True
                break

        # Compute power
        power = np.zeros(n)
        for g in range(G):
            for i in range(n):
                power[i] += (self._xs_map[i].kappaSigma_f[g] *
                             phi[g, i] * self.mesh.cells[i].volume)
        if power.sum() > 0:
            power = power / power.sum()

        # Total flux (sum over groups)
        total_flux = phi.sum(axis=0)

        warnings = []
        if not converged:
            warnings.append(f"Not converged after {iterations} iterations. "
                           f"Final residual: {residual_history[-1]:.2e}")

        return Result(
            case_name="",
            solver=f"steady_diffusion_{G}g",
            keff=float(keff),
            flux=phi,  # shape (G, n)
            power=power,
            converged=converged,
            iterations=iterations,
            residual_history=residual_history,
            warnings=warnings,
            xs_source=self.xs_provider.get_source_info(),
        )
