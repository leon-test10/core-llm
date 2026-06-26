"""
Generic Multi-Group cross-section data structure.

Replaces the separate OneGroupXS / TwoGroupXS with a unified
representation that works for any number of energy groups.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class MultiGroupXS:
    """Macroscopic cross sections for G energy groups.

    Group indexing convention:
      - Group 0 = fastest (highest energy)
      - Group G-1 = thermal (lowest energy)
      - Scattering: Sigma_s[from_group][to_group]

    This replaces OneGroupXS and TwoGroupXS with a single unified type.
    """

    n_groups: int
    D: np.ndarray                 # (G,) diffusion coefficients [cm]
    Sigma_a: np.ndarray           # (G,) absorption [cm⁻¹]
    nuSigma_f: np.ndarray         # (G,) nu * fission production [cm⁻¹]
    kappaSigma_f: np.ndarray      # (G,) kappa * fission [energy units × cm⁻¹]
    chi: np.ndarray               # (G,) fission spectrum (Σχ_g = 1)
    Sigma_s: np.ndarray           # (G, G) scattering kernel [cm⁻¹]
    Sigma_t: np.ndarray | None = None  # (G,) total XS, computed if None

    def __post_init__(self):
        # Ensure arrays are numpy
        for field_name in ("D", "Sigma_a", "nuSigma_f", "kappaSigma_f",
                           "chi", "Sigma_s"):
            val = getattr(self, field_name)
            if not isinstance(val, np.ndarray):
                setattr(self, field_name, np.array(val, dtype=float))

        if self.Sigma_t is None:
            # Σ_t = Σ_a + Σ_{g'} Σ_s(g→g')
            self.Sigma_t = self.Sigma_a + self.Sigma_s.sum(axis=1)

    @classmethod
    def from_1g(cls, D: float, Sigma_a: float, nuSigma_f: float,
                kappaSigma_f: float = 1.0, Sigma_s: float = 0.0) -> "MultiGroupXS":
        """Create from 1-group parameters."""
        return cls(
            n_groups=1,
            D=np.array([D]),
            Sigma_a=np.array([Sigma_a]),
            nuSigma_f=np.array([nuSigma_f]),
            kappaSigma_f=np.array([kappaSigma_f]),
            chi=np.array([1.0]),
            Sigma_s=np.array([[Sigma_s]]),
        )

    @classmethod
    def from_2g(cls, D: list, Sigma_a: list, nuSigma_f: list,
                kappaSigma_f: list | None = None,
                Sigma_s: list | None = None,
                chi: list | None = None) -> "MultiGroupXS":
        """Create from 2-group parameters.

        Sigma_s: [[Σs00, Σs01], [Σs10, Σs11]]
        chi: [χ0, χ1]
        """
        if kappaSigma_f is None:
            kappaSigma_f = [1.0, 1.0]
        if chi is None:
            chi = [1.0, 0.0]
        if Sigma_s is None:
            Sigma_s = [[0.0, 0.0], [0.0, 0.0]]
        return cls(
            n_groups=2,
            D=np.array(D),
            Sigma_a=np.array(Sigma_a),
            nuSigma_f=np.array(nuSigma_f),
            kappaSigma_f=np.array(kappaSigma_f),
            chi=np.array(chi),
            Sigma_s=np.array(Sigma_s),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "MultiGroupXS":
        """Create from a dict of lists/arrays. Infers n_groups from D length.

        Supported keys:
          D, Sigma_a, nuSigma_f               — required
          kappaSigma_f, chi, Sigma_s          — optional with sensible defaults
          Sigma_t                             — optional, auto-computed

        For 2-group YAML compatibility, also accepts:
          Sigma_s as [[S00,S01],[S10,S11]]  → 2-group scattering matrix
        """
        D_arr = np.array(data["D"], dtype=float)
        G = len(D_arr)

        Sigma_a_arr = np.array(data["Sigma_a"], dtype=float)
        nuSigma_f_arr = np.array(data["nuSigma_f"], dtype=float)

        kappa_arr = data.get("kappaSigma_f", [1.0] * G)
        if isinstance(kappa_arr, list):
            kappa_arr = np.array(kappa_arr, dtype=float)

        chi_arr = data.get("chi", None)
        if chi_arr is None:
            chi_arr = np.zeros(G)
            chi_arr[0] = 1.0  # all fission in fastest group
        else:
            chi_arr = np.array(chi_arr, dtype=float)

        Sigma_s_arr = data.get("Sigma_s", None)
        if Sigma_s_arr is None:
            Sigma_s_arr = np.zeros((G, G))
        else:
            Sigma_s_arr = np.array(Sigma_s_arr, dtype=float)
            if Sigma_s_arr.ndim == 1:
                # 1D → diag only (within-group scattering)
                diag = Sigma_s_arr.copy()
                Sigma_s_arr = np.diag(diag)
            elif Sigma_s_arr.shape != (G, G):
                raise ValueError(
                    f"Sigma_s shape {Sigma_s_arr.shape} does not match "
                    f"n_groups={G}. Expected ({G}, {G})"
                )

        Sigma_t_arr = data.get("Sigma_t", None)
        if Sigma_t_arr is not None:
            Sigma_t_arr = np.array(Sigma_t_arr, dtype=float)

        return cls(
            n_groups=G,
            D=D_arr,
            Sigma_a=Sigma_a_arr,
            nuSigma_f=nuSigma_f_arr,
            kappaSigma_f=kappa_arr,
            chi=chi_arr,
            Sigma_s=Sigma_s_arr,
            Sigma_t=Sigma_t_arr,
        )

    def to_1g(self) -> "MultiGroupXS":
        """Collapse to 1-group using flat flux weighting (unweighted)."""
        return MultiGroupXS(
            n_groups=1,
            D=np.array([self.D.mean()]),
            Sigma_a=np.array([self.Sigma_a.sum()]),
            nuSigma_f=np.array([self.nuSigma_f.sum()]),
            kappaSigma_f=np.array([self.kappaSigma_f.sum()]),
            chi=np.array([1.0]),
            Sigma_s=np.array([[0.0]]),
        )

    def validate(self) -> list[str]:
        """Return list of validation issues (empty = valid)."""
        issues = []
        G = self.n_groups
        for name, arr in [("D", self.D), ("Sigma_a", self.Sigma_a),
                          ("nuSigma_f", self.nuSigma_f), ("kappaSigma_f", self.kappaSigma_f),
                          ("chi", self.chi)]:
            if len(arr) != G:
                issues.append(f"{name} length {len(arr)} != n_groups {G}")
        if self.Sigma_s.shape != (G, G):
            issues.append(f"Sigma_s shape {self.Sigma_s.shape} != ({G}, {G})")
        if np.any(self.D < 0):
            issues.append("Negative diffusion coefficient")
        if np.any(self.Sigma_a < 0):
            issues.append("Negative absorption XS")
        if np.any(self.nuSigma_f < 0):
            issues.append("Negative fission XS")
        return issues

    def __repr__(self) -> str:
        return (f"MultiGroupXS(n_groups={self.n_groups}, "
                f"D={self.D.tolist()}, "
                f"Σa={self.Sigma_a.tolist()}, "
                f"νΣf={self.nuSigma_f.tolist()})")
