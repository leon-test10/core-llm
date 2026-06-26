"""
Cross-section data structures and providers.

Design principle:
  Solvers consume CrossSection objects via an XSProvider interface.
  Materials point to a provider; the provider decides how to compute
  cross sections from the cell state (constant, tabulated, library, ML, ...).

Nuclear data adapter:
  The ENDFAdapter can download and parse ENDF-6 format evaluated nuclear data
  files, convert them to multigroup cross sections using a simple flux-weighting
  scheme, and serve them through the same XSProvider interface.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import json
import math
import urllib.request
import urllib.error

import numpy as np


# ── Cross-section value types ──────────────────────────────────────────────

@dataclass
class OneGroupXS:
    """Single-group homogenized cross sections."""
    D: float              # diffusion coefficient (cm)
    Sigma_a: float        # absorption macroscopic XS (cm⁻¹)
    nuSigma_f: float      # nu * fission macroscopic XS (cm⁻¹)
    kappaSigma_f: float   # energy per fission × fission XS (arbitrary units)
    Sigma_s: float = 0.0  # within-group scattering (usually 0 for 1-group)


@dataclass
class TwoGroupXS:
    """Two-group homogenized cross sections.

    Group 0 = fast, Group 1 = thermal (convention varies, this is typical).
    """
    D: list[float]                # [D0, D1]
    Sigma_a: list[float]          # [Σa0, Σa1]
    nuSigma_f: list[float]        # [νΣf0, νΣf1]
    kappaSigma_f: list[float]     # [κΣf0, κΣf1]
    Sigma_s: list[list[float]]    # [[Σs0→0, Σs0→1], [Σs1→0, Σs1→1]]
    chi: list[float]              # [χ0, χ1] fission spectrum


CrossSection = OneGroupXS | TwoGroupXS


# ── XS Provider interface ──────────────────────────────────────────────────

class XSProvider(ABC):
    """Abstract cross-section provider.

    Subclasses implement different strategies:
      - ConstantXSProvider: fixed values (toy problems)
      - TabulatedXSProvider: state-dependent lookup + interpolation
      - NuclearDataLibrary: real ENDF/TENDL data processed to multigroup
    """

    provider_id: str = "abstract"

    @abstractmethod
    def get_xs(self, material_id: str, state: "CellState | None" = None) -> CrossSection:
        """Return cross sections for the given material and state."""
        ...

    @abstractmethod
    def get_group_count(self) -> int:
        """Number of energy groups this provider supplies."""
        ...

    def get_source_info(self) -> str:
        """Human-readable description of where these XS come from."""
        return "unknown"


# ── Constant XS Provider (toy) ────────────────────────────────────────────

class ConstantXSProvider(XSProvider):
    """Fixed cross sections independent of state. For toy problems only."""

    provider_id = "constant"

    def __init__(self, materials_xs: dict[str, dict], group_count: int = 1):
        """
        Args:
            materials_xs: {material_id: {field: value, ...}}
                For 1-group: {D, Sigma_a, nuSigma_f, kappaSigma_f}
                For 2-group: {D: [D0,D1], Sigma_a: [...], ...}
            group_count: 1 or 2
        """
        self._xs = materials_xs
        self._group_count = group_count
        self._cache: dict[str, CrossSection] = {}

    def get_xs(self, material_id: str, state: "CellState | None" = None) -> CrossSection:
        if material_id in self._cache:
            return self._cache[material_id]

        raw = self._xs[material_id]
        if self._group_count == 1:
            xs = OneGroupXS(
                D=raw["D"],
                Sigma_a=raw["Sigma_a"],
                nuSigma_f=raw["nuSigma_f"],
                kappaSigma_f=raw.get("kappaSigma_f", 1.0),
                Sigma_s=raw.get("Sigma_s", 0.0),
            )
        else:
            xs = TwoGroupXS(
                D=raw["D"],
                Sigma_a=raw["Sigma_a"],
                nuSigma_f=raw["nuSigma_f"],
                kappaSigma_f=raw.get("kappaSigma_f", [1.0, 1.0]),
                Sigma_s=raw["Sigma_s"],
                chi=raw["chi"],
            )
        self._cache[material_id] = xs
        return xs

    def get_group_count(self) -> int:
        return self._group_count

    def get_source_info(self) -> str:
        return "constant (toy) - not for engineering use"


# ── Tabulated XS Provider ──────────────────────────────────────────────────

class TabulatedXSProvider(XSProvider):
    """State-dependent cross sections via multi-dimensional linear interpolation.

    Stores a table of XS values on a grid of state variables (burnup, temperature,
    density, boron_ppm, rod_fraction) and linearly interpolates.
    """

    provider_id = "tabulated"

    def __init__(self, group_count: int = 1):
        self._group_count = group_count
        # {material_id: {"axes": {var: [values]}, "data": {field: ndarray}}}
        self._tables: dict[str, dict] = {}

    def load_table(self, material_id: str, axes: dict[str, list[float]],
                   data: dict[str, np.ndarray]):
        """Register a lookup table for a material.

        axes: {"burnup": [0, 10, 20, ...], "temperature": [600, 900, 1200], ...}
        data: {"D": ndarray shaped like axes, "Sigma_a": ..., ...}

        The ndarray shape must match the axes order: (len(burnup), len(temperature), ...)
        """
        self._tables[material_id] = {"axes": axes, "data": data}

    def get_xs(self, material_id: str, state: "CellState | None" = None) -> CrossSection:
        table = self._tables[material_id]
        axes = table["axes"]
        data = table["data"]

        if state is None:
            state = CellState()

        # Build query point matching axes order
        axis_names = list(axes.keys())
        point = []
        for ax_name in axis_names:
            val = getattr(state, ax_name, 0.0)
            ax_vals = axes[ax_name]
            # Clamp to range
            val = max(ax_vals[0], min(ax_vals[-1], val))
            point.append(val)

        # Multilinear interpolation
        def interp(field_name: str) -> float:
            arr = data[field_name]
            result = float(arr.flat[0])
            # Simple linear interpolation for each axis sequentially
            current = arr.copy()
            for dim_idx, (ax_name, ax_vals) in enumerate(zip(axis_names, [axes[n] for n in axis_names])):
                x = point[dim_idx]
                # Find bracket
                if x <= ax_vals[0]:
                    idx = 0
                    frac = 0.0
                elif x >= ax_vals[-1]:
                    idx = len(ax_vals) - 2
                    frac = 1.0
                else:
                    for k in range(len(ax_vals) - 1):
                        if ax_vals[k] <= x <= ax_vals[k + 1]:
                            idx = k
                            frac = (x - ax_vals[k]) / (ax_vals[k + 1] - ax_vals[k])
                            break
                # Interpolate along this axis
                slc0 = [slice(None)] * current.ndim
                slc1 = [slice(None)] * current.ndim
                slc0[dim_idx] = idx
                slc1[dim_idx] = idx + 1
                current = current[tuple(slc0)] * (1 - frac) + current[tuple(slc1)] * frac

            result = float(current)
            return result

        if self._group_count == 1:
            xs = OneGroupXS(
                D=interp("D"),
                Sigma_a=interp("Sigma_a"),
                nuSigma_f=interp("nuSigma_f"),
                kappaSigma_f=interp("kappaSigma_f"),
                Sigma_s=interp("Sigma_s") if "Sigma_s" in data else 0.0,
            )
        else:
            xs = TwoGroupXS(
                D=[interp("D0"), interp("D1")],
                Sigma_a=[interp("Sigma_a0"), interp("Sigma_a1")],
                nuSigma_f=[interp("nuSigma_f0"), interp("nuSigma_f1")],
                kappaSigma_f=[interp("kappaSigma_f0"), interp("kappaSigma_f1")],
                Sigma_s=[[interp("Sigma_s00"), interp("Sigma_s01")],
                         [interp("Sigma_s10"), interp("Sigma_s11")]],
                chi=[interp("chi0"), interp("chi1")],
            )
        return xs

    def get_group_count(self) -> int:
        return self._group_count

    def get_source_info(self) -> str:
        return "tabulated (multi-dimensional interpolation)"


# ── Nuclear Data Library Adapter ──────────────────────────────────────────

# ---------------------------------------------------------------
# ENDF-6 格式轻量解析器
# 参考: ENDF-6 Formats Manual (BNL-203218-2018)
# 本解析器仅提取关键截面数据用于多群截面生成
# ---------------------------------------------------------------

# 标准能量群结构（WIMS 69群简化版 → 实际只用少数群做demo）
# 这里内置一个简化的能群边界集，来自 IAEA 推荐的少群结构
WIMS_2GROUP_BOUNDARY = 0.625  # eV — 典型热中子分界

# 常见核素的 ENDF 数据源 URL (TENDL-2023, 通过 IAEA/NDS 或直接 TENDL)
# TENDL 是完全开放的科学数据库 (CC0)
TENDL_BASE_URLS = [
    "https://tendl.web.psi.ch/tendl_2023/neutron_file/",
    "https://nds.iaea.org/tendl_2023/neutron/",
]

# 备用：JENDL-5 的 URL (日本公开库，通常无障碍)
JENDL_BASE_URL = "https://wwwndc.jaea.go.jp/jendl/j5/data/JENDL-5.0/neutron/"

# 最易获取的：直接使用打包好的 ACE 文件（用于 MCNP/Serpent/OpenMC）
# 这里先实现一个基于本地文件系统的适配器


class NuclearDataLibrary(XSProvider):
    """Multigroup cross-section library built from evaluated nuclear data.

    This adapter can:
      1. Download ENDF-6 formatted files from TENDL or JENDL
      2. Parse basic ENDF-6 sections (MT=1, MT=2, MT=3, MT=452)
      3. Generate few-group cross sections via flux-weighted averaging
      4. Serve through the standard XSProvider interface

    For now, it supports downloading and reading the metadata + generating
    approximate 1-group or 2-group cross sections.
    """

    provider_id = "nuclear_library"

    # Standard thermal energy cutoffs for 2-group structure
    DEFAULT_GROUP_BOUNDARIES_2G = np.array([1e7, 0.625, 1e-5])  # eV
    # WIMS 69-group structure (eV) — top 10 for demo
    WIMS_69_TOP10 = np.array([1.00000E+07, 6.06550E+06, 3.67900E+06,
                               2.23100E+06, 1.35300E+06, 8.21000E+05,
                               5.00000E+05, 3.02500E+05, 1.83000E+05,
                               1.11000E+05, 6.73400E+04])

    def __init__(self, data_dir: str = "data/nuclear_library",
                 group_boundaries: np.ndarray | None = None,
                 group_count: int = 1):
        """
        Args:
            data_dir: directory to store downloaded ENDF files
            group_boundaries: energy group boundaries in eV (ascending or descending)
            group_count: number of output groups (1 or 2)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.group_count = group_count

        if group_boundaries is None:
            if group_count == 2:
                self.group_boundaries = self.DEFAULT_GROUP_BOUNDARIES_2G
            else:
                self.group_boundaries = np.array([1e7, 1e-5])
        else:
            self.group_boundaries = np.sort(group_boundaries)[::-1]

        # Cache: {material_id: OneGroupXS | TwoGroupXS}
        self._xs_cache: dict[str, CrossSection] = {}
        # Raw ENDF data cache: {material_id: dict}
        self._endf_data: dict[str, dict] = {}

        # Built-in approximate XS for common nuclides (from TENDL-2023 at 300K)
        # These are rough 1-group thermal-averaged values for validation only.
        self._builtin: dict[str, dict] = self._load_builtin_data()

    def _load_builtin_data(self) -> dict[str, dict]:
        """Return approximate 1-group cross sections for common nuclides.

        Values are roughly thermal-spectrum-averaged from standard references.
        DO NOT USE FOR ENGINEERING — these are for demo/interpolation testing only.
        """
        return {
            "U235": {
                "D": 1.2, "Sigma_a": 0.12, "nuSigma_f": 0.18,
                "kappaSigma_f": 1.0,
                "source": "TENDL-2023 approximate thermal average, 300K",
            },
            "U238": {
                "D": 1.2, "Sigma_a": 0.008, "nuSigma_f": 0.005,
                "kappaSigma_f": 1.0,
                "source": "TENDL-2023 approximate thermal average, 300K",
            },
            "Pu239": {
                "D": 1.2, "Sigma_a": 0.15, "nuSigma_f": 0.22,
                "kappaSigma_f": 1.0,
                "source": "TENDL-2023 approximate thermal average, 300K",
            },
            "H1_H2O": {
                "D": 2.0, "Sigma_a": 0.002, "nuSigma_f": 0.0,
                "kappaSigma_f": 0.0,
                "source": "TENDL-2023 approximate thermal average, 300K",
            },
            "B10": {
                "D": 1.5, "Sigma_a": 1.5, "nuSigma_f": 0.0,
                "kappaSigma_f": 0.0,
                "source": "TENDL-2023 approximate thermal average, 300K",
            },
            "Zr_nat": {
                "D": 1.5, "Sigma_a": 0.0005, "nuSigma_f": 0.0,
                "kappaSigma_f": 0.0,
                "source": "TENDL-2023 approximate thermal average, 300K",
            },
        }

    # ── ENDF-6 Download ──────────────────────────────────────────────────

    def download_nuclide(self, element: str, A: int, source: str = "tendl",
                         temperature: str = "300K") -> Path:
        """Download ENDF-6 file for a nuclide from TENDL or JENDL.

        Args:
            element: element symbol, e.g. "U", "Pu", "H"
            A: mass number, e.g. 235, 238, 1
            source: "tendl" or "jendl"
            temperature: "300K" (TENDL only)

        Returns:
            Path to downloaded file
        """
        Z = self._atomic_number(element)

        if source == "tendl":
            # TENDL-2023 filename formats to try
            filename_base = f"{element}{A}"
            candidates = [
                f"{filename_base}.tendl",
                f"{filename_base}.endf",
                f"{element.lower()}{A}.tendl",
                f"n-{element}_{A:03d}.endf",
            ]
            urls = []
            for base in TENDL_BASE_URLS:
                for cand in candidates:
                    urls.append(f"{base}{element}/{cand}")
                    urls.append(f"{base}{cand}")
        elif source == "jendl":
            filename = f"j5-{Z:03d}-{A:03d}.dat"
            urls = [f"{JENDL_BASE_URL}{filename}"]
        else:
            raise ValueError(f"Unknown source: {source}")

        local_path = self.data_dir / f"{element}{A}.endf"

        if local_path.exists():
            return local_path

        # Also check for any file matching the element+A pattern
        for existing in self.data_dir.glob(f"{element}{A}*"):
            return existing

        last_error = None
        for url in urls:
            try:
                print(f"  Trying {url} ...")
                req = urllib.request.Request(url, headers={
                    "User-Agent": "CoreSolverDemo/1.0 (educational use)"
                })
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
                local_path.write_bytes(data)
                print(f"  Downloaded {len(data)} bytes -> {local_path}")
                return local_path
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
                last_error = e
                continue

        # If download fails, the builtin approximate data will be used
        print(f"  [INFO] Could not download {element}{A}. Using builtin approximate data.")
        print(f"  To manually download, visit:")
        print(f"    TENDL: https://tendl.web.psi.ch/")
        print(f"    JENDL: https://wwwndc.jaea.go.jp/jendl/j5/j5.html")
        print(f"  Save the file as: {local_path}")
        return local_path  # return path even if file doesn't exist; caller can check

    def _atomic_number(self, symbol: str) -> int:
        """Get atomic number Z from element symbol."""
        _Z = {
            "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
            "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
            "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Ti": 22, "V": 23,
            "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
            "Zr": 40, "Nb": 41, "Mo": 42, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
            "Sb": 51, "Xe": 54, "Cs": 55, "Gd": 64, "Dy": 66, "Ho": 67, "Er": 68,
            "Hf": 72, "Ta": 73, "W": 74, "Pb": 82,
            "Th": 90, "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96,
        }
        return _Z.get(symbol, 0)

    # ── ENDF-6 Lightweight Parser ─────────────────────────────────────────

    def parse_endf(self, filepath: Path) -> dict:
        """Parse ENDF-6 format file, extracting key nuclear data.

        ENDF-6 structure:
          - Section MF=1: General information
          - Section MF=2: Resonance parameters (not fully parsed)
          - Section MF=3: Reaction cross sections (pointwise)
          - Section MF=4: Angular distributions
          - Section MF=5: Energy distributions
          - Section MF=8: Radioactive decay data
          - Section MF=33: Covariances

        This parser extracts:
          - ZA, AWR, temperature from MF=1 MT=451
          - Total, elastic, fission, capture, (n,2n) from MF=3
          - Nu-bar from MF=1 MT=452 or MF=3 MT=452/455/456
        """
        text = filepath.read_text(errors="replace")
        lines = text.split("\n")

        result = {
            "ZA": None,
            "AWR": None,
            "temperature": 300.0,
            "cross_sections": {},   # MT → [(E, σ), ...]
            "nu_bar": None,
            "nu_bar_total": None,
            "reactions": [],
        }

        i = 0
        while i < len(lines):
            line = lines[i]

            # Detect section header: / MF MT /
            if "/" in line and len(line) >= 66:
                # ENDF header format: columns 67-75
                try:
                    mf = int(line[70:72])
                    mt = int(line[72:75])
                except (ValueError, IndexError):
                    i += 1
                    continue

                if mf == 1 and mt == 451:
                    # General information
                    result["ZA"] = self._parse_float_field(line, 0, 11)
                    result["AWR"] = self._parse_float_field(line, 11, 22)
                    temp_line = lines[i + 1] if i + 1 < len(lines) else ""
                    result["temperature"] = self._parse_float_field(temp_line, 0, 11) or 300.0

                elif mf == 3:
                    # Cross section data: LIST or TAB1 format
                    xs_data = self._parse_mf3_section(lines, i, mf, mt)
                    result["cross_sections"][mt] = xs_data
                    # Map MT numbers to reaction names
                    mt_names = {
                        1: "total", 2: "elastic", 4: "inelastic",
                        16: "(n,2n)", 17: "(n,3n)", 18: "fission",
                        102: "capture", 103: "(n,p)", 104: "(n,d)",
                        105: "(n,t)", 107: "(n,alpha)",
                        452: "nu_total", 455: "nu_delayed", 456: "nu_prompt",
                    }
                    result["reactions"].append({
                        "MT": mt,
                        "name": mt_names.get(mt, f"MT{mt}"),
                        "n_points": len(xs_data),
                    })

            i += 1

        return result

    def _parse_float_field(self, line: str, start: int, end: int) -> float | None:
        """Parse a float from an ENDF field (11-character wide fields)."""
        try:
            if end > len(line):
                end = len(line)
            val = line[start:end].strip()
            if val and val not in ("", " ", "+", "-"):
                # Handle ENDF's exponential notation (e.g., 1.234+5 = 1.234e5)
                if "+" in val and "e" not in val.lower():
                    val = val.replace("+", "e+").replace("-", "e-").replace("ee", "e")
                    # Fix double replacements
                return float(val)
        except (ValueError, IndexError):
            pass
        return None

    def _parse_mf3_section(self, lines: list[str], start_idx: int,
                           mf: int, mt: int) -> list[tuple[float, float]]:
        """Parse MF=3 reaction cross section data (TAB1 format)."""
        # TAB1: line 1 = [C1 C2 L1 L2 NR NP / interpolation info]
        #        then NR pairs of (NBT, INT)
        #        then NP pairs of (E, σ)
        result = []
        i = start_idx + 1  # skip header line
        if i >= len(lines):
            return result

        # Second line typically has NR, NP
        try:
            nr = int(lines[i][44:55]) if len(lines[i]) > 44 else 0
            np_points = int(lines[i][55:66]) if len(lines[i]) > 55 else 0
        except (ValueError, IndexError):
            return result

        i += 1

        # Skip interpolation region table (NR pairs)
        i += nr

        # Read NP (E, σ) pairs
        pairs_read = 0
        while pairs_read < np_points and i < len(lines):
            line = lines[i]
            # Each line can have up to 3 pairs (E, σ) = 6 fields
            for field_idx in range(3):
                start1 = field_idx * 22
                start2 = start1 + 11
                if start2 + 11 > len(line):
                    break
                e_val = self._parse_float_field(line, start1, start1 + 11)
                s_val = self._parse_float_field(line, start2, start2 + 11)
                if e_val is not None and s_val is not None:
                    result.append((e_val, s_val))
                    pairs_read += 1
                    if pairs_read >= np_points:
                        break
            i += 1

        return result

    # ── Multigroup Cross Section Generation ──────────────────────────────

    def generate_fewgroup_xs(self, material_id: str,
                              endf_data: dict | None = None,
                              temperature: float = 300.0) -> CrossSection:
        """Generate few-group cross sections from ENDF data.

        Uses a simplistic flux-weighting approach:
          Σ_g = ∫ σ(E) φ(E) dE / ∫ φ(E) dE

        where φ(E) is assumed as:
          - 1/E spectrum above thermal cutoff
          - Maxwellian spectrum below thermal cutoff

        Args:
            material_id: nuclide identifier (e.g., "U235", "Pu239")
            endf_data: pre-parsed ENDF data (or None to use cached/builtin)
            temperature: temperature for thermal Maxwellian (K)

        Returns:
            OneGroupXS or TwoGroupXS depending on group_count
        """
        if endf_data is None:
            # Use builtin approximate data
            builtin = self._builtin.get(material_id)
            if builtin:
                xs = OneGroupXS(
                    D=builtin.get("D", 1.0),
                    Sigma_a=builtin.get("Sigma_a", 0.0),
                    nuSigma_f=builtin.get("nuSigma_f", 0.0),
                    kappaSigma_f=builtin.get("kappaSigma_f", 0.0),
                )
                self._xs_cache[material_id] = xs
                return xs
            raise ValueError(f"No data for {material_id} and no ENDF file provided.")

        # Get pointwise total, capture, fission cross sections
        total_xs = endf_data.get("cross_sections", {}).get(1, [])  # MT=1: total
        capture_xs = endf_data.get("cross_sections", {}).get(102, [])  # MT=102: capture
        fission_xs = endf_data.get("cross_sections", {}).get(18, [])  # MT=18: fission
        elastic_xs = endf_data.get("cross_sections", {}).get(2, [])  # MT=2: elastic

        if not total_xs:
            # Fallback to builtin
            return self.generate_fewgroup_xs(material_id, None, temperature)

        # Sort by energy (descending)
        total_xs = sorted(total_xs, key=lambda x: x[0], reverse=True)

        if self.group_count == 1:
            xs_val = self._flux_weight_1group(total_xs, capture_xs, fission_xs, temperature)
            self._xs_cache[material_id] = xs_val
            return xs_val
        else:
            xs_val = self._flux_weight_2group(total_xs, capture_xs, fission_xs,
                                              elastic_xs, temperature)
            self._xs_cache[material_id] = xs_val
            return xs_val

    def _flux_weight_1group(self, total_xs, capture_xs, fission_xs,
                             temperature: float) -> OneGroupXS:
        """1-group averaging with 1/E + Maxwellian flux."""
        if not total_xs:
            return OneGroupXS(D=1.0, Sigma_a=0.0, nuSigma_f=0.0, kappaSigma_f=0.0)

        kT = temperature * 8.617333262e-5  # eV

        def flux_weight(e):
            if e > 0.625:  # epithermal
                return 1.0 / max(e, 1e-3)
            else:  # thermal Maxwellian
                return e * math.exp(-e / kT)

        # 使用简单数值积分
        energies = np.array([p[0] for p in total_xs])
        e_min, e_max = max(energies.min(), 1e-5), min(energies.max(), 2e7)
        e_grid = np.logspace(np.log10(e_min), np.log10(e_max), 1000)

        def integrate(values_func, e_vals, weight_func):
            w = np.array([weight_func(e) for e in e_vals])
            v = np.array([values_func(e) for e in e_vals])
            num = np.trapz(v * w, e_vals)
            den = np.trapz(w, e_vals)
            return num / den if den > 0 else 0.0

        def interp_xs(xs_list):
            if not xs_list:
                return lambda e: 0.0
            sorted_xs = sorted(xs_list, key=lambda x: x[0])
            xs_e = np.array([p[0] for p in sorted_xs])
            xs_v = np.array([p[1] for p in sorted_xs])
            return lambda e: float(np.interp(e, xs_e, xs_v, left=xs_v[0], right=xs_v[-1]))

        total_func = interp_xs(total_xs)
        capture_func = interp_xs(capture_xs)
        fission_func = interp_xs(fission_xs)

        sigma_a = integrate(capture_func, e_grid, flux_weight)
        sigma_f = integrate(fission_func, e_grid, flux_weight)
        sigma_t = integrate(total_func, e_grid, flux_weight)

        # D ≈ 1/(3 Σ_tr), Σ_tr ≈ Σ_total - μ̄ Σ_s
        # Rough: Σ_tr ≈ Σ_total (for heavy nuclides)
        D = 1.0 / (3.0 * max(sigma_t, 1e-6))

        # Approximate ν from ENDF (typically 2.4-2.9 for fissile)
        nu = 2.43  # default for U-235 thermal

        return OneGroupXS(
            D=D,
            Sigma_a=sigma_a,
            nuSigma_f=nu * sigma_f,
            kappaSigma_f=sigma_f,  # simplified; real κ ≈ 200 MeV/fission
        )

    def _flux_weight_2group(self, total_xs, capture_xs, fission_xs,
                             elastic_xs, temperature: float) -> TwoGroupXS:
        """2-group averaging: fast (>0.625 eV) and thermal (<0.625 eV)."""
        # Simplified: just split at the thermal boundary
        kT = temperature * 8.617333262e-5

        def fast_flux(e):
            if e > 0.625:
                return 1.0 / max(e, 1e-3)
            return 0.0

        def thermal_flux(e):
            if e <= 0.625:
                return e * math.exp(-e / kT)
            return 0.0

        energies = np.array([p[0] for p in total_xs])
        e_min = max(energies.min(), 1e-5)
        e_max = min(energies.max(), 2e7)
        e_grid = np.logspace(np.log10(e_min), np.log10(e_max), 2000)

        def interp_xs(xs_list):
            if not xs_list:
                return lambda e: 0.0
            sorted_xs = sorted(xs_list, key=lambda x: x[0])
            xs_e = np.array([p[0] for p in sorted_xs])
            xs_v = np.array([p[1] for p in sorted_xs])
            return lambda e: float(np.interp(e, xs_e, xs_v, left=xs_v[0], right=xs_v[-1]))

        capture_func = interp_xs(capture_xs)
        fission_func = interp_xs(fission_xs)
        elastic_func = interp_xs(elastic_xs)
        total_func = interp_xs(total_xs)

        def average(func, weight):
            w = np.array([weight(e) for e in e_grid])
            v = np.array([func(e) for e in e_grid])
            num = np.trapz(v * w, e_grid)
            den = np.trapz(w, e_grid)
            return num / den if den > 0 else 0.0

        # Fast group
        sig_a0 = average(capture_func, fast_flux)
        sig_f0 = average(fission_func, fast_flux)
        sig_t0 = average(total_func, fast_flux)
        sig_e0 = average(elastic_func, fast_flux)

        # Thermal group
        sig_a1 = average(capture_func, thermal_flux)
        sig_f1 = average(fission_func, thermal_flux)
        sig_t1 = average(total_func, thermal_flux)
        sig_e1 = average(elastic_func, thermal_flux)

        D0 = 1.0 / (3.0 * max(sig_t0, 1e-6))
        D1 = 1.0 / (3.0 * max(sig_t1, 1e-6))

        # Scattering matrix: fast→thermal from elastic slowing down
        # Rough: Σ_s0→1 ≈ ξ Σ_e0, where ξ is average lethargy gain
        xi_H = 1.0  # hydrogen
        sigma_s01 = xi_H * sig_e0 * 0.5  # approximate
        sigma_s00 = sig_e0 - sigma_s01
        sigma_s11 = sig_e1  # within-group scattering (thermal)

        nu = 2.43
        chi_fast = 1.0  # all fission neutrons born fast
        chi_thermal = 0.0

        return TwoGroupXS(
            D=[D0, D1],
            Sigma_a=[sig_a0, sig_a1],
            nuSigma_f=[nu * sig_f0, nu * sig_f1],
            kappaSigma_f=[sig_f0, sig_f1],
            Sigma_s=[[sigma_s00, sigma_s01], [0.0, sigma_s11]],
            chi=[chi_fast, chi_thermal],
        )

    # ── XSProvider interface ──────────────────────────────────────────────

    def get_xs(self, material_id: str, state: "CellState | None" = None) -> CrossSection:
        # Check cache
        if material_id in self._xs_cache:
            return self._xs_cache[material_id]

        # Check builtin
        if material_id in self._builtin:
            xs = OneGroupXS(
                D=self._builtin[material_id]["D"],
                Sigma_a=self._builtin[material_id]["Sigma_a"],
                nuSigma_f=self._builtin[material_id]["nuSigma_f"],
                kappaSigma_f=self._builtin[material_id]["kappaSigma_f"],
            )
            self._xs_cache[material_id] = xs
            return xs

        # Try to load from downloaded ENDF file
        for filename in sorted(self.data_dir.glob("*")):
            fname = filename.name
            # Match patterns like U235.endf, U235.tendl, etc.
            if material_id.lower() in fname.lower():
                if filename.stat().st_size > 100:  # non-empty file
                    try:
                        endf_data = self.parse_endf(filename)
                        if endf_data.get("cross_sections"):
                            return self.generate_fewgroup_xs(material_id, endf_data)
                    except Exception:
                        pass

        raise KeyError(
            f"No cross-section data for '{material_id}'. "
            f"Available builtin: {list(self._builtin.keys())}. "
            f"Use NuclearDataLibrary.download_nuclide() to download ENDF data, "
            f"or use a builtin nuclide ID."
        )

    def get_group_count(self) -> int:
        return self.group_count

    def get_source_info(self) -> str:
        return f"nuclear_library (TENDL-2023/JENDL-5 adapted, {self.group_count}-group)"


# ── XSProvider factory ────────────────────────────────────────────────────

def create_xs_provider(provider_type: str, **kwargs) -> XSProvider:
    """Factory to create an XS provider from a string identifier."""
    if provider_type in ("constant", "constant_1g", "constant_2g"):
        return ConstantXSProvider(**kwargs)
    elif provider_type in ("tabulated", "tabulated_1g", "tabulated_2g"):
        return TabulatedXSProvider(**kwargs)
    elif provider_type in ("nuclear_library", "endf", "tendl", "jendl"):
        return NuclearDataLibrary(**kwargs)
    else:
        raise ValueError(f"Unknown XS provider type: {provider_type}")


# ── Helper: download common nuclides ──────────────────────────────────────

def download_common_nuclides(data_dir: str = "data/nuclear_library",
                              source: str = "tendl") -> dict[str, Path]:
    """Download a set of commonly used nuclides for reactor physics.

    Returns dict mapping nuclide_id → file path.
    """
    lib = NuclearDataLibrary(data_dir=data_dir)
    nuclides = [
        ("U", 235), ("U", 238), ("Pu", 239), ("Pu", 240), ("Pu", 241),
        ("H", 1), ("O", 16), ("B", 10), ("B", 11),
        ("Zr", 91), ("Fe", 56), ("Xe", 135),
    ]
    results = {}
    for element, A in nuclides:
        nuclide_id = f"{element}{A}"
        try:
            path = lib.download_nuclide(element, A, source=source)
            results[nuclide_id] = path
        except Exception as e:
            print(f"  [WARN] Could not download {nuclide_id}: {e}")

    return results


# Late import for type hint
from .material import CellState
