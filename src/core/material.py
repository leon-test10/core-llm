"""
Material and state definitions.

Materials are identified by string IDs; cross sections come from providers.
This decouples material identity from its physical properties.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CellState:
    """Physical state of a single computational cell.

    All fields are optional — a solver queries what it needs.
    """
    temperature: float = 900.0       # K
    density: float = 1.0             # g/cm³ (moderator density)
    burnup: float = 0.0              # GWd/tU
    boron_ppm: float = 0.0           # ppm
    rod_fraction: float = 0.0        # 0.0 = fully withdrawn, 1.0 = fully inserted
    nuclide_density: dict[str, float] | None = None  # nuclide → atoms/barn·cm

    def copy(self) -> "CellState":
        return CellState(
            temperature=self.temperature,
            density=self.density,
            burnup=self.burnup,
            boron_ppm=self.boron_ppm,
            rod_fraction=self.rod_fraction,
            nuclide_density=dict(self.nuclide_density) if self.nuclide_density else None,
        )

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "density": self.density,
            "burnup": self.burnup,
            "boron_ppm": self.boron_ppm,
            "rod_fraction": self.rod_fraction,
        }


@dataclass
class Material:
    """A material identified by a unique ID.

    The material itself does NOT store cross sections — they come from
    an XSProvider, which may be constant, tabulated, or computed.
    """
    material_id: str
    name: str
    xs_provider_id: str = "constant_1g"
    metadata: dict[str, Any] = field(default_factory=dict)


class MaterialRegistry:
    """Container for all materials in a problem.

    Usage:
        registry = MaterialRegistry()
        registry.register(Material("fuel", "UO2 Fuel", xs_provider_id="tabulated_1g"))
        mat = registry.get("fuel")
    """

    def __init__(self):
        self._materials: dict[str, Material] = {}

    def register(self, material: Material):
        self._materials[material.material_id] = material

    def get(self, material_id: str) -> Material:
        if material_id not in self._materials:
            raise KeyError(f"Material '{material_id}' not registered. "
                           f"Available: {list(self._materials.keys())}")
        return self._materials[material_id]

    def list_ids(self) -> list[str]:
        return list(self._materials.keys())

    def to_dict(self) -> dict:
        return {mid: {"name": m.name, "xs_provider": m.xs_provider_id,
                       "metadata": m.metadata}
                for mid, m in self._materials.items()}
