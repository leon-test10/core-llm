"""
Pre-built visualization recipes for common analysis tasks.

Each function takes standard solver outputs and returns a PlotSpec.
LLM can call these directly or use them as templates for custom plots.

Recipe categories:
  - flux_heatmap: 2D color map of neutron flux
  - power_heatmap: 2D color map of power distribution
  - material_map: colored layout of materials
  - convergence_plot: residual vs iteration
  - axial_profile: line plot along an axis
  - power_peaking: bar chart of pin peaking factors
  - dashboard: 4-panel overview
  - keff_vs_burnup: line plot (for multi-step results)
  - nuclide_composition: bar chart of nuclide densities
  - cross_section_map: heatmap of a specific cross section field
"""

import numpy as np
from .renderer import PlotSpec, PlotBuilder


# ── 2D Spatial Plots ──────────────────────────────────────────────────

def flux_heatmap(flux: np.ndarray, mesh_shape: tuple | None = None,
                  title: str = "Neutron Flux Distribution",
                  output: str = "flux_heatmap.png") -> PlotSpec:
    """2D heatmap of neutron flux.

    Args:
        flux: 1D or 2D flux array
        mesh_shape: (ny, nx) — required if flux is 1D
        title: plot title
        output: output filename
    """
    if flux.ndim == 1 and mesh_shape is not None:
        flux = flux.reshape(mesh_shape)

    return (PlotBuilder("heatmap_2d")
            .with_title(title)
            .with_data(flux)
            .with_colormap("inferno")
            .with_labels("X", "Y", "Relative Flux")
            .save_to(output)
            .build())


def power_heatmap(power: np.ndarray, mesh_shape: tuple | None = None,
                   title: str = "Power Distribution",
                   output: str = "power_heatmap.png") -> PlotSpec:
    """2D heatmap of power distribution.

    Args:
        power: 1D or 2D power array
        mesh_shape: (ny, nx) — required if power is 1D
        title: plot title
        output: output filename
    """
    if power.ndim == 1 and mesh_shape is not None:
        power = power.reshape(mesh_shape)

    return (PlotBuilder("heatmap_2d")
            .with_title(title)
            .with_data(power)
            .with_colormap("hot")
            .with_labels("X", "Y", "Normalized Power")
            .save_to(output)
            .build())


def material_layout(material_map: np.ndarray,
                    title: str = "Material Layout",
                    output: str = "material_map.png") -> PlotSpec:
    """Color-coded material layout map.

    Args:
        material_map: 2D list or array of material ID strings
        title: plot title
        output: output filename
    """
    if isinstance(material_map, list):
        material_map = np.array(material_map, dtype=str)
    return (PlotBuilder("material_map")
            .with_title(title)
            .with_data(material_map)
            .save_to(output)
            .build())


def power_3d_surface(power: np.ndarray, mesh_shape: tuple | None = None,
                      title: str = "3D Power Surface",
                      output: str = "power_3d.html") -> PlotSpec:
    """Interactive 3D surface plot of power distribution.

    Args:
        power: 1D or 2D power array
        mesh_shape: (ny, nx) — required if power is 1D
        title: plot title
        output: output filename (.html for plotly)
    """
    if power.ndim == 1 and mesh_shape is not None:
        power = power.reshape(mesh_shape)

    return (PlotBuilder("surface_3d")
            .with_title(title)
            .with_data(power)
            .with_colormap("hot")
            .with_labels("X", "Y", "Power")
            .interactive(True)
            .save_to(output)
            .build())


# ── 1D Plots ──────────────────────────────────────────────────────────

def convergence_plot(residual_history: np.ndarray | list,
                      title: str = "Convergence History",
                      output: str = "convergence.png") -> PlotSpec:
    """Semilog plot of residual vs iteration.

    Args:
        residual_history: list/array of residuals per iteration
        title: plot title
        output: output filename
    """
    res = np.asarray(residual_history)
    return (PlotBuilder("line")
            .with_title(title)
            .with_data(res)
            .with_labels("Iteration", "Residual (L2 norm)")
            .with_size(800, 400)
            .save_to(output)
            .build())


def axial_profile(flux_or_power: np.ndarray, axis: str = "y",
                   mesh_shape: tuple | None = None,
                   title: str = "Axial Profile",
                   output: str = "axial_profile.png") -> PlotSpec:
    """Line plot of flux/power averaged along one axis.

    Args:
        flux_or_power: 1D or 2D array
        axis: "x" or "y" — axis to average along
        mesh_shape: (ny, nx) — required if 1D
        title: plot title
        output: output filename
    """
    if flux_or_power.ndim == 1 and mesh_shape is not None:
        flux_or_power = flux_or_power.reshape(mesh_shape)

    if axis == "y":
        profile = flux_or_power.mean(axis=1)
        xlabel = "Y position"
    else:
        profile = flux_or_power.mean(axis=0)
        xlabel = "X position"

    return (PlotBuilder("line")
            .with_title(title)
            .with_data(profile)
            .with_labels(xlabel, "Average Value")
            .with_size(800, 350)
            .save_to(output)
            .build())


def keff_vs_burnup(burnup: list[float], keff: list[float],
                    title: str = "k_eff vs Burnup",
                    output: str = "keff_vs_burnup.png") -> PlotSpec:
    """k_eff as a function of burnup.

    Args:
        burnup: list of burnup values [GWd/tU]
        keff: list of corresponding k_eff values
        title: plot title
        output: output filename
    """
    return (PlotBuilder("line")
            .with_title(title)
            .with_data(np.array(keff))
            .with_x(np.array(burnup))
            .with_labels("Burnup [GWd/tU]", "k_eff")
            .with_size(800, 400)
            .save_to(output)
            .build())


def nuclide_bar(nuclides: dict[str, float],
                 title: str = "Nuclide Composition",
                 output: str = "nuclide_composition.png") -> PlotSpec:
    """Bar chart of nuclide densities.

    Args:
        nuclides: {nuclide_name: density} dict
        title: plot title
        output: output filename
    """
    labels = list(nuclides.keys())
    values = np.array(list(nuclides.values()))
    return (PlotBuilder("bar")
            .with_title(title)
            .with_data(values)
            .with_x(np.array(labels))
            .with_labels("Nuclide", "Density [atoms/barn·cm]")
            .with_size(900, 450)
            .save_to(output)
            .build())


def cross_section_map(xs_field: np.ndarray, field_name: str = "Sigma_a",
                       mesh_shape: tuple | None = None,
                       output: str = "xs_map.png") -> PlotSpec:
    """Heatmap of a cross-section field (e.g., absorption, fission).

    Args:
        xs_field: 1D or 2D array of cross-section values
        field_name: name of the XS field (for title/label)
        mesh_shape: (ny, nx)
        output: output filename
    """
    if xs_field.ndim == 1 and mesh_shape is not None:
        xs_field = xs_field.reshape(mesh_shape)

    return (PlotBuilder("heatmap_2d")
            .with_title(f"{field_name} Distribution")
            .with_data(xs_field)
            .with_colormap("plasma")
            .with_labels("X", "Y", f"{field_name} [cm⁻¹]")
            .save_to(output)
            .build())


# ── Dashboard ──────────────────────────────────────────────────────────

def result_dashboard(flux: np.ndarray, power: np.ndarray,
                      residual: np.ndarray | list,
                      material_map: np.ndarray | None = None,
                      mesh_shape: tuple | None = None,
                      title: str = "Core Solver Results",
                      output: str = "dashboard.png") -> PlotSpec:
    """4-panel dashboard: flux, power, convergence, materials.

    Args:
        flux: flux array
        power: power array
        residual: residual history
        material_map: optional material layout
        mesh_shape: (ny, nx) for 1D→2D reshaping
        title: overall title
        output: output filename
    """
    if flux.ndim == 1 and mesh_shape is not None:
        flux = flux.reshape(mesh_shape)
    if power.ndim == 1 and mesh_shape is not None:
        power = power.reshape(mesh_shape)

    spec = PlotSpec(plot_type="dashboard", title=title, output_path=output)
    spec.data_labels["flux"] = flux
    spec.data_labels["power"] = power
    spec.data_labels["residual"] = np.asarray(residual)
    if material_map is not None:
        if isinstance(material_map, list):
            material_map = np.array(material_map, dtype=str)
        spec.data_labels["material"] = material_map

    return spec
