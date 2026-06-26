# Skill: renderer

## Purpose

Render solver results as publication-quality visualizations.

This skill generates plots (heatmaps, surface plots, line charts, bar charts,
dashboards) from solver output data. It supports both static PNG images
(matplotlib) and interactive HTML plots (plotly).

The LLM should use this skill when the user asks to:
- "show me the power distribution"
- "plot the flux"
- "visualize the convergence"
- "make a 3D plot of power"
- "show the material layout"
- "create a dashboard"
- "plot keff vs burnup"
- "show nuclide composition"

## Supported Plot Types

| plot_type | description | when to use |
|---|---|---|
| heatmap_2d | 2D colored map | spatial distributions (flux, power, XS) |
| surface_3d | 3D interactive surface | spatial topography |
| line | 1D line plot | convergence, profiles, keff vs burnup |
| bar | bar chart | nuclide composition, reaction rates |
| material_map | color-coded layout | geometry visualization |
| dashboard | 4-panel overview | summary of all results |

## Inputs

| name | type | required | description |
|---|---|---|---|
| plot_type | string | yes | one of the supported plot types |
| data | array | yes* | primary data array |
| title | string | no | plot title |
| colormap | string | no | matplotlib colormap name (viridis, inferno, hot, plasma, coolwarm) |
| xlabel | string | no | x-axis label |
| ylabel | string | no | y-axis label |
| zlabel | string | no | colorbar/z-axis label |
| mesh_shape | [ny, nx] | no | reshape 1D data to 2D |
| output_path | string | no | output file path (default: outputs/plot.png) |
| interactive | bool | no | use plotly for interactive HTML (default: false) |
| residual_history | array | no | for convergence plots |
| material_map | array | no | for material layout plots |
| burnup | array | no | x-axis for keff vs burnup |
| keff | array | no | y-axis for keff vs burnup |
| nuclides | dict | no | {name: density} for bar charts |
| width | int | no | plot width in pixels (default: 800) |
| height | int | no | plot height in pixels (default: 600) |

## Outputs

| name | type | description |
|---|---|---|
| result | RenderResult | contains output_paths, plot_type, success, metadata |
| output_path | string | path to the generated image/HTML |

## Preconditions

- Data arrays must be non-empty numpy arrays
- matplotlib must be installed for static plots
- plotly must be installed for interactive plots
- Output directory must be writable

## Postconditions

- Output file exists at specified path
- Image/HTML is ready for display

## Failure Modes

- matplotlib or plotly not installed
- empty data array
- invalid plot_type
- shape mismatch between data and mesh_shape

## Common Recipes (LLM can generate these)

### Recipe 1: Power Heatmap

```python
from src.visualization.plots import power_heatmap
spec = power_heatmap(power_array, mesh_shape=(ny, nx),
                     output="power_heatmap.png")
renderer.render(spec)
```

### Recipe 2: 3D Interactive Power

```python
from src.visualization.plots import power_3d_surface
spec = power_3d_surface(power_array, mesh_shape=(ny, nx),
                        output="power_3d.html")
renderer.render(spec)
```

### Recipe 3: Convergence Plot

```python
from src.visualization.plots import convergence_plot
spec = convergence_plot(residual_history, output="convergence.png")
renderer.render(spec)
```

### Recipe 4: Material Map

```python
from src.visualization.plots import material_layout
spec = material_layout(material_array, output="material_map.png")
renderer.render(spec)
```

### Recipe 5: Full Dashboard

```python
from src.visualization.plots import result_dashboard
spec = result_dashboard(flux, power, residual,
                        material_map=mat_map,
                        mesh_shape=(ny, nx),
                        output="dashboard.png")
renderer.render(spec)
```

### Recipe 6: k_eff vs Burnup

```python
from src.visualization.plots import keff_vs_burnup
spec = keff_vs_burnup(burnup_list, keff_list, output="keff_vs_bu.png")
renderer.render(spec)
```

## LLM Decision Rules

When the user says something like:

| User Request | Plot Type | Colormap | Notes |
|---|---|---|---|
| "show power" | heatmap_2d | hot | reshape 1D→2D if needed |
| "3D power map" | surface_3d | hot | use plotly backend |
| "flux distribution" | heatmap_2d | inferno | |
| "convergence history" | line | — | semilogy scale |
| "material layout" | material_map | — | tab20 colormap |
| "dashboard / overview" | dashboard | — | 4-panel |
| "keff vs burnup" | line | — | x=burnup, y=keff |
| "nuclide composition" | bar | — | |
| "absorption cross section" | heatmap_2d | plasma | |
| "compare flux and power" | dashboard | — | side-by-side |

## Backend Selection

- Use `matplotlib` backend for static PNG images (default)
- Use `plotly` backend for interactive 3D HTML plots
- Specify `interactive: true` in the spec for plotly backend

## Example Workflow Step

```yaml
- id: render_power
  skill: renderer
  input:
    plot_type: heatmap_2d
    data: "{{ solve.power }}"
    mesh_shape: [10, 10]
    title: "Power Distribution"
    colormap: hot
    output_path: power_heatmap.png
```
