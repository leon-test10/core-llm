"""
Visualization renderer for core solver results.

Design principles:
  1. Renderers consume standard data containers (ndarray + metadata dict)
  2. Each plot type is a standalone function — composable by LLM
  3. Output formats: PNG (static), HTML (interactive via plotly)
  4. LLM selects plot + parameters based on natural language intent

Supported visualizations:
  - 2D heatmap: flux, power, material map, cross sections
  - 3D surface: power/flux topography
  - 1D line: convergence history, axial/radial profiles, spectra
  - Bar: nuclide composition, reaction rates
  - Dashboard: multi-panel overview
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── Optional imports with graceful degradation ─────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.ticker import FuncFormatter
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ── Data containers ────────────────────────────────────────────────────

@dataclass
class PlotSpec:
    """Specification for a single plot, LLM-friendly."""
    plot_type: str                     # "heatmap_2d", "surface_3d", "line", "bar", "dashboard"
    title: str = ""
    data: np.ndarray | None = None     # primary data
    data_labels: dict[str, np.ndarray] = field(default_factory=dict)  # auxiliary data
    xlabel: str = ""
    ylabel: str = ""
    zlabel: str = ""
    colormap: str = "viridis"
    x: np.ndarray | None = None        # x-axis values
    y: np.ndarray | None = None        # y-axis values
    annotations: list[dict] = field(default_factory=list)
    output_path: str = "outputs/plot.png"
    width: int = 800
    height: int = 600
    interactive: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class RenderResult:
    """Result of a rendering operation."""
    output_paths: list[str] = field(default_factory=list)
    plot_type: str = ""
    success: bool = True
    error: str = ""
    metadata: dict = field(default_factory=dict)


# ── Renderer engine ────────────────────────────────────────────────────

class Renderer:
    """Unified rendering engine. Dispatches to matplotlib or plotly.

    Usage:
        renderer = Renderer(backend="matplotlib")
        spec = PlotSpec(plot_type="heatmap_2d", title="Power", data=power_2d)
        result = renderer.render(spec)
    """

    def __init__(self, backend: str = "matplotlib", output_dir: str = "outputs"):
        self.backend = backend
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if backend == "plotly" and not HAS_PLOTLY:
            raise ImportError("plotly not installed. pip install plotly")
        if backend == "matplotlib" and not HAS_MPL:
            raise ImportError("matplotlib not installed. pip install matplotlib")

    def render(self, spec: PlotSpec) -> RenderResult:
        """Render a single plot specification."""
        try:
            if spec.plot_type == "heatmap_2d":
                return self._render_heatmap_2d(spec)
            elif spec.plot_type == "surface_3d":
                return self._render_surface_3d(spec)
            elif spec.plot_type == "line":
                return self._render_line(spec)
            elif spec.plot_type == "bar":
                return self._render_bar(spec)
            elif spec.plot_type == "dashboard":
                return self._render_dashboard(spec)
            elif spec.plot_type == "material_map":
                return self._render_material_map(spec)
            else:
                return RenderResult(success=False, error=f"Unknown plot_type: {spec.plot_type}")
        except Exception as e:
            import traceback
            return RenderResult(success=False, error=f"{e}\n{traceback.format_exc()}")

    # ── 2D Heatmap ────────────────────────────────────────────────────

    def _render_heatmap_2d(self, spec: PlotSpec) -> RenderResult:
        data = spec.data
        if data is None:
            return RenderResult(success=False, error="No data provided")

        if self.backend == "matplotlib":
            return self._heatmap_2d_mpl(spec)
        else:
            return self._heatmap_2d_plotly(spec)

    def _heatmap_2d_mpl(self, spec: PlotSpec) -> RenderResult:
        fig, ax = plt.subplots(figsize=(spec.width / 100, spec.height / 100))
        data = spec.data

        extent = None
        if spec.x is not None and spec.y is not None:
            extent = [spec.x[0], spec.x[-1], spec.y[0], spec.y[-1]]

        im = ax.imshow(data, cmap=spec.colormap, origin="lower",
                       aspect="auto", extent=extent, interpolation="bilinear")
        cbar = plt.colorbar(im, ax=ax, shrink=0.92)
        cbar.set_label(spec.zlabel or "Value")

        ax.set_title(spec.title, fontsize=13, fontweight="bold")
        ax.set_xlabel(spec.xlabel or "X")
        ax.set_ylabel(spec.ylabel or "Y")

        # Annotations
        for ann in spec.annotations:
            ax.annotate(ann.get("text", ""),
                        xy=(ann.get("x", 0), ann.get("y", 0)),
                        fontsize=ann.get("fontsize", 8),
                        color=ann.get("color", "white"),
                        ha="center", va="center")

        output_path = str(self.output_dir / spec.output_path)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return RenderResult(output_paths=[output_path], plot_type="heatmap_2d",
                            metadata={"colormap": spec.colormap,
                                       "data_range": [float(data.min()), float(data.max())]})

    def _heatmap_2d_plotly(self, spec: PlotSpec) -> RenderResult:
        if not HAS_PLOTLY:
            return self._heatmap_2d_mpl(spec)

        data = spec.data
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=spec.x,
            y=spec.y,
            colorscale=spec.colormap,
            colorbar=dict(title=spec.zlabel or "Value"),
        ))
        fig.update_layout(
            title=spec.title,
            xaxis_title=spec.xlabel or "X",
            yaxis_title=spec.ylabel or "Y",
            width=spec.width,
            height=spec.height,
        )

        output_path = str(self.output_dir / spec.output_path).replace(".png", ".html")
        fig.write_html(output_path)
        return RenderResult(output_paths=[output_path], plot_type="heatmap_2d",
                            metadata={"interactive": True})

    # ── 3D Surface ────────────────────────────────────────────────────

    def _render_surface_3d(self, spec: PlotSpec) -> RenderResult:
        if spec.data is None:
            return RenderResult(success=False, error="No data provided")

        if self.backend == "plotly" and HAS_PLOTLY:
            return self._surface_3d_plotly(spec)
        else:
            return self._surface_3d_mpl(spec)

    def _surface_3d_mpl(self, spec: PlotSpec) -> RenderResult:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(spec.width / 100, spec.height / 100))
        ax = fig.add_subplot(111, projection="3d")

        data = spec.data
        ny, nx = data.shape if data.ndim == 2 else (1, len(data))

        if spec.x is None:
            spec.x = np.arange(nx)
        if spec.y is None:
            spec.y = np.arange(ny)

        X, Y = np.meshgrid(spec.x, spec.y)
        surf = ax.plot_surface(X, Y, data, cmap=spec.colormap,
                               linewidth=0, antialiased=True, alpha=0.85)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10,
                     label=spec.zlabel or "Value")
        ax.set_title(spec.title, fontsize=13)
        ax.set_xlabel(spec.xlabel or "X")
        ax.set_ylabel(spec.ylabel or "Y")
        ax.set_zlabel(spec.zlabel or "Z")

        output_path = str(self.output_dir / spec.output_path)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return RenderResult(output_paths=[output_path], plot_type="surface_3d")

    def _surface_3d_plotly(self, spec: PlotSpec) -> RenderResult:
        data = spec.data
        ny, nx = data.shape if data.ndim == 2 else (1, len(data))
        if spec.x is None:
            spec.x = np.arange(nx)
        if spec.y is None:
            spec.y = np.arange(ny)

        fig = go.Figure(data=[go.Surface(
            z=data, x=spec.x, y=spec.y,
            colorscale=spec.colormap,
        )])
        fig.update_layout(
            title=spec.title,
            scene=dict(
                xaxis_title=spec.xlabel or "X",
                yaxis_title=spec.ylabel or "Y",
                zaxis_title=spec.zlabel or "Z",
            ),
            width=spec.width,
            height=spec.height,
        )

        output_path = str(self.output_dir / spec.output_path).replace(".png", ".html")
        fig.write_html(output_path)
        return RenderResult(output_paths=[output_path], plot_type="surface_3d",
                            metadata={"interactive": True})

    # ── 1D Line Plot ──────────────────────────────────────────────────

    def _render_line(self, spec: PlotSpec) -> RenderResult:
        if self.backend == "plotly" and HAS_PLOTLY:
            return self._line_plotly(spec)
        else:
            return self._line_mpl(spec)

    def _line_mpl(self, spec: PlotSpec) -> RenderResult:
        fig, ax = plt.subplots(figsize=(spec.width / 100, spec.height / 100))

        # Primary data
        if spec.data is not None:
            x = spec.x if spec.x is not None else np.arange(len(spec.data))
            ax.plot(x, spec.data, "b-", linewidth=1.5, label=spec.title or "data")

        # Auxiliary data (multiple lines)
        colors = plt.cm.tab10.colors
        for i, (label, arr) in enumerate(spec.data_labels.items()):
            x = spec.x if spec.x is not None else np.arange(len(arr))
            ax.plot(x, arr, "-", color=colors[(i + 1) % len(colors)],
                    linewidth=1.5, label=label)

        ax.set_title(spec.title, fontsize=13, fontweight="bold")
        ax.set_xlabel(spec.xlabel or "X")
        ax.set_ylabel(spec.ylabel or "Y")
        ax.grid(True, alpha=0.3)
        if spec.data_labels:
            ax.legend(fontsize=9)

        output_path = str(self.output_dir / spec.output_path)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return RenderResult(output_paths=[output_path], plot_type="line")

    def _line_plotly(self, spec: PlotSpec) -> RenderResult:
        fig = go.Figure()
        if spec.data is not None:
            x = spec.x if spec.x is not None else np.arange(len(spec.data))
            fig.add_trace(go.Scatter(x=x, y=spec.data, mode="lines",
                                     name=spec.title or "data"))
        for label, arr in spec.data_labels.items():
            x_arr = spec.x if spec.x is not None else np.arange(len(arr))
            fig.add_trace(go.Scatter(x=x_arr, y=arr, mode="lines", name=label))

        fig.update_layout(
            title=spec.title,
            xaxis_title=spec.xlabel or "X",
            yaxis_title=spec.ylabel or "Y",
            width=spec.width, height=spec.height,
        )
        output_path = str(self.output_dir / spec.output_path).replace(".png", ".html")
        fig.write_html(output_path)
        return RenderResult(output_paths=[output_path], plot_type="line",
                            metadata={"interactive": True})

    # ── Bar Chart ──────────────────────────────────────────────────────

    def _render_bar(self, spec: PlotSpec) -> RenderResult:
        if not HAS_MPL:
            return RenderResult(success=False, error="matplotlib required")

        fig, ax = plt.subplots(figsize=(spec.width / 100, spec.height / 100))

        data = spec.data
        if data is not None:
            labels = spec.x if spec.x is not None else [str(i) for i in range(len(data))]
            bars = ax.bar(labels, data, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(data))))
            # Add value labels on bars
            for bar, val in zip(bars, data):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * data.max(),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(spec.title, fontsize=13, fontweight="bold")
        ax.set_xlabel(spec.xlabel or "")
        ax.set_ylabel(spec.ylabel or "")
        ax.tick_params(axis="x", rotation=45)

        output_path = str(self.output_dir / spec.output_path)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return RenderResult(output_paths=[output_path], plot_type="bar")

    # ── Material Map ──────────────────────────────────────────────────

    def _render_material_map(self, spec: PlotSpec) -> RenderResult:
        """Render a material layout map with color-coded cells and labels."""
        if not HAS_MPL:
            return RenderResult(success=False, error="matplotlib required")

        data = spec.data  # 2D array of material_id strings
        if data is None:
            return RenderResult(success=False, error="No data for material map")
        # Convert list to numpy if needed
        if isinstance(data, list):
            data = np.array(data, dtype=str)
        if data.dtype.kind not in ("U", "S", "O"):
            return RenderResult(success=False,
                                error=f"Material map requires string array, got {data.dtype}")

        # Build color mapping
        unique_materials = sorted(set(data.flat))
        cmap = plt.cm.tab20
        color_map = {mat: cmap(i / max(len(unique_materials), 1))
                     for i, mat in enumerate(unique_materials)}

        ny, nx = data.shape
        fig, ax = plt.subplots(figsize=(spec.width / 100, spec.height / 100))

        # Draw colored rectangles
        for j in range(ny):
            for i in range(nx):
                mat = data[j, i]
                color = color_map.get(mat, (0.5, 0.5, 0.5, 1.0))
                rect = plt.Rectangle((i, j), 1, 1, facecolor=color,
                                     edgecolor="white", linewidth=0.5)
                ax.add_patch(rect)
                # Label
                short_label = mat[:6] if len(mat) > 6 else mat
                ax.text(i + 0.5, j + 0.5, short_label, ha="center", va="center",
                        fontsize=6, color="white" if sum(color[:3]) < 1.5 else "black")

        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_aspect("equal")
        ax.set_title(spec.title or "Material Map", fontsize=13, fontweight="bold")
        ax.set_xlabel(f"X ({nx} cells)")
        ax.set_ylabel(f"Y ({ny} cells)")

        # Legend
        legend_patches = [plt.Rectangle((0, 0), 1, 1, facecolor=color_map[m],
                                         label=m) for m in unique_materials]
        ax.legend(handles=legend_patches, loc="upper right",
                  fontsize=7, ncol=max(1, len(unique_materials) // 8))

        output_path = str(self.output_dir / spec.output_path)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return RenderResult(output_paths=[output_path], plot_type="material_map")

    # ── Dashboard ──────────────────────────────────────────────────────

    def _render_dashboard(self, spec: PlotSpec) -> RenderResult:
        """Multi-panel overview: flux + power + residual + material."""
        if not HAS_MPL:
            return RenderResult(success=False, error="matplotlib required")

        # data_labels should contain: flux, power, residual, material
        flux = spec.data_labels.get("flux")
        power = spec.data_labels.get("power")
        residual = spec.data_labels.get("residual")
        material = spec.data_labels.get("material")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        output_paths = []

        # Flux heatmap
        if flux is not None:
            ax = axes[0, 0]
            im = ax.imshow(flux, cmap="inferno", origin="lower", aspect="auto")
            plt.colorbar(im, ax=ax)
            ax.set_title("Neutron Flux", fontweight="bold")

        # Power heatmap
        if power is not None:
            ax = axes[0, 1]
            im = ax.imshow(power, cmap="hot", origin="lower", aspect="auto")
            plt.colorbar(im, ax=ax)
            ax.set_title("Power Distribution", fontweight="bold")

        # Residual
        if residual is not None:
            ax = axes[1, 0]
            ax.semilogy(residual, "b-", linewidth=1.5)
            ax.set_title("Convergence History", fontweight="bold")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Residual")
            ax.grid(True, alpha=0.3)

        # Material map
        if material is not None:
            if isinstance(material, list):
                material = np.array(material, dtype=str)
            ax = axes[1, 1]
            unique = sorted(set(material.flat))
            cmap = plt.cm.tab20
            color_map = {m: cmap(i / len(unique)) for i, m in enumerate(unique)}
            ny, nx = material.shape
            for j in range(ny):
                for i in range(nx):
                    mat = material[j, i]
                    rect = plt.Rectangle((i, j), 1, 1,
                                         facecolor=color_map.get(mat, (0.5,)*4),
                                         edgecolor="white", linewidth=0.3)
                    ax.add_patch(rect)
            ax.set_xlim(0, nx)
            ax.set_ylim(0, ny)
            ax.set_aspect("equal")
            ax.set_title("Material Layout", fontweight="bold")

        fig.suptitle(spec.title or "Core Solver Dashboard", fontsize=15, fontweight="bold")
        output_path = str(self.output_dir / spec.output_path)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

        return RenderResult(output_paths=output_paths, plot_type="dashboard")


# ── Convenience builders (LLM-friendly) ────────────────────────────────

class PlotBuilder:
    """Fluent builder for PlotSpec, designed for LLM code generation.

    Example (LLM can generate this):
        spec = (PlotBuilder("heatmap_2d")
                .with_title("Power Distribution")
                .with_data(power_2d)
                .with_colormap("hot")
                .with_labels("X [cm]", "Y [cm]", "Power")
                .save_to("power_map.png")
                .build())
        renderer.render(spec)
    """

    def __init__(self, plot_type: str = "heatmap_2d"):
        self._spec = PlotSpec(plot_type=plot_type)

    def with_title(self, title: str) -> "PlotBuilder":
        self._spec.title = title
        return self

    def with_data(self, data: np.ndarray) -> "PlotBuilder":
        self._spec.data = data
        return self

    def with_x(self, x: np.ndarray) -> "PlotBuilder":
        self._spec.x = x
        return self

    def with_y(self, y: np.ndarray) -> "PlotBuilder":
        self._spec.y = y
        return self

    def with_colormap(self, cmap: str) -> "PlotBuilder":
        self._spec.colormap = cmap
        return self

    def with_labels(self, xlabel: str = "", ylabel: str = "",
                    zlabel: str = "") -> "PlotBuilder":
        self._spec.xlabel = xlabel
        self._spec.ylabel = ylabel
        self._spec.zlabel = zlabel
        return self

    def with_aux_data(self, label: str, data: np.ndarray) -> "PlotBuilder":
        self._spec.data_labels[label] = data
        return self

    def with_size(self, width: int, height: int) -> "PlotBuilder":
        self._spec.width = width
        self._spec.height = height
        return self

    def save_to(self, path: str) -> "PlotBuilder":
        self._spec.output_path = path
        return self

    def interactive(self, yes: bool = True) -> "PlotBuilder":
        self._spec.interactive = yes
        return self

    def build(self) -> PlotSpec:
        return self._spec
