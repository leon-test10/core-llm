"""
Streamlit UI for Core Solver Demo.

Provides interactive visualization of assembly modeling,
solver execution, and result analysis with GPU backend support.

Usage:
    streamlit run src/ui_app.py
"""

import sys
import os
import time
from pathlib import Path

# Ensure the project root is in sys.path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

from src.core.mesh import RegularGrid2D
from src.core.material import MaterialRegistry, Material, CellState
from src.core.xs import ConstantXSProvider, NuclearDataLibrary
from src.core.xs_general import MultiGroupXS
from src.core.backends import gpu_available, gpu_vendor, gpu_info, get_backend
from src.core.benchmarks import (
    C5G7Benchmark, Takeda1Benchmark, IAEA2DBenchmark,
    compare_keff, BenchmarkResult,
)
from src.solvers.diffusion_ng import MultiGroupDiffusionSolver
from src.solvers.diffusion_1g import OneGroupDiffusionSolver
from src.visualization.renderer import Renderer, PlotSpec, PlotBuilder

# ── Page Config ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Core Solver Demo",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State ──────────────────────────────────────────────────────

DEFAULT_MATERIALS = {
    "fuel": {
        "name": "UO2 Fuel",
        "D": [1.4, 0.4],
        "Sigma_a": [0.010, 0.080],
        "nuSigma_f": [0.006, 0.120],
        "Sigma_s": [[0.0, 0.020], [0.0, 0.0]],
        "chi": [1.0, 0.0],
        "kappaSigma_f": [1.0, 1.0],
    },
    "moderator": {
        "name": "Water Moderator",
        "D": [2.0, 1.0],
        "Sigma_a": [0.001, 0.020],
        "nuSigma_f": [0.0, 0.0],
        "Sigma_s": [[0.0, 0.040], [0.0, 0.0]],
        "chi": [0.0, 0.0],
        "kappaSigma_f": [0.0, 0.0],
    },
    "control_rod": {
        "name": "B4C Control Rod",
        "D": [1.2, 0.3],
        "Sigma_a": [0.010, 1.000],
        "nuSigma_f": [0.0, 0.0],
        "Sigma_s": [[0.0, 0.020], [0.0, 0.0]],
        "chi": [0.0, 0.0],
        "kappaSigma_f": [0.0, 0.0],
    },
    "reflector": {
        "name": "Reflector",
        "D": [1.8, 0.6],
        "Sigma_a": [0.002, 0.010],
        "nuSigma_f": [0.0, 0.0],
        "Sigma_s": [[0.0, 0.030], [0.0, 0.0]],
        "chi": [0.0, 0.0],
        "kappaSigma_f": [0.0, 0.0],
    },
}

if "materials" not in st.session_state:
    st.session_state.materials = DEFAULT_MATERIALS.copy()

if "material_map" not in st.session_state:
    st.session_state.material_map = None

if "solver_result" not in st.session_state:
    st.session_state.solver_result = None

if "benchmark_result" not in st.session_state:
    st.session_state.benchmark_result = None

if "convergence_history" not in st.session_state:
    st.session_state.convergence_history = []

if "mesh_size" not in st.session_state:
    st.session_state.mesh_size = (10, 10)

if "backend" not in st.session_state:
    st.session_state.backend = "auto"


# ── Sidebar ────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.title("⚛️ Core Solver")
        st.caption("Interactive Reactor Physics Demo")

        # GPU Status
        if gpu_available():
            st.success(f"⚡ GPU Available: {gpu_vendor().upper()}")
            st.session_state.backend = st.selectbox(
                "Backend", ["auto", "gpu", "cpu", "cuda", "rocm"],
                index=0
            )
        else:
            st.info("💻 Running on CPU")
            st.caption("`pip install cupy-rocm-*-12x` for AMD GPU")
            st.session_state.backend = "cpu"

        st.divider()

        # Quick presets
        st.subheader("Quick Presets")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎯 Takeda-1", use_container_width=True):
                load_takeda_preset()
            if st.button("🔬 C5G7 7G", use_container_width=True):
                load_c5g7_preset()
        with col2:
            if st.button("🏭 IAEA PWR", use_container_width=True):
                load_iaea_preset()
            if st.button("🧪 Toy Demo", use_container_width=True):
                load_toy_preset()

        st.divider()

        # Mesh controls
        st.subheader("Mesh")
        nx = st.slider("Cells (X)", 2, 40, st.session_state.mesh_size[0], key="nx_slider")
        ny = st.slider("Cells (Y)", 2, 40, st.session_state.mesh_size[1], key="ny_slider")
        st.session_state.mesh_size = (nx, ny)

        st.divider()
        st.caption(f"v1.0 | GPU: {gpu_info()}")


def load_takeda_preset():
    """Load Takeda-1 benchmark preset."""
    st.session_state.materials = {
        "core": {
            "name": "Fast Reactor Core",
            "D": [1.4, 0.4], "Sigma_a": [0.010, 0.080],
            "nuSigma_f": [0.006, 0.120], "Sigma_s": [[0.0, 0.020], [0.0, 0.0]],
            "chi": [1.0, 0.0], "kappaSigma_f": [1.0, 1.0],
        },
        "blanket": {
            "name": "Radial Blanket",
            "D": [1.4, 0.4], "Sigma_a": [0.005, 0.040],
            "nuSigma_f": [0.0, 0.0], "Sigma_s": [[0.0, 0.020], [0.0, 0.0]],
            "chi": [0.0, 0.0], "kappaSigma_f": [0.0, 0.0],
        },
        "reflector": {
            "name": "Reflector",
            "D": [1.2, 0.3], "Sigma_a": [0.001, 0.010],
            "nuSigma_f": [0.0, 0.0], "Sigma_s": [[0.0, 0.030], [0.0, 0.0]],
            "chi": [0.0, 0.0], "kappaSigma_f": [0.0, 0.0],
        },
    }
    st.session_state.mesh_size = (40, 40)
    st.rerun()

def load_c5g7_preset():
    """Load C5G7 7-group homogenized preset."""
    b = C5G7Benchmark()
    st.session_state.materials = {
        "uo2": {"name": "UO2", "D": b.uo2["D"], "Sigma_a": b.uo2["Sigma_a"],
                "nuSigma_f": b.uo2["nuSigma_f"], "chi": b.uo2["chi"],
                "Sigma_s": b.uo2["Sigma_s"], "kappaSigma_f": b.uo2["kappaSigma_f"]},
        "mox43": {"name": "MOX 4.3%", "D": b.mox43["D"], "Sigma_a": b.mox43["Sigma_a"],
                  "nuSigma_f": b.mox43["nuSigma_f"], "chi": b.mox43["chi"],
                  "Sigma_s": b.mox43["Sigma_s"], "kappaSigma_f": b.mox43["kappaSigma_f"]},
        "mox70": {"name": "MOX 7.0%", "D": b.mox70["D"], "Sigma_a": b.mox70["Sigma_a"],
                  "nuSigma_f": b.mox70["nuSigma_f"], "chi": b.mox70["chi"],
                  "Sigma_s": b.mox70["Sigma_s"], "kappaSigma_f": b.mox70["kappaSigma_f"]},
        "mox87": {"name": "MOX 8.7%", "D": b.mox87["D"], "Sigma_a": b.mox87["Sigma_a"],
                  "nuSigma_f": b.mox87["nuSigma_f"], "chi": b.mox87["chi"],
                  "Sigma_s": b.mox87["Sigma_s"], "kappaSigma_f": b.mox87["kappaSigma_f"]},
    }
    st.session_state.mesh_size = (2, 2)
    st.rerun()

def load_iaea_preset():
    st.session_state.materials = {
        "fuel_inner": {"name": "Inner Fuel", "D": [1.5, 0.4], "Sigma_a": [0.010, 0.085],
                       "nuSigma_f": [0.006, 0.120], "Sigma_s": [[0.0, 0.020], [0.0, 0.0]],
                       "chi": [1.0, 0.0], "kappaSigma_f": [1.0, 1.0]},
        "fuel_outer": {"name": "Outer Fuel", "D": [1.5, 0.4], "Sigma_a": [0.010, 0.080],
                       "nuSigma_f": [0.006, 0.110], "Sigma_s": [[0.0, 0.020], [0.0, 0.0]],
                       "chi": [1.0, 0.0], "kappaSigma_f": [1.0, 1.0]},
        "reflector": {"name": "Reflector", "D": [1.3, 0.3], "Sigma_a": [0.0, 0.015],
                      "nuSigma_f": [0.0, 0.0], "Sigma_s": [[0.0, 0.040], [0.0, 0.0]],
                      "chi": [0.0, 0.0], "kappaSigma_f": [0.0, 0.0]},
    }
    st.session_state.mesh_size = (30, 30)
    st.rerun()

def load_toy_preset():
    st.session_state.materials = DEFAULT_MATERIALS.copy()
    st.session_state.mesh_size = (10, 10)
    st.rerun()


# ── Assembly Builder Tab ───────────────────────────────────────────────

def render_assembly_builder():
    st.header("🔧 Assembly Builder")
    st.caption("Click cells to assign materials, then build the mesh.")

    materials = st.session_state.materials
    mat_ids = list(materials.keys())
    nx, ny = st.session_state.mesh_size

    col_left, col_right = st.columns([3, 1])

    with col_left:
        # Material map editor
        if st.session_state.material_map is None or \
           st.session_state.material_map.shape != (ny, nx):
            # Initialize with first material
            default_mat = mat_ids[0] if mat_ids else "fuel"
            st.session_state.material_map = np.full((ny, nx), default_mat, dtype=object)

        # Show grid with clickable cells
        selected_mat = st.selectbox("Paint Material", mat_ids, key="paint_mat")

        # Render grid as interactive dataframe
        st.caption(f"Material Map ({nx}×{ny}) — click to edit cells")
        map_df = pd.DataFrame(st.session_state.material_map)

        edited_df = st.data_editor(
            map_df,
            column_config={
                col: st.column_config.SelectboxColumn(
                    str(col), options=mat_ids, width="small"
                ) for col in map_df.columns
            },
            hide_index=True,
            use_container_width=True,
            height=min(400, ny * 25 + 38),
            key="material_editor",
        )

        # Update session state from editor
        if edited_df is not None:
            st.session_state.material_map = edited_df.values

    with col_right:
        st.subheader("Material Preview")
        # Color-coded legend
        colors = plt.cm.tab20(np.linspace(0, 1, len(mat_ids)))
        for i, mat_id in enumerate(mat_ids):
            mat = materials[mat_id]
            color_hex = mcolors.rgb2hex(colors[i])
            st.markdown(
                f'<div style="display:flex;align-items:center;margin:4px 0">'
                f'<div style="width:20px;height:20px;background:{color_hex};'
                f'border-radius:3px;margin-right:8px"></div>'
                f'<span style="font-size:0.85em"><b>{mat_id}</b><br/>'
                f'<span style="color:#888">{mat["name"]}</span></span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.subheader("Material Editor")
        edit_mat = st.selectbox("Edit Material", mat_ids, key="edit_mat_select")

        mat_data = materials[edit_mat]
        st.caption(f"**{mat_data['name']}** ({edit_mat})")

        # Editable cross sections
        new_name = st.text_input("Name", mat_data["name"], key="mat_name")
        D_str = st.text_input("D", str(mat_data["D"]), key="mat_D")
        Sa_str = st.text_input("Sigma_a", str(mat_data["Sigma_a"]), key="mat_Sa")
        nSf_str = st.text_input("nuSigma_f", str(mat_data["nuSigma_f"]), key="mat_nSf")

        if st.button("Update Material", key="update_mat"):
            try:
                updated = materials[edit_mat].copy()
                updated["name"] = new_name
                updated["D"] = eval(D_str)
                updated["Sigma_a"] = eval(Sa_str)
                updated["nuSigma_f"] = eval(nSf_str)
                st.session_state.materials[edit_mat] = updated
                st.success(f"Updated {edit_mat}")
            except Exception as e:
                st.error(f"Invalid: {e}")

    # Build & visualize material map
    if st.button("🎨 Render Material Map", use_container_width=True):
        renderer = Renderer("matplotlib", "outputs")
        spec = PlotSpec(
            plot_type="material_map",
            title=f"Material Layout ({nx}×{ny})",
            data=st.session_state.material_map,
            output_path="ui_material_map.png",
        )
        result = renderer.render(spec)
        if result.success:
            st.image(result.output_paths[0], use_container_width=True)


# ── Solver Tab ─────────────────────────────────────────────────────────

def render_solver():
    st.header("🧮 Solver")
    st.caption("Configure and run the diffusion solver.")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_groups = st.selectbox("Energy Groups", [1, 2, 7, 47], index=1,
                                 help="Number of energy groups")
    with col2:
        tolerance = st.selectbox("Tolerance", [1e-6, 1e-8, 1e-10, 1e-12], index=1,
                                  format_func=lambda x: f"{x:.0e}")
    with col3:
        max_iter = st.number_input("Max Iterations", 100, 5000, 500, 100)

    bc_type = st.radio("Boundary Condition", ["vacuum", "reflective"],
                        horizontal=True)

    if st.button("🚀 Run Solver", type="primary", use_container_width=True):
        run_solver(n_groups, tolerance, max_iter, bc_type)

    # Show results if available
    if st.session_state.solver_result is not None:
        result = st.session_state.solver_result
        st.divider()
        st.subheader("Results")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("k_eff", f"{result.keff:.6f}")
        with col2:
            st.metric("Iterations", result.iterations)
        with col3:
            st.metric("Converged", "✅" if result.converged else "❌")
        with col4:
            if result.power is not None:
                st.metric("Peak Power", f"{result.power.max():.4f}")

        # Flux & Power heatmaps
        col_left, col_right = st.columns(2)
        with col_left:
            st.caption("Neutron Flux")
            plot_flux_power(result, st.session_state.mesh_size, "flux")
        with col_right:
            st.caption("Power Distribution")
            plot_flux_power(result, st.session_state.mesh_size, "power")

        # Convergence plot
        if result.residual_history:
            st.caption("Convergence History")
            fig, ax = plt.subplots(figsize=(8, 2.5))
            ax.semilogy(result.residual_history, "b-", linewidth=1.5)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Residual")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)


def run_solver(n_groups, tolerance, max_iter, bc_type):
    """Execute the solver and update session state."""
    materials = st.session_state.materials
    material_map = st.session_state.material_map
    nx, ny = st.session_state.mesh_size

    if material_map is None:
        st.error("Please build an assembly first (Assembly Builder tab).")
        return

    # Build mesh
    progress = st.progress(0, "Building mesh...")
    mesh = RegularGrid2D(
        nx=nx, ny=ny, dx=1.0, dy=1.0,
        material_map=material_map.tolist(),
        boundary_x=bc_type, boundary_y=bc_type,
    )

    # Prepare cross sections for the selected number of groups
    progress.progress(10, "Preparing cross sections...")
    xs_data = {}
    for mat_id, mat_info in materials.items():
        D = mat_info["D"]
        if n_groups == 1:
            xs_data[mat_id] = {
                "D": float(D[0]) if isinstance(D, list) else float(D),
                "Sigma_a": float(mat_info["Sigma_a"][0]) if isinstance(mat_info["Sigma_a"], list) else float(mat_info["Sigma_a"]),
                "nuSigma_f": float(mat_info["nuSigma_f"][0]) if isinstance(mat_info["nuSigma_f"], list) else float(mat_info["nuSigma_f"]),
                "kappaSigma_f": 1.0,
            }
        else:
            # Truncate or pad to n_groups
            def pad_to(arr, n, default=0.0):
                if isinstance(arr, list):
                    if len(arr) >= n:
                        return arr[:n]
                    return arr + [default] * (n - len(arr))
                return [float(arr)] * n

            xs_data[mat_id] = {
                "D": pad_to(D, n_groups, 1.0),
                "Sigma_a": pad_to(mat_info["Sigma_a"], n_groups, 0.01),
                "nuSigma_f": pad_to(mat_info["nuSigma_f"], n_groups, 0.0),
                "chi": pad_to(mat_info.get("chi", [1.0] + [0.0]*(n_groups-1)), n_groups, 0.0),
                "Sigma_s": mat_info.get("Sigma_s", [[0.0]*n_groups]*n_groups),
                "kappaSigma_f": pad_to(mat_info.get("kappaSigma_f", [1.0]), n_groups, 1.0),
            }

    provider = ConstantXSProvider(materials_xs=xs_data, group_count=n_groups)

    # Run solver
    progress.progress(20, "Running solver...")
    backend = st.session_state.backend

    if n_groups <= 2:
        solver = OneGroupDiffusionSolver(
            mesh=mesh, xs_provider=provider,
            tolerance=tolerance, max_iter=max_iter,
        )
    else:
        solver = MultiGroupDiffusionSolver(
            mesh=mesh, xs_provider=provider,
            tolerance=tolerance, max_iter=max_iter,
            backend=backend,
        )

    t0 = time.time()
    result = solver.solve()
    elapsed = time.time() - t0

    progress.progress(90, "Post-processing...")
    st.session_state.solver_result = result
    st.session_state.convergence_history = result.residual_history

    progress.progress(100, "Done!")
    time.sleep(0.3)
    progress.empty()

    st.success(f"Solved in {elapsed:.2f}s ({result.iterations} iterations) | "
               f"Backend: {result.solver.split('_')[-1]}")


def plot_flux_power(result, mesh_size, field="flux"):
    """Render flux or power heatmap inline."""
    data = result.flux if field == "flux" else result.power
    if data is None:
        return
    nx, ny = mesh_size

    # Handle multi-group flux: show total
    if data.ndim == 2 and data.shape[0] > 1:
        data_2d = data.sum(axis=0).reshape(ny, nx)
    elif data.ndim == 2:
        data_2d = data.reshape(ny, nx)
    else:
        data_2d = data.reshape(ny, nx)

    fig, ax = plt.subplots(figsize=(4, 4))
    cmap = "inferno" if field == "flux" else "hot"
    im = ax.imshow(data_2d, cmap=cmap, origin="lower", aspect="auto",
                   interpolation="bilinear")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{'Flux' if field == 'flux' else 'Power'} Distribution")
    st.pyplot(fig)
    plt.close(fig)


# ── Benchmark Tab ──────────────────────────────────────────────────────

def render_benchmark():
    st.header("📊 Benchmark Comparison")
    st.caption("Compare solver results against published reference values.")

    benchmark_options = {
        "Takeda-1 (2-group quarter-core)": "takeda1",
        "IAEA 2D PWR (2-group)": "iaea2d",
        "C5G7 homogenized (7-group)": "c5g7",
    }
    selected = st.selectbox("Benchmark", list(benchmark_options.keys()))

    ref_keff_map = {
        "takeda1": 0.9779,
        "iaea2d": 1.029,
        "c5g7": 1.18655,
    }
    bm_key = benchmark_options[selected]
    ref_keff = ref_keff_map[bm_key]
    tolerance_pcm = st.number_input("Tolerance (pcm)", 10, 100000, 5000, 100)

    if st.button("🏆 Run Benchmark Comparison", use_container_width=True):
        if st.session_state.solver_result is None:
            st.warning("Run the solver first (Solver tab).")
        else:
            result = st.session_state.solver_result
            comparison = compare_keff(
                k_comp=result.keff,
                k_ref=ref_keff,
                tolerance_pcm=tolerance_pcm,
                benchmark_name=selected,
                solver=result.solver,
            )
            st.session_state.benchmark_result = comparison

    if st.session_state.benchmark_result is not None:
        bm = st.session_state.benchmark_result
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("k_eff (computed)", f"{bm.keff_computed:.6f}")
        with col2:
            st.metric("k_eff (reference)", f"{bm.keff_reference:.6f}")
        with col3:
            delta = f"{bm.keff_error_pcm:.1f} pcm"
            st.metric("Error", delta,
                      delta="PASS" if bm.passed else "FAIL",
                      delta_color="normal" if bm.passed else "inverse")

        st.text(bm.summary())


# ── Agent Tab ──────────────────────────────────────────────────────────

def render_agent():
    st.header("🤖 Agent Interface")
    st.caption("Natural language control for the solver. Describe what you want to compute.")

    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []

    # Chat display
    for msg in st.session_state.agent_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input
    prompt = st.chat_input("e.g. 'Build a 10x10 core with fuel in center and a control rod, run diffusion, show power'")

    if prompt:
        st.session_state.agent_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            response = process_agent_prompt(prompt)
            st.write(response)

        st.session_state.agent_history.append({"role": "assistant", "content": response})


def process_agent_prompt(prompt: str) -> str:
    """Simple rule-based agent that interprets natural language prompts.

    In a production system, this would call an LLM. Here we use keyword matching.
    """
    prompt_lower = prompt.lower()

    # Detect intent
    actions = []
    response_parts = []

    # Geometry
    if any(w in prompt_lower for w in ["build", "create", "make", "assembly", "core", "mesh"]):
        # Extract size
        import re
        size_match = re.search(r'(\d+)\s*[x×]\s*(\d+)', prompt_lower)
        if size_match:
            nx, ny = int(size_match.group(1)), int(size_match.group(2))
            st.session_state.mesh_size = (nx, ny)
            actions.append(f"Set mesh to {nx}×{ny}")

        # Detect control rod
        if "control rod" in prompt_lower or "rod" in prompt_lower:
            actions.append("Added control rod to center")
            # Auto-modify material map
            nx, ny = st.session_state.mesh_size
            mat_map = np.full((ny, nx), "moderator", dtype=object)
            cx, cy = nx // 2, ny // 2
            for j in range(ny):
                for i in range(nx):
                    if (i - cx)**2 + (j - cy)**2 < (min(nx, ny) // 3)**2:
                        mat_map[j, i] = "fuel"
            mat_map[cy, cx] = "control_rod" if "control rod" in prompt_lower else "fuel"
            st.session_state.material_map = mat_map
            actions.append("Generated material layout")

    # Materials
    if "fuel" in prompt_lower or "material" in prompt_lower:
        if "uo2" in prompt_lower:
            actions.append("Using UO2 fuel cross sections")

    # Solve
    if any(w in prompt_lower for w in ["solve", "run", "compute", "diffusion"]):
        actions.append("Running diffusion solver...")
        try:
            run_solver(
                n_groups=2,
                tolerance=1e-8,
                max_iter=500,
                bc_type="vacuum",
            )
            if st.session_state.solver_result:
                r = st.session_state.solver_result
                actions.append(f"Done: k_eff={r.keff:.6f}, {r.iterations} iterations")
        except Exception as e:
            actions.append(f"Solver error: {e}")

    # Visualize
    if any(w in prompt_lower for w in ["show", "plot", "visualize", "power", "flux", "see"]):
        actions.append("Generating visualizations...")
        if st.session_state.solver_result is not None:
            actions.append("Power and flux heatmaps available in Solver tab")

    # Benchmark
    if any(w in prompt_lower for w in ["benchmark", "compare", "validate", "reference"]):
        actions.append("Running benchmark comparison...")

    if not actions:
        return ("I can help you:\n"
                "- Build a core: 'Build a 10x10 core with fuel and a control rod'\n"
                "- Run solver: 'Run diffusion with 2 groups'\n"
                "- Visualize: 'Show power distribution'\n"
                "- Benchmark: 'Compare with Takeda benchmark'\n\n"
                "Try combining: 'Build 12x12 core with UO2 fuel, run solver, show power'")

    response_parts.append("**Actions taken:**")
    for a in actions:
        response_parts.append(f"- ✅ {a}")

    response_parts.append("\n💡 Tip: Switch to **Solver** tab to see detailed results, "
                          "or **Assembly Builder** to edit the layout.")

    return "\n".join(response_parts)


# ── Main App ───────────────────────────────────────────────────────────

def main():
    render_sidebar()

    tabs = st.tabs([
        "🔧 Assembly Builder",
        "🧮 Solver",
        "📊 Benchmark",
        "🤖 Agent",
    ])

    with tabs[0]:
        render_assembly_builder()

    with tabs[1]:
        render_solver()

    with tabs[2]:
        render_benchmark()

    with tabs[3]:
        render_agent()


if __name__ == "__main__":
    main()
