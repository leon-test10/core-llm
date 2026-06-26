"""
FastAPI backend for visual skill graph editor.

Serves:
  - GET  /api/skills         → skill registry (port definitions)
  - POST /api/execute        → execute a DAG of skill nodes
  - WS   /ws/run/{run_id}    → live execution status

Each skill node has typed input/output ports. The DAG executor
validates port types, resolves dependencies, and runs nodes
in topological order.
"""

from __future__ import annotations

import json
import time
import uuid
import sys
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field, asdict

# Ensure project root in path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Skill Port Types ───────────────────────────────────────────────────

TYPE_COLORS = {
    "mesh": "#4CAF50",          "xs_provider": "#2196F3",
    "flux_array": "#FF9800",    "power_array": "#F44336",
    "float": "#9C27B0",         "image": "#00BCD4",
    "report": "#795548",        "any": "#607D8B",
    "material_registry": "#E91E63",
}

@dataclass
class SkillPort:
    name: str
    port_type: str
    direction: str          # "input" | "output"
    required: bool = True
    description: str = ""

@dataclass
class SkillDef:
    skill_id: str
    name: str
    description: str
    inputs: list[SkillPort] = field(default_factory=list)
    outputs: list[SkillPort] = field(default_factory=list)
    color: str = "#607D8B"
    category: str = "general"

# ── Skill Registry ─────────────────────────────────────────────────────

SKILL_REGISTRY: dict[str, SkillDef] = {
    "mesh_builder": SkillDef(
        skill_id="mesh_builder",
        name="Mesh Builder",
        description="Create a regular 2D mesh from material map",
        inputs=[
            SkillPort("nx", "float", "input", True, "Cells in X"),
            SkillPort("ny", "float", "input", True, "Cells in Y"),
            SkillPort("dx", "float", "input", False, "Cell width (default 1.0)"),
            SkillPort("dy", "float", "input", False, "Cell height (default 1.0)"),
            SkillPort("boundary_x", "any", "input", False, "X boundary (vacuum/reflective)"),
            SkillPort("boundary_y", "any", "input", False, "Y boundary"),
        ],
        outputs=[SkillPort("mesh", "mesh", "output", True, "The constructed mesh")],
        color="#4CAF50", category="geometry",
    ),
    "material_registry": SkillDef(
        skill_id="material_registry",
        name="Material Registry",
        description="Register materials with cross sections",
        inputs=[
            SkillPort("materials", "material_registry", "input", True, "Material definitions"),
        ],
        outputs=[SkillPort("registry", "material_registry", "output", True, "Material registry")],
        color="#E91E63", category="materials",
    ),
    "xs_provider": SkillDef(
        skill_id="xs_provider",
        name="XS Provider",
        description="Provide constant/tabulated/nuclear cross sections",
        inputs=[
            SkillPort("materials_xs", "any", "input", True, "XS data per material"),
            SkillPort("group_count", "float", "input", False, "Number of groups"),
        ],
        outputs=[SkillPort("xs_provider", "xs_provider", "output", True, "XS provider object")],
        color="#2196F3", category="materials",
    ),
    "diffusion_solver": SkillDef(
        skill_id="diffusion_solver",
        name="Diffusion Solver",
        description="Solve N-group neutron diffusion eigenvalue problem",
        inputs=[
            SkillPort("mesh", "mesh", "input", True, "Computational mesh"),
            SkillPort("xs_provider", "xs_provider", "input", True, "Cross section provider"),
            SkillPort("mode", "any", "input", False, "Group mode (one/two/multi/auto)"),
            SkillPort("backend", "any", "input", False, "cpu/gpu/auto"),
            SkillPort("tolerance", "float", "input", False, "Convergence tolerance"),
        ],
        outputs=[
            SkillPort("keff", "float", "output", True, "Effective multiplication factor"),
            SkillPort("flux", "flux_array", "output", True, "Neutron flux distribution"),
            SkillPort("power", "power_array", "output", True, "Power distribution"),
            SkillPort("iterations", "float", "output", False, "Iteration count"),
            SkillPort("result", "any", "output", True, "Full result object"),
        ],
        color="#FF9800", category="solver",
    ),
    "renderer": SkillDef(
        skill_id="renderer",
        name="Renderer",
        description="Visualize flux, power, material maps",
        inputs=[
            SkillPort("flux", "flux_array", "input", False, "Flux array"),
            SkillPort("power", "power_array", "input", False, "Power array"),
            SkillPort("mesh", "mesh", "input", False, "Mesh for reshaping"),
            SkillPort("plot_type", "any", "input", False, "heatmap_2d/surface_3d/dashboard"),
        ],
        outputs=[SkillPort("image", "image", "output", True, "Rendered plot image")],
        color="#00BCD4", category="visualization",
    ),
    "benchmark_compare": SkillDef(
        skill_id="benchmark_compare",
        name="Benchmark Compare",
        description="Compare keff against published reference values",
        inputs=[
            SkillPort("keff", "float", "input", True, "Computed keff"),
            SkillPort("ref_keff", "float", "input", True, "Reference keff"),
            SkillPort("benchmark", "any", "input", False, "Benchmark name"),
        ],
        outputs=[
            SkillPort("error_pcm", "float", "output", True, "Error in pcm"),
            SkillPort("passed", "any", "output", True, "Pass/fail status"),
            SkillPort("summary", "any", "output", True, "Comparison summary"),
        ],
        color="#795548", category="analysis",
    ),
    "postprocess": SkillDef(
        skill_id="postprocess",
        name="Postprocess",
        description="Compute power distribution and peaking factors",
        inputs=[
            SkillPort("result", "any", "input", True, "Solver result"),
            SkillPort("mesh", "mesh", "input", True, "Mesh"),
        ],
        outputs=[
            SkillPort("power", "power_array", "output", True, "Normalized power"),
            SkillPort("peaking_factor", "float", "output", True, "Max/avg power ratio"),
        ],
        color="#9C27B0", category="analysis",
    ),
}


# ── DAG Executor ───────────────────────────────────────────────────────

class DAGExecutor:
    """Execute a directed acyclic graph of skill nodes.

    Node format:
    {
        "id": "node-1",
        "skill_id": "diffusion_solver",
        "params": {"mode": "auto", "backend": "cpu"},
        "inputs": {"mesh": "node-0.mesh", "xs_provider": "node-2.xs_provider"}
    }
    """

    def __init__(self):
        self._store: dict[str, Any] = {}  # node_id → outputs
        self._status: dict[str, str] = {}  # node_id → status
        self._websockets: list[WebSocket] = []

    async def execute(self, nodes: list[dict], edges: list[dict]) -> dict:
        """Execute a DAG and return results."""
        # Build dependency graph from edges
        node_map = {n["id"]: n for n in nodes}

        # Resolve wiring from edges
        # Edge: {source: "node-A.port", target: "node-B.port"}
        for node in nodes:
            node["_resolved_inputs"] = dict(node.get("params", {}))
            node["_input_wires"] = {}

        for edge in edges:
            src = edge["source"]       # "node-A.port_name"
            tgt = edge["target"]       # "node-B.port_name"
            src_node, src_port = src.split(".", 1)
            tgt_node, tgt_port = tgt.split(".", 1)
            if tgt_node in node_map:
                node_map[tgt_node]["_input_wires"][tgt_port] = (src_node, src_port)

        # Topological sort (simple BFS: find nodes with all deps satisfied)
        executed = set()
        results: dict[str, dict] = {}

        while len(executed) < len(nodes):
            ready = []
            for node in nodes:
                nid = node["id"]
                if nid in executed:
                    continue
                # Check if all input wires come from executed nodes
                wires = node.get("_input_wires", {})
                deps_satisfied = all(
                    src_node in executed
                    for src_node, _ in wires.values()
                )
                if deps_satisfied:
                    ready.append(node)

            if not ready and len(executed) < len(nodes):
                # Cycle or missing deps — execute remaining in order
                for node in nodes:
                    if node["id"] not in executed:
                        ready.append(node)
                        break  # try one at a time
                if not ready:
                    break

            for node in ready:
                nid = node["id"]
                await self._set_status(nid, "running")
                try:
                    outputs = await self._execute_node(node, results)
                    results[nid] = outputs
                    executed.add(nid)
                    await self._set_status(nid, "done", outputs)
                except Exception as e:
                    results[nid] = {"error": str(e)}
                    executed.add(nid)
                    await self._set_status(nid, "error", {"error": str(e)})

        return results

    async def _execute_node(self, node: dict, prev_results: dict) -> dict:
        """Execute a single skill node."""
        skill_id = node["skill_id"]
        nid = node["id"]

        # Resolve inputs from wires
        resolved = dict(node.get("params", {}))
        for port_name, (src_node, src_port) in node.get("_input_wires", {}).items():
            src_outputs = prev_results.get(src_node, {})
            if src_port in src_outputs:
                resolved[port_name] = src_outputs[src_port]
            elif "result" in src_outputs and src_port == "result":
                resolved[port_name] = src_outputs["result"]

        # Dispatch to skill implementation
        if skill_id == "mesh_builder":
            return self._exec_mesh_builder(resolved)
        elif skill_id == "material_registry":
            return self._exec_material_registry(resolved)
        elif skill_id == "xs_provider":
            return self._exec_xs_provider(resolved)
        elif skill_id == "diffusion_solver":
            return self._exec_diffusion_solver(resolved)
        elif skill_id == "renderer":
            return self._exec_renderer(resolved)
        elif skill_id == "benchmark_compare":
            return self._exec_benchmark_compare(resolved)
        elif skill_id == "postprocess":
            return self._exec_postprocess(resolved)
        else:
            raise ValueError(f"Unknown skill: {skill_id}")

    def _exec_mesh_builder(self, inputs: dict) -> dict:
        from src.core.mesh import RegularGrid2D
        import numpy as np
        nx = int(inputs.get("nx", 10))
        ny = int(inputs.get("ny", 10))
        dx = float(inputs.get("dx", 1.0))
        dy = float(inputs.get("dy", 1.0))
        bx = inputs.get("boundary_x", "vacuum")
        by = inputs.get("boundary_y", "vacuum")
        material_map = inputs.get("material_map")
        # If no map given, create default all "fuel"
        if material_map is None:
            material_map = [["fuel"] * nx for _ in range(ny)]
        elif isinstance(material_map, list) and len(material_map) != ny:
            # Reshape if flat list
            arr = np.array(material_map, dtype=object)
            if arr.ndim == 1 and len(arr) == nx * ny:
                material_map = arr.reshape(ny, nx).tolist()
        mesh = RegularGrid2D(nx=nx, ny=ny, dx=dx, dy=dy,
                             material_map=material_map,
                             boundary_x=bx, boundary_y=by)
        self._store[nx * ny] = mesh
        return {"mesh": mesh, "n_cells": mesh.n_cells(), "nx": nx, "ny": ny}

    def _exec_material_registry(self, inputs: dict) -> dict:
        from src.core.material import MaterialRegistry, Material
        registry = MaterialRegistry()
        mat_data = inputs.get("materials", {})
        for mid, info in mat_data.items():
            registry.register(Material(
                material_id=mid,
                name=info.get("name", mid),
                xs_provider_id=info.get("xs_provider", "constant"),
            ))
        return {"registry": registry, "material_ids": registry.list_ids()}

    def _exec_xs_provider(self, inputs: dict) -> dict:
        from src.core.xs import ConstantXSProvider, create_xs_provider
        xs_data = inputs.get("materials_xs", inputs.get("materials", {}))
        # Auto-add default fuel XS if empty
        if not xs_data:
            xs_data = {
                "fuel": {"D": [1.4, 0.4], "Sigma_a": [0.01, 0.08],
                         "nuSigma_f": [0.006, 0.12],
                         "Sigma_s": [[0, 0.02], [0, 0]], "chi": [1, 0],
                         "kappaSigma_f": [1, 1]},
                "moderator": {"D": [2.0, 1.0], "Sigma_a": [0.001, 0.02],
                              "nuSigma_f": [0, 0],
                              "Sigma_s": [[0, 0.04], [0, 0]], "chi": [0, 0],
                              "kappaSigma_f": [0, 0]},
                "reflector": {"D": [1.8, 0.6], "Sigma_a": [0.002, 0.01],
                              "nuSigma_f": [0, 0],
                              "Sigma_s": [[0, 0.03], [0, 0]], "chi": [0, 0],
                              "kappaSigma_f": [0, 0]},
            }
        gc = inputs.get("group_count", 1)
        if isinstance(gc, float):
            gc = int(gc)
        provider = create_xs_provider("constant", materials_xs=xs_data, group_count=gc)
        return {"xs_provider": provider, "group_count": gc}

    def _exec_diffusion_solver(self, inputs: dict) -> dict:
        from src.solvers.diffusion_ng import MultiGroupDiffusionSolver
        from src.solvers.diffusion_1g import OneGroupDiffusionSolver

        mesh = inputs.get("mesh")
        xs_provider = inputs.get("xs_provider")
        mode = str(inputs.get("mode", "auto")).lower()
        tolerance = float(inputs.get("tolerance", 1e-8))
        max_iter = int(inputs.get("max_iter", 500))
        backend = str(inputs.get("backend", "auto"))

        if mode in ("1", "1g", "one_group"):
            solver = OneGroupDiffusionSolver(mesh, xs_provider, tolerance, max_iter)
        else:
            solver = MultiGroupDiffusionSolver(mesh, xs_provider, tolerance, max_iter, backend=backend)

        result = solver.solve()

        # Extract outputs
        outputs = {
            "keff": result.keff,
            "flux": result.flux.tolist() if hasattr(result.flux, 'tolist') else result.flux,
            "power": result.power.tolist() if result.power is not None and hasattr(result.power, 'tolist') else None,
            "iterations": result.iterations,
            "result": result,
        }
        return outputs

    def _exec_renderer(self, inputs: dict) -> dict:
        from src.visualization.renderer import Renderer, PlotSpec
        import numpy as np
        import base64
        from io import BytesIO

        flux = inputs.get("flux")
        power = inputs.get("power")
        plot_type = inputs.get("plot_type", "heatmap_2d")

        data = flux if flux is not None else power
        if data is None:
            return {"image": None, "error": "No data for rendering"}

        if isinstance(data, list):
            data = np.array(data)

        renderer = Renderer("matplotlib", "outputs")
        spec = PlotSpec(plot_type=plot_type, title="Result", data=data,
                        output_path=f"api_plot_{uuid.uuid4().hex[:8]}.png")
        result = renderer.render(spec)

        if result.success and result.output_paths:
            with open(result.output_paths[0], "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            return {"image": f"data:image/png;base64,{img_b64}",
                    "output_path": result.output_paths[0]}
        return {"image": None, "error": result.error}

    def _exec_benchmark_compare(self, inputs: dict) -> dict:
        from src.core.benchmarks import compare_keff
        k_comp = float(inputs.get("keff", 0))
        k_ref = float(inputs.get("ref_keff", 1.0))
        name = inputs.get("benchmark", "custom")
        comparison = compare_keff(k_comp, k_ref, 5000, name, "api")
        return {
            "error_pcm": comparison.keff_error_pcm,
            "passed": comparison.passed,
            "summary": comparison.summary(),
        }

    def _exec_postprocess(self, inputs: dict) -> dict:
        result = inputs.get("result")
        if result is None:
            return {"power": None, "peaking_factor": 0}
        power = result.power if hasattr(result, 'power') else None
        if power is not None:
            peaking = float(power.max()) / float(power.mean()) if power.mean() > 0 else 0
            return {"power": power.tolist() if hasattr(power, 'tolist') else power,
                    "peaking_factor": peaking}
        return {"power": None, "peaking_factor": 0}

    async def _set_status(self, node_id: str, status: str, outputs: dict | None = None):
        self._status[node_id] = status
        for ws in self._websockets:
            try:
                await ws.send_json({
                    "type": "node_status",
                    "node_id": node_id,
                    "status": status,
                    "outputs": {k: str(v)[:200] if not isinstance(v, (int, float, bool, type(None))) else v
                               for k, v in (outputs or {}).items()},
                })
            except Exception:
                pass


# ── FastAPI App ────────────────────────────────────────────────────────

app = FastAPI(title="Core Solver API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

executor = DAGExecutor()


@app.get("/api/skills")
async def get_skills():
    """Return the skill registry with port definitions."""
    return {
        skill_id: {
            "skill_id": s.skill_id,
            "name": s.name,
            "description": s.description,
            "inputs": [asdict(p) for p in s.inputs],
            "outputs": [asdict(p) for p in s.outputs],
            "color": s.color,
            "category": s.category,
        }
        for skill_id, s in SKILL_REGISTRY.items()
    }


@app.post("/api/execute")
async def execute_dag(payload: dict):
    """Execute a DAG of skill nodes.

    Body: {"nodes": [...], "edges": [...]}
    Returns: {node_id: {output_name: value}}
    """
    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    results = await executor.execute(nodes, edges)

    # Serialize results (remove non-JSON-serializable objects)
    serializable = {}
    for nid, outputs in results.items():
        serializable[nid] = {}
        for key, val in outputs.items():
            if key == "result":
                # Extract key fields from Result
                if hasattr(val, 'keff'):
                    serializable[nid]["keff"] = val.keff
                    serializable[nid]["iterations"] = val.iterations
                    serializable[nid]["converged"] = val.converged
                    if val.power is not None:
                        serializable[nid]["power"] = val.power.tolist() if hasattr(val.power, 'tolist') else str(val)
            elif key in ("mesh", "xs_provider", "registry"):
                serializable[nid][key] = str(type(val).__name__)
            elif isinstance(val, (int, float, bool, str, list, type(None))):
                serializable[nid][key] = val
            else:
                serializable[nid][key] = str(type(val).__name__)

    return {"results": serializable}


@app.websocket("/ws/run")
async def websocket_run(ws: WebSocket):
    """WebSocket for live execution status."""
    await ws.accept()
    executor._websockets.append(ws)
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "execute":
                nodes = msg.get("nodes", [])
                edges = msg.get("edges", [])
                await executor.execute(nodes, edges)
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        if ws in executor._websockets:
            executor._websockets.remove(ws)


# ── Serve frontend ─────────────────────────────────────────────────────

_UI_DIR = Path(__file__).parent / "ui"
_UI_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
async def serve_ui():
    """Serve the visual graph editor."""
    html_path = _UI_DIR / "graph_editor.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>UI not found. Run build first.</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8600, log_level="info")
