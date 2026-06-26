"""
Workflow engine: parses YAML input, runs skills in sequence,
passes data between steps.
"""

import yaml
from pathlib import Path
from typing import Any

from ..core.mesh import RegularGrid2D
from ..core.material import Material, MaterialRegistry, CellState
from ..core.xs import (
    XSProvider, ConstantXSProvider, TabulatedXSProvider,
    NuclearDataLibrary, create_xs_provider,
)
from ..core.result import Result

from ..solvers.diffusion_1g import OneGroupDiffusionSolver
from ..solvers.diffusion_2g import TwoGroupDiffusionSolver
from ..solvers.postprocess import PowerPostprocessor, ReportWriter

from ..visualization.renderer import Renderer, PlotSpec
from ..visualization import plots as viz_plots


class WorkflowEngine:
    """Execute a workflow defined in a YAML file.

    The workflow file defines steps that produce named outputs.
    Outputs from earlier steps are available as {{ step_id.field }} templates.
    """

    def __init__(self):
        self._store: dict[str, Any] = {}  # {step_id: {field: value}}

    def run(self, workflow_path: str | Path) -> dict[str, Any]:
        """Execute the workflow defined in the given YAML file."""
        workflow_path = Path(workflow_path)

        with open(workflow_path) as f:
            spec = yaml.safe_load(f)

        workflow = spec.get("workflow", spec)
        name = workflow.get("name", workflow_path.stem)
        steps = workflow.get("steps", [])

        print(f"Workflow: {name}")
        print(f"  {len(steps)} steps")

        for step in steps:
            step_id = step["id"]
            skill = step.get("skill", step_id)
            raw_input = step.get("input", {})

            # Resolve template references {{ step_id.field }}
            resolved_input = self._resolve_input(raw_input)

            print(f"  [{step_id}] skill={skill}")

            # Execute the skill
            outputs = self._execute_skill(skill, resolved_input, step_id)
            self._store[step_id] = outputs

        # Collect final outputs
        final_outputs = {}
        if "outputs" in workflow:
            for output_ref in workflow["outputs"]:
                resolved = self._resolve_template(output_ref)
                final_outputs[output_ref] = resolved
        else:
            final_outputs = dict(self._store)

        return final_outputs

    def _resolve_input(self, raw: dict) -> dict:
        """Resolve {{ ... }} templates in input values."""
        resolved = {}
        for key, value in raw.items():
            if isinstance(value, str) and "{{" in value:
                resolved[key] = self._resolve_template(value)
            elif isinstance(value, list):
                resolved[key] = [
                    self._resolve_template(v) if isinstance(v, str) and "{{" in v else v
                    for v in value
                ]
            else:
                resolved[key] = value
        return resolved

    def _resolve_template(self, template: str) -> Any:
        """Resolve a {{ step_id.field }} template.

        Supports chaining: {{ step_id.field.subfield }}
        """
        import re
        result = template
        pattern = r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s*\}\}'

        def replacer(match):
            path = match.group(1).strip()
            parts = path.split(".")
            current = self._store
            for p in parts:
                if isinstance(current, dict):
                    current = current.get(p)
                elif hasattr(current, p):
                    current = getattr(current, p)
                elif hasattr(current, 'get'):
                    current = current.get(p)
                else:
                    return match.group(0)  # unresolved
            if current is None:
                return match.group(0)
            return current

        # If the entire string is a template, return the object directly
        if re.fullmatch(r'\{\{\s*[a-zA-Z_][a-zA-Z0-9_.]*\s*\}\}', template.strip()):
            path = template.strip().strip("{}").strip()
            parts = path.split(".")
            current = self._store
            for p in parts:
                if isinstance(current, dict):
                    current = current.get(p)
                elif hasattr(current, p):
                    current = getattr(current, p)
                else:
                    return template
            return current

        return re.sub(pattern, replacer, result)

    def _execute_skill(self, skill: str, inputs: dict, step_id: str) -> dict:
        """Dispatch to the appropriate skill implementation."""
        if skill in ("mesh_builder", "build_mesh"):
            return self._skill_mesh_builder(inputs)
        elif skill in ("material_registry", "register_materials"):
            return self._skill_material_registry(inputs)
        elif skill in ("xs_provider_constant", "constant_xs"):
            return self._skill_xs_constant(inputs)
        elif skill in ("xs_provider_nuclear", "nuclear_library"):
            return self._skill_xs_nuclear(inputs)
        elif skill in ("steady_diffusion_solver", "diffusion_solver", "solve"):
            return self._skill_diffusion_solver(inputs)
        elif skill in ("power_postprocess", "postprocess"):
            return self._skill_postprocess(inputs)
        elif skill in ("report_writer", "write_report"):
            return self._skill_report_writer(inputs)
        elif skill in ("renderer", "render", "visualize"):
            return self._skill_renderer(inputs)
        elif skill in ("benchmark_compare", "compare_benchmark"):
            return self._skill_benchmark_compare(inputs)
        else:
            raise ValueError(f"Unknown skill: {skill}")

    # ── Skill implementations ─────────────────────────────────────────────

    def _skill_mesh_builder(self, inputs: dict) -> dict:
        mesh_type = inputs.get("type", "regular_2d")
        if mesh_type == "regular_2d":
            nx = inputs["nx"]
            ny = inputs["ny"]
            dx = inputs.get("dx", 1.0)
            dy = inputs.get("dy", 1.0)

            # Material map: can be a CSV file or inline list
            material_map = inputs.get("material_map")
            if isinstance(material_map, str):
                # Load from CSV
                import csv
                mat = []
                with open(material_map) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        mat.append([cell.strip() for cell in row if cell.strip()])
                material_map = mat

            mesh = RegularGrid2D(
                nx=nx, ny=ny, dx=dx, dy=dy,
                material_map=material_map,
                boundary_x=inputs.get("boundary_x", "vacuum"),
                boundary_y=inputs.get("boundary_y", "vacuum"),
            )
            return {"mesh": mesh, "mesh_info": mesh.to_dict()}
        else:
            raise ValueError(f"Unknown mesh type: {mesh_type}")

    def _skill_material_registry(self, inputs: dict) -> dict:
        registry = MaterialRegistry()

        # Load from file or inline
        materials_data = inputs.get("materials")
        if isinstance(materials_data, str):
            with open(materials_data) as f:
                materials_data = yaml.safe_load(f)
        if materials_data is None and "materials_file" in inputs:
            with open(inputs["materials_file"]) as f:
                materials_data = yaml.safe_load(f)

        if materials_data is None:
            raise ValueError("No materials defined")

        for mat_id, mat_info in materials_data.items():
            mat = Material(
                material_id=mat_id,
                name=mat_info.get("name", mat_id),
                xs_provider_id=mat_info.get("xs_provider", mat_info.get("type", "constant")),
                metadata=mat_info.get("metadata", {}),
            )
            registry.register(mat)

        return {"materials": registry, "material_ids": registry.list_ids()}

    def _skill_xs_constant(self, inputs: dict) -> dict:
        """Build constant XS provider from inline or file data."""
        materials_xs = inputs.get("materials_xs", {})
        if isinstance(materials_xs, str):
            with open(materials_xs) as f:
                materials_xs = yaml.safe_load(f)
        # Also check if materials from registry contain XS
        if not materials_xs and "materials" in inputs:
            materials_data = inputs["materials"]
            if isinstance(materials_data, dict):
                materials_xs = materials_data

        group_count = inputs.get("group_count", inputs.get("mode", 1))
        if group_count == "one_group" or group_count == 1:
            group_count = 1
        elif group_count == "two_group" or group_count == 2:
            group_count = 2

        provider = create_xs_provider(
            "constant",
            materials_xs=materials_xs,
            group_count=group_count,
        )
        return {"xs_provider": provider, "group_count": group_count}

    def _skill_xs_nuclear(self, inputs: dict) -> dict:
        """Build nuclear data library XS provider."""
        data_dir = inputs.get("data_dir", "data/nuclear_library")
        group_count = inputs.get("group_count", inputs.get("mode", 1))
        if group_count == "one_group" or group_count == 1:
            group_count = 1
        else:
            group_count = 2

        provider = NuclearDataLibrary(data_dir=data_dir, group_count=group_count)

        # Optionally download nuclides
        if inputs.get("download", False):
            source = inputs.get("source", "tendl")
            from ..core.xs import download_common_nuclides
            download_common_nuclides(data_dir=data_dir, source=source)

        return {"xs_provider": provider, "group_count": group_count}

    def _skill_diffusion_solver(self, inputs: dict) -> dict:
        mesh = inputs["mesh"]
        xs_provider = inputs["xs_provider"]
        mode = inputs.get("mode", "one_group")
        eigen_solver = inputs.get("eigen_solver", "power_iteration")
        tolerance = inputs.get("tolerance", 1e-8)
        max_iter = inputs.get("max_iter", 500)
        initial_flux = inputs.get("initial_flux")
        initial_keff = inputs.get("initial_keff", 1.0)

        if mode in ("one_group", "1g", 1):
            solver = OneGroupDiffusionSolver(
                mesh=mesh,
                xs_provider=xs_provider,
                tolerance=tolerance,
                max_iter=max_iter,
                initial_flux=initial_flux,
                initial_keff=initial_keff,
            )
        elif mode in ("two_group", "2g", 2):
            solver = TwoGroupDiffusionSolver(
                mesh=mesh,
                xs_provider=xs_provider,
                tolerance=tolerance,
                max_iter=max_iter,
                initial_flux=initial_flux,
                initial_keff=initial_keff,
            )
        else:
            raise ValueError(f"Unknown solver mode: {mode}")

        result = solver.solve()
        return {
            "result": result,
            "keff": result.keff,
            "flux": result.flux,
            "power": result.power,
            "iterations": result.iterations,
            "converged": result.converged,
        }

    def _skill_postprocess(self, inputs: dict) -> dict:
        result = inputs.get("result")
        mesh = inputs.get("mesh")
        flux = inputs.get("flux", result.flux if result else None)
        power = inputs.get("power", result.power if result else None)
        normalize = inputs.get("normalize_power", 1.0)

        if power is not None and normalize != 1.0:
            power = power * normalize / power.sum()

        processor = PowerPostprocessor(mesh)
        peaking = processor.compute_peaking_factor(power)

        power_grid = processor.power_to_grid(power)

        return {
            "power": power,
            "power_grid": power_grid,
            "peaking_factor": peaking,
        }

    def _skill_report_writer(self, inputs: dict) -> dict:
        result = inputs["result"]
        mesh = inputs["mesh"]
        case_name = inputs.get("case_name", "default")
        output_dir = inputs.get("output_dir", "outputs")

        ReportWriter.write(result, mesh, output_dir, case_name)
        return {"output_dir": output_dir, "case_name": case_name}

    def _skill_renderer(self, inputs: dict) -> dict:
        """Execute a visualization skill.

        Supports multiple ways to specify the plot:
          1. plot_type + data directly
          2. recipe name (pre-built recipe from viz_plots)
          3. auto-detection from available data fields
        """
        backend = inputs.get("backend", "matplotlib")
        output_dir = inputs.get("output_dir", "outputs")
        renderer = Renderer(backend=backend, output_dir=output_dir)

        # Check for pre-built recipe
        recipe = inputs.get("recipe")
        if recipe:
            return self._run_recipe(renderer, recipe, inputs)

        # Manual plot specification
        plot_type = inputs.get("plot_type", "heatmap_2d")
        data = inputs.get("data")

        # Resolve data references (could be step_id.field path)
        if isinstance(data, str):
            data = self._resolve_template(data)

        # Convert list to numpy array
        if isinstance(data, list):
            data = np.array(data)

        # Handle mesh_shape for reshaping
        mesh_shape = inputs.get("mesh_shape")
        if mesh_shape and data is not None and data.ndim == 1:
            data = data.reshape(mesh_shape)

        # Extract mesh info if available
        mesh = inputs.get("mesh")
        if mesh is not None and mesh_shape is None:
            if hasattr(mesh, 'nx') and hasattr(mesh, 'ny'):
                mesh_shape = (mesh.ny, mesh.nx)
                if data is not None and data.ndim == 1:
                    data = data.reshape(mesh_shape)

        spec = PlotSpec(
            plot_type=plot_type,
            title=inputs.get("title", ""),
            data=data,
            xlabel=inputs.get("xlabel", ""),
            ylabel=inputs.get("ylabel", ""),
            zlabel=inputs.get("zlabel", ""),
            colormap=inputs.get("colormap", "viridis"),
            output_path=inputs.get("output_path", f"plot_{plot_type}.png"),
            interactive=inputs.get("interactive", False),
            width=inputs.get("width", 800),
            height=inputs.get("height", 600),
        )

        # Auxiliary data
        for key in ("residual_history", "burnup", "keff", "material_map",
                     "x_values", "y_values", "nuclides"):
            val = inputs.get(key)
            if val is not None:
                if isinstance(val, list):
                    val = np.array(val)
                if key == "residual_history":
                    spec.data_labels["residual"] = val
                elif key == "x_values":
                    spec.x = val
                elif key == "y_values":
                    spec.y = val
                else:
                    spec.data_labels[key] = val

        result = renderer.render(spec)

        return {
            "result": result,
            "output_path": result.output_paths[0] if result.output_paths else "",
            "output_paths": result.output_paths,
            "success": result.success,
            "error": result.error,
            "metadata": result.metadata,
        }

    def _run_recipe(self, renderer: Renderer, recipe: str, inputs: dict) -> dict:
        """Execute a named visualization recipe."""
        import numpy as np

        # Extract common data fields
        flux = inputs.get("flux")
        if isinstance(flux, str):
            flux = self._resolve_template(flux)
        if isinstance(flux, list):
            flux = np.array(flux)
        # Extract from Result object
        if flux is not None and hasattr(flux, 'flux'):
            flux = flux.flux

        power = inputs.get("power")
        if isinstance(power, str):
            power = self._resolve_template(power)
        if isinstance(power, list):
            power = np.array(power)
        if power is not None and hasattr(power, 'power'):
            power = power.power
        if power is not None and hasattr(power, 'flux') and not hasattr(power, 'power'):
            # It might be a Result passed as power; use result.power
            pass  # handled above via hasattr check

        mesh = inputs.get("mesh")
        mesh_shape = inputs.get("mesh_shape")
        if mesh_shape is None and mesh is not None:
            if hasattr(mesh, 'nx') and hasattr(mesh, 'ny'):
                mesh_shape = (mesh.ny, mesh.nx)

        residual = inputs.get("residual_history", inputs.get("residual"))
        if isinstance(residual, list):
            residual = np.array(residual)
        # Extract residual_history from Result object
        if residual is not None and hasattr(residual, 'residual_history'):
            residual = np.array(residual.residual_history)

        material_map = inputs.get("material_map")
        if isinstance(material_map, str):
            material_map = self._resolve_template(material_map)
        if isinstance(material_map, list):
            material_map = np.array(material_map, dtype=str)

        burnup = inputs.get("burnup")
        keff_vals = inputs.get("keff")

        nuclides = inputs.get("nuclides", {})

        if recipe == "power_heatmap":
            spec = viz_plots.power_heatmap(
                power, mesh_shape=mesh_shape,
                output=inputs.get("output_path", "power_heatmap.png"),
                title=inputs.get("title", "Power Distribution"),
            )
        elif recipe == "flux_heatmap":
            spec = viz_plots.flux_heatmap(
                flux, mesh_shape=mesh_shape,
                output=inputs.get("output_path", "flux_heatmap.png"),
                title=inputs.get("title", "Neutron Flux"),
            )
        elif recipe == "power_3d":
            spec = viz_plots.power_3d_surface(
                power, mesh_shape=mesh_shape,
                output=inputs.get("output_path", "power_3d.html"),
                title=inputs.get("title", "3D Power Surface"),
            )
        elif recipe == "convergence":
            spec = viz_plots.convergence_plot(
                residual,
                output=inputs.get("output_path", "convergence.png"),
                title=inputs.get("title", "Convergence History"),
            )
        elif recipe == "material_layout":
            spec = viz_plots.material_layout(
                material_map,
                output=inputs.get("output_path", "material_map.png"),
                title=inputs.get("title", "Material Layout"),
            )
        elif recipe == "keff_vs_burnup":
            spec = viz_plots.keff_vs_burnup(
                burnup if isinstance(burnup, list) else list(burnup or []),
                keff_vals if isinstance(keff_vals, list) else list(keff_vals or []),
                output=inputs.get("output_path", "keff_vs_burnup.png"),
                title=inputs.get("title", "k_eff vs Burnup"),
            )
        elif recipe == "nuclide_bar":
            spec = viz_plots.nuclide_bar(
                nuclides,
                output=inputs.get("output_path", "nuclide_composition.png"),
                title=inputs.get("title", "Nuclide Composition"),
            )
        elif recipe == "dashboard":
            spec = viz_plots.result_dashboard(
                flux, power, residual if residual is not None else np.array([]),
                material_map=material_map,
                mesh_shape=mesh_shape,
                output=inputs.get("output_path", "dashboard.png"),
                title=inputs.get("title", "Core Solver Dashboard"),
            )
        elif recipe == "cross_section_map":
            xs_field = inputs.get("xs_field", inputs.get("data"))
            if isinstance(xs_field, list):
                xs_field = np.array(xs_field)
            field_name = inputs.get("field_name", "Sigma")
            spec = viz_plots.cross_section_map(
                xs_field, field_name=field_name,
                mesh_shape=mesh_shape,
                output=inputs.get("output_path", "xs_map.png"),
            )
        else:
            return {"success": False, "error": f"Unknown recipe: {recipe}",
                    "available": ["power_heatmap", "flux_heatmap", "power_3d",
                                  "convergence", "material_layout", "keff_vs_burnup",
                                  "nuclide_bar", "dashboard", "cross_section_map"]}

        result = renderer.render(spec)
        return {
            "result": result,
            "output_path": result.output_paths[0] if result.output_paths else "",
            "output_paths": result.output_paths,
            "success": result.success,
            "error": result.error,
        }

    def _skill_benchmark_compare(self, inputs: dict) -> dict:
        """Compare solver result against benchmark reference values."""
        from ..core.benchmarks import compare_keff

        result = inputs.get("result")
        if result is None:
            return {"error": "No result provided"}

        k_comp = result.keff if hasattr(result, 'keff') else float(result)
        k_ref = float(inputs.get("ref_keff", 1.0))
        tolerance_pcm = float(inputs.get("tolerance_pcm", 500.0))
        benchmark_name = inputs.get("benchmark", "unknown")

        comparison = compare_keff(
            k_comp=k_comp, k_ref=k_ref,
            tolerance_pcm=tolerance_pcm,
            benchmark_name=benchmark_name,
            solver=result.solver if hasattr(result, 'solver') else "unknown",
        )

        print(f"\n{'='*60}")
        print(comparison.summary())
        print(f"{'='*60}\n")

        return {
            "comparison": comparison,
            "keff_computed": k_comp,
            "keff_reference": k_ref,
            "error_pcm": comparison.keff_error_pcm,
            "passed": comparison.passed,
            "summary": comparison.summary(),
        }
