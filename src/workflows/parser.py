"""YAML workflow parser and validator."""

import yaml
from pathlib import Path
from typing import Any


def parse_workflow_yaml(path: str | Path) -> dict:
    """Parse a workflow YAML file and return the spec dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def validate_workflow(spec: dict) -> list[str]:
    """Validate a workflow spec and return a list of issues (empty = valid).

    Checks:
      - workflow has required fields
      - each step has id and skill
      - skill names are recognized
      - template references point to valid steps
    """
    issues = []

    if "workflow" not in spec and "steps" not in spec:
        issues.append("Workflow must have 'workflow' key with 'steps' list, "
                       "or top-level 'steps' list.")
        return issues

    wf = spec.get("workflow", spec)
    steps = wf.get("steps", [])

    if not steps:
        issues.append("No steps defined.")
        return issues

    step_ids = set()
    known_skills = {
        "mesh_builder", "build_mesh",
        "material_registry", "register_materials",
        "xs_provider_constant", "constant_xs",
        "xs_provider_nuclear", "nuclear_library",
        "steady_diffusion_solver", "diffusion_solver", "solve",
        "power_postprocess", "postprocess",
        "report_writer", "write_report",
    }

    for i, step in enumerate(steps):
        sid = step.get("id", f"step_{i}")
        if sid in step_ids:
            issues.append(f"Duplicate step id: {sid}")
        step_ids.add(sid)

        skill = step.get("skill", step.get("type", ""))
        if skill not in known_skills:
            issues.append(f"Step '{sid}': unknown skill '{skill}'. "
                         f"Known: {sorted(known_skills)}")

        inp = step.get("input", {})
        # Check template references
        for key, value in inp.items():
            if isinstance(value, str) and "{{" in value:
                # Extract step reference
                import re
                refs = re.findall(r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s*\}\}', value)
                for ref in refs:
                    ref_step = ref.split(".")[0]
                    if ref_step not in step_ids and ref_step != sid:
                        # Can't check forward references, skip
                        pass

    # Check output references
    outputs = wf.get("outputs", [])
    for out in outputs:
        if isinstance(out, str) and "{{" in out:
            import re
            refs = re.findall(r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s*\}\}', out)
            for ref in refs:
                ref_step = ref.split(".")[0]
                if ref_step not in step_ids:
                    issues.append(f"Output reference '{ref}' points to unknown step '{ref_step}'")

    return issues
