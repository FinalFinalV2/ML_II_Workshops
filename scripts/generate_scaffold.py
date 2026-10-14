#!/usr/bin/env python3
import json, re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

STUB_LINES = [
    "    # -------------------------------------------------------------------------\n",
    "    # YOUR CODE HERE\n",
    "    # -------------------------------------------------------------------------\n",
    "    raise NotImplementedError(\"Student implementation missing\")\n"
]

def strip_cell_solution(source_lines):
    full_text = "".join(source_lines)
    # Pattern 1: delimited by ### BEGIN SOLUTION ... ### END SOLUTION
    if "### BEGIN SOLUTION" in full_text and "### END SOLUTION" in full_text:
        new_lines = []
        in_solution = False
        for line in source_lines:
            if "### BEGIN SOLUTION" in line:
                in_solution = True
                new_lines.extend(STUB_LINES)
            elif "### END SOLUTION" in line:
                in_solution = False
            elif not in_solution:
                new_lines.append(line)
        return new_lines
    
    # Pattern 2: cell tagged as solution entirely
    hints = []
    for line in source_lines:
        if line.strip().startswith("#") and not line.strip().startswith("# SOLUTION"):
            hints.append(line)
        elif line.strip().startswith("def "):
            hints.append(line)
            hints.append("    \"\"\"Implement function.\"\"\"\n")
            hints.extend(STUB_LINES)
            return hints
    return hints + [s.lstrip() for s in STUB_LINES]

def process_notebook(sol_path: Path, out_path: Path):
    with open(sol_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    new_cells = []
    for cell in nb.get("cells", []):
        cell_type = cell.get("cell_type", "")
        tags = cell.get("metadata", {}).get("tags", [])
        src = "".join(cell.get("source", []))
        if cell_type == "code" and ("solution" in tags or "### BEGIN SOLUTION" in src):
            new_source = strip_cell_solution(cell.get("source", []))
            cell_copy = dict(cell)
            cell_copy["source"] = new_source
            cell_copy["execution_count"] = None
            cell_copy["outputs"] = []
            new_cells.append(cell_copy)
        elif cell_type == "markdown" and ("solution" in tags or "### BEGIN SOLUTION" in src):
            cell_copy = dict(cell)
            cell_copy["source"] = ["*Write your answer here in markdown...*\n"]
            new_cells.append(cell_copy)
        elif cell_type == "code":
            cell_copy = dict(cell)
            cell_copy["execution_count"] = None
            cell_copy["outputs"] = []
            new_cells.append(cell_copy)
        else:
            new_cells.append(cell)
    nb["cells"] = new_cells
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Scaffolded {out_path.name}")

if __name__ == "__main__":
    for sol in ROOT_DIR.glob("workshop_*/instructor/*_solution.ipynb"):
        target = sol.parent.parent / "student" / sol.name.replace("_solution.ipynb", ".ipynb")
        process_notebook(sol, target)
