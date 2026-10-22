import os
import sys
import json
import ast
import types
from pathlib import Path

def load_submission(workshop_dir: Path = None):
    if workshop_dir is None:
        workshop_dir = Path(__file__).resolve().parent.parent

    target = os.environ.get("AUTOGRADE_TARGET", "auto").lower()

    instructor_solution_file = workshop_dir / "instructor" / "workshop_02_solution.py"
    if (target in ("instructor", "solution")) and instructor_solution_file.exists():
        sys.path.insert(0, str(workshop_dir / "instructor"))
        import workshop_02_solution
        return workshop_02_solution

    student_nb_path = workshop_dir / "student" / "workshop_02.ipynb"
    if not student_nb_path.exists():
        raise FileNotFoundError(f"Student notebook not found at {student_nb_path}")

    with open(student_nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    mod = types.ModuleType("student_submission")
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            lines = [l for l in cell.get("source", []) if not l.strip().startswith(("%", "!"))]
            src = "".join(lines)
            if not src.strip():
                continue
            try:
                tree = ast.parse(src)
                for node in tree.body:
                    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign)):
                        exec(compile(ast.Module(body=[node], type_ignores=[]), filename=str(student_nb_path), mode="exec"), mod.__dict__)
            except Exception:
                pass
    return mod
