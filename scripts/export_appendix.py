"""Export the project code as the report's code appendix (Word and Markdown).

Usage  : python scripts/export_appendix.py
Inputs : src/*.py, scripts/*.py, tests/*.py, notebooks/*.ipynb (code cells only)
Outputs: eval/appendix/code_appendix.docx  one heading per file with its owner,
                                           code in Consolas 8 pt, A4
         eval/appendix/code_appendix.md    the same as Markdown (fallback)
Files appear in pipeline order (data -> methods -> measurement -> evaluation,
then the tests); the owner is read from each file's docstring ("Owner:" or
"Author:"), or from FALLBACK_OWNERS when a docstring names nobody. The
script warns about functions still marked @placeholder, which should all be
gone before submission. Needs python-docx (pip install python-docx).
Author : Seungmin Lee
"""
from __future__ import annotations

import ast
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "eval" / "appendix"
ORDER = [  # pipeline order; files not listed here (e.g. tests) are added at the end
    "src/config.py", "src/utils.py", "src/data.py", "scripts/make_split.py",
    "src/method_a.py", "scripts/tune_a.py", "scripts/figure_a.py",
    "src/method_b.py", "scripts/prepare_b1_dataset.py", "scripts/prepare_b2_dataset.py",
    "scripts/train_b.py", "notebooks/method_b_hog_glcm_svm.ipynb",
    "src/method_c.py", "notebooks/train_unet_colab.ipynb", "src/method_d.py",
    "scripts/run_method.py", "src/measure.py", "scripts/run_resolution.py",
    "src/evaluate.py", "src/viz.py", "scripts/run_evaluation.py",
    "scripts/human_study.py", "scripts/failure_analysis.py",
    "scripts/check_interfaces.py", "scripts/export_appendix.py",
]
FALLBACK_OWNERS = {  # file-name prefix -> owner, for files whose docstring names nobody
    "data": "Zitong Luo", "make_split": "Zitong Luo",
    "method_a": "Wenjie Hu", "tune_a": "Wenjie Hu", "figure_a": "Wenjie Hu",
    "method_b": "Bowen Xu", "prepare_b": "Bowen Xu", "train_b": "Bowen Xu",
    "method_c": "Lingyu Wu", "method_d": "Lingyu Wu", "train_unet": "Lingyu Wu",
    "test_method_c": "Lingyu Wu", "run_method_cd": "Lingyu Wu", "run_method_d": "Lingyu Wu",
    "measure": "Junsen Liang", "run_resolution": "Junsen Liang",
}
OWNER_LINE = re.compile(r"\b(?:Owner|Author)\s*:\s*\**\s*([A-Z][\w-]+(?: [A-Z][\w-]+){0,2})")


def project_files() -> list[Path]:
    """Every code file of the project, in ORDER first and then alphabetically."""
    found = [p for folder in ("src", "scripts", "tests") for p in sorted((ROOT / folder).glob("*.py"))]
    found += sorted((ROOT / "notebooks").glob("*.ipynb"))
    found = [p for p in found if p.read_text(encoding="utf-8").strip()]  # skip empty files
    rank = {name: i for i, name in enumerate(ORDER)}
    return sorted(found, key=lambda p: (rank.get(p.relative_to(ROOT).as_posix(), len(ORDER)), p.as_posix()))


def code_of(path: Path) -> str:
    """Source text; for a notebook, its code cells separated by '# --- cell k ---'."""
    if path.suffix != ".ipynb":
        return path.read_text(encoding="utf-8").replace("\t", "    ").rstrip() + "\n"
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    parts = [f"# --- cell {k} ---\n" + "".join(cell["source"]).rstrip()
             for k, cell in enumerate(cells, 1) if cell["cell_type"] == "code" and "".join(cell["source"]).strip()]
    return "\n\n".join(parts).replace("\t", "    ") + "\n"


def describe(path: Path) -> tuple[str, str]:
    """(owner, purpose) from the module docstring, or the notebook's first text cell."""
    if path.suffix == ".ipynb":
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        text = "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown")
        purpose = text.lstrip("# ").splitlines()[0] if text else path.stem
    else:
        text = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
        purpose = text.splitlines()[0] if text else path.stem
    match = OWNER_LINE.search(text)
    if match:
        owner = match.group(1)
    else:
        prefixes = sorted((k for k in FALLBACK_OWNERS if path.stem.startswith(k)), key=len, reverse=True)
        owner = FALLBACK_OWNERS[prefixes[0]] if prefixes else "Team"
    return owner, purpose.strip()


def placeholders_left(files: list[Path]) -> list[str]:
    """Functions still decorated with @placeholder (they should be replaced before submission)."""
    left = []
    for path in files:
        if path.suffix != ".py" or path.name == "utils.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and any(
                    getattr(d, "id", None) == "placeholder" for d in node.decorator_list):
                left.append(f"{path.relative_to(ROOT).as_posix()}: {node.name}")
    return left


def git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else "not a git checkout"
    except (OSError, subprocess.SubprocessError):
        return "git not available"


def write_markdown(entries: list[dict], intro: str, path: Path) -> None:
    lines = ["# Appendix: Source code", "", intro, "", "| File | Owner | Lines | Purpose |", "|---|---|---|---|"]
    lines += [f"| `{e['name']}` | {e['owner']} | {e['lines']} | {e['purpose']} |" for e in entries]
    for e in entries:
        language = "python"
        lines += ["", f"## {e['name']} ({e['owner']})", "", f"```{language}", e["code"].rstrip(), "```"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_docx(entries: list[dict], intro: str, path: Path) -> bool:
    """Word version: A4, Calibri text, Consolas 8 pt code, one heading per file."""
    try:
        from docx import Document
        from docx.shared import Cm, Pt
    except ImportError:
        print("python-docx is not installed (pip install python-docx): wrote the Markdown version only.")
        return False
    document = Document()
    for section in document.sections:
        section.page_width, section.page_height = Cm(21.0), Cm(29.7)
        section.left_margin = section.right_margin = Cm(1.8)
        section.top_margin = section.bottom_margin = Cm(1.8)
    document.styles["Normal"].font.name = "Calibri"
    document.styles["Normal"].font.size = Pt(10)
    document.add_heading("Appendix: Source code", level=1)
    document.add_paragraph(intro)
    table = document.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    table.autofit = False
    widths = (Cm(5.6), Cm(2.8), Cm(1.4), Cm(7.6))
    for cell, text in zip(table.rows[0].cells, ("File", "Owner", "Lines", "Purpose")):
        cell.text = text
        cell.paragraphs[0].runs[0].bold = True
    for e in entries:
        cells = table.add_row().cells
        for cell, text in zip(cells, (e["name"], e["owner"], str(e["lines"]), e["purpose"])):
            cell.text = text
            for run in cell.paragraphs[0].runs:
                run.font.size = Pt(8.5)
    for column, width in zip(table.columns, widths):   # Word reads the cell widths,
        column.width = width                            # LibreOffice the column grid
        for cell in column.cells:
            cell.width = width
    for e in entries:
        document.add_heading(f"{e['name']} ({e['owner']})", level=2)
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.line_spacing = 1.0
        lines = e["code"].rstrip("\n").split("\n")
        for i, line in enumerate(lines):
            run = paragraph.add_run(line)
            run.font.name = "Consolas"
            run.font.size = Pt(8)
            if i < len(lines) - 1:
                run.add_break()
    document.save(str(path))
    return True


def main() -> None:
    files = project_files()
    entries = []
    for path in files:
        owner, purpose = describe(path)
        code = code_of(path)
        entries.append({"name": path.relative_to(ROOT).as_posix(), "owner": owner, "purpose": purpose,
                        "code": code, "lines": code.count("\n")})
    intro = (f"Repository crack-width-measurement, commit {git_commit()}, exported {dt.date.today().isoformat()}. "
             f"{len(entries)} files, {sum(e['lines'] for e in entries)} lines, listed in pipeline order. "
             "Each file starts with a docstring giving its purpose, inputs, outputs and owner. "
             "Notebook cells are reproduced in order; shell commands (lines starting with !) run on Colab.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_markdown(entries, intro, OUT_DIR / "code_appendix.md")
    wrote_docx = write_docx(entries, intro, OUT_DIR / "code_appendix.docx")
    for e in entries:
        print(f"{e['lines']:5d}  {e['owner']:<14} {e['name']}")
    left = placeholders_left(files)
    if left:
        print(f"\nWarning: {len(left)} function(s) still marked @placeholder:\n  " + "\n  ".join(left))
    names = "code_appendix.docx and code_appendix.md" if wrote_docx else "code_appendix.md"
    print(f"\nSaved {names} in {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())