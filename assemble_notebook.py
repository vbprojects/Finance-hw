"""Assemble the verified section cells into the single assignment notebook."""

from pathlib import Path

import nbformat

import section1_repair
import sections_2_3
import sections_4_5


ROOT = Path(__file__).resolve().parent
PATH = ROOT / "case_study2.ipynb"
original = nbformat.read(PATH, as_version=4)

sections = (
    section1_repair.get_cells()
    + sections_2_3.get_cells()
    + sections_4_5.get_cells()
)
sections[0]["source"] += """

To reproduce the local environment (Python 3.12), run `uv venv .venv --python 3.12`
and `uv pip install --python .venv/Scripts/python.exe -r requirements.txt` from
this folder. The required packages are numpy, pandas, scipy, statsmodels,
arch, matplotlib, yfinance, nbformat, nbclient, ipykernel, and IPython.
The grader can run the notebook in any compatible Python environment; the
notebook obtains its own data when the cache is absent.
"""
cells = []
for section_cell in sections:
    kind = section_cell["cell_type"]
    source = section_cell["source"]
    if kind == "markdown":
        cells.append(nbformat.v4.new_markdown_cell(source))
    elif kind == "code":
        compile(source, f"notebook cell {len(cells)}", "exec")
        cells.append(nbformat.v4.new_code_cell(source))
    else:
        raise ValueError(f"Unexpected cell type: {kind}")

assembled = nbformat.v4.new_notebook(cells=cells, metadata=original.metadata)
nbformat.validate(assembled)
nbformat.write(assembled, PATH)
print(f"Assembled {PATH} with {len(cells)} cells")
