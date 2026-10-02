"""Execute the case-study notebook in the local uv environment and save outputs."""

from pathlib import Path
import os

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "case_study2.ipynb"
KERNELS = ROOT / ".venv" / "share" / "jupyter" / "kernels"

os.environ["JUPYTER_PATH"] = str(KERNELS) + os.pathsep + os.environ.get("JUPYTER_PATH", "")

nb = nbformat.read(NOTEBOOK, as_version=4)
client = NotebookClient(
    nb,
    timeout=3600,
    kernel_name="cs2-uv",
    resources={"metadata": {"path": str(ROOT)}},
    allow_errors=False,
)
client.execute()
nbformat.write(nb, NOTEBOOK)
print(f"Executed and saved {NOTEBOOK}")
