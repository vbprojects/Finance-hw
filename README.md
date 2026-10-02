# Masters Finance Class

`case_study2.ipynb` is the complete, self-contained Case Study 2 submission. It includes code, saved outputs, figures, and dated interpretation for Sections 1–6. Submit that single notebook for the assignment. `case_study1.ipynb` is the earlier completed assignment.

The notebook downloads FRED rates and Yahoo Finance FX closes on a clean run. It keeps raw downloads in the relative `data_cache/` folder and refreshes stale series. Internet access is needed the first time; saved notebook outputs can be read offline. Results and written discussion were last checked with data available on 2026-09-30. Refreshed sources can change the numbers.

To recreate the local Windows environment and execute the notebook:

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name cs2-uv --display-name "Case Study 2 (uv)"
.\.venv\Scripts\python.exe run_notebook.py
```

`assemble_notebook.py` and the section scripts are build helpers used to develop the notebook. The submitted `.ipynb` does not import them.
