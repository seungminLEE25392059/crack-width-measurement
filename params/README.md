# Validation-selected parameters

One JSON file per method (`A.json`, `B.json`, `C.json`, `D.json`), written with
`utils.save_params(...)` after tuning on the **validation** split. `run_method.py`
loads them on top of each module's `DEFAULT_PARAMS`, so the final test run uses
exactly the committed values.

`severity.json` holds the severity thresholds, computed once from the
**training** reference masks by `run_evaluation.py`. Recompute it with
`--recompute-severity` after `src/measure.py` changes.

Commit these files: they are small and they document every setting for the report.
