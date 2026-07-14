# Presidential Profiles

## Environment
- The `.venv` is x86_64 under Rosetta. Always invoke Python as `arch -x86_64 .venv/bin/python`
  (or `arch -x86_64 uv run ...`) — a plain `uv run`/`python` invocation on Apple Silicon can
  resolve to the wrong architecture's interpreter.
- The venv is uv-managed and has no `pip`; install packages with `uv pip install ...` (with
  `VIRTUAL_ENV` pointed at `.venv`) rather than `python -m pip`.

## Data conventions
- Paragraph-level tables (`data/paragraphs.parquet`, `data/paragraph_issues.parquet`) share a
  real `(doc_name, para_idx)` key. Always join on it (`validate="one_to_one"`) — never assume
  row order lines the two tables up, even though it happens to hold today. This replaced a
  positional-alignment bug that could silently mislabel every paragraph on reorder.

## Testing
- `tests/` (pytest) covers the paragraph/issue-label keyed-merge logic. Run with
  `uv sync --extra dev && uv run pytest`, or directly:
  `arch -x86_64 .venv/bin/python -m pytest`.
- Favor small synthetic DataFrames over the real 36k-row corpus or the real CorEx topic model
  (`issues.build_issues()` fits a full model — too slow/heavy for unit tests). Mirror the
  specific logic under test on a tiny frame instead of invoking the real pipeline function.
