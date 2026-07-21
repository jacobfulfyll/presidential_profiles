---
name: cli-path-arg-containment
description: Recurring finding class in this repo — CLI path/id args flow into a write with no containment check; how to grade it
metadata:
  type: feedback
---

This repo's analysis modules repeatedly ship a CLI flag that flows into a filesystem write with no
containment check. Seen at least twice: `sanitize-run-id-cli-arg` (an unsanitized `run_id` joined into
a path, escapable with `../..`) and `attention.py`'s `--out` (`type=Path`, passed straight to
`to_parquet` after `mkdir(parents=True, exist_ok=True)`). Expect it again; check every new `main()`.

**Why it is NOT a vulnerability here, and must not be graded as one:** the operator running the CLI
already has full filesystem write access — they could write the file directly. There is no untrusted
input source, so path traversal has no privilege boundary to cross. Grading it CRITICAL/HIGH is
exactly the "manufacture a finding to justify the stage" failure.

**Why it is still worth reporting (MEDIUM):** the real harm is **integrity of paid artifacts**.
`data/llm_annotations/` is documented read-only — provenance-stamped output of a $38.56 paid run that
cannot be regenerated for free. A fat-fingered or copy-pasted `--out data/llm_annotations/....parquet`
silently clobbers it, and `mkdir(parents=True)` will happily build arbitrary directory trees on the way.
Frame it as a fat-finger/integrity guard against a documented repo invariant, and say plainly that it
is not a security vulnerability. See [[project-attack-surface]].

**How to apply — where the guard goes:** validate in `main()` (the CLI boundary), NOT in the library
write function. Library writers are called with `tmp_path` by tests (e.g.
`test_write_lifecycles_is_byte_identical_on_rerun` passes `tmp_path / "nested" / "one.parquet"`), so a
containment check inside the writer breaks the suite. A `main()`-only guard also leaves the artifact
sha untouched — important because these tasks carry a byte-reproducibility oracle.
Suggested shape: `if not path.resolve().is_relative_to(ATTENTION_DIR.resolve()): raise ValueError(...)`.
