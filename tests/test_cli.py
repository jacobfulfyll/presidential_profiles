"""CLI wiring: main() dispatches subcommands and enforces --run-id."""

from __future__ import annotations

import pytest

from presidential_profiles import annotate as A


def test_main_dry_run_dispatches_and_stays_offline(monkeypatch, redirect_annotation_dirs, anthropic_construction_bomb):
    monkeypatch.setattr(
        "sys.argv",
        ["pp-annotate", "dry-run", "--run-id", "cli", "--limit", "2", "--spec", "placeholder"],
    )
    A.main()
    assert (A.ann.RUNS_DIR / "cli" / "requests.jsonl").exists()


def test_main_requires_run_id_for_ingest(monkeypatch):
    monkeypatch.setattr("sys.argv", ["pp-annotate", "ingest"])
    with pytest.raises(SystemExit):
        A.main()
