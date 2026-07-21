"""Target #8 — run_id is sanitized before it becomes a path.

``run_id`` becomes a directory under ``runs/`` and a manifest filename. A
fat-fingered ``/``, ``\\`` or ``..`` could escape that tree, so main() rejects it
with a clear argparse error BEFORE dispatching to any command (i.e. before any
run dir is created).
"""

from __future__ import annotations

import sys

import pytest

from presidential_profiles import annotate as A


@pytest.mark.parametrize("bad", ["run/evil", "run\\evil", "..", "x/../y", "../escape"])
def test_run_id_with_path_metacharacters_is_rejected_before_touching_disk(
    bad, monkeypatch, capsys, redirect_annotation_dirs
):
    monkeypatch.setattr(sys, "argv", ["pp-annotate", "dry-run", "--run-id", bad])
    with pytest.raises(SystemExit) as exc:
        A.main()

    # argparse's parser.error exits code 2 and prints to stderr
    assert exc.value.code == 2
    assert "must not contain" in capsys.readouterr().err
    # the guard fired before dispatch -> no run directory was ever created
    assert not A.ann.RUNS_DIR.exists()


def test_clean_run_id_passes_the_sanitizer(monkeypatch, capsys, redirect_annotation_dirs):
    """A well-formed run_id must NOT trip the guard. Using `status` (which then
    fails on the missing batch) proves the sanitizer let a clean id through
    rather than the guard being an accidental catch-all."""
    monkeypatch.setattr(sys, "argv", ["pp-annotate", "status", "--run-id", "2026-07-20-clean"])
    with pytest.raises(SystemExit) as exc:
        A.main()
    err = capsys.readouterr().err
    # it got PAST sanitization into cmd_status, which then complains about the
    # absent batch — not about the run_id charset.
    assert "must not contain" not in err
    assert "no batch_id" in str(exc.value)
