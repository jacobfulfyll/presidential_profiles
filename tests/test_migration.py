"""migrate_invocation_tone: provenance must be exact and honest (regression #4).

The source JSON (deleted from data/ once the parquet superseded it) is preserved
as a hermetic fixture; the migration is replayed against the REAL corpus and every
re-anchored offset is checked to re-slice out of the right speech's transcript.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from presidential_profiles import llm_annotations as ann
from presidential_profiles.corpus import load

FIXTURE_JSON = Path(__file__).parent / "fixtures" / "invocation_tone.json"


@pytest.fixture
def migrated(redirect_annotation_dirs, monkeypatch):
    """Run the migration once, offline, against the real corpus and the fixture
    JSON, writing into the redirected annotation dir."""
    json_path = redirect_annotation_dirs / "invocation_tone.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(FIXTURE_JSON.read_text())
    monkeypatch.setattr(ann, "INVOCATION_TONE_JSON", json_path)
    df = load()
    out = ann.migrate_invocation_tone(df=df, force=True)
    return out, df


def test_migration_yields_exact_row_counts(migrated):
    out, _ = migrated
    assert len(out) == 101
    assert (out["speaker"] == "Joe Biden").sum() == 69
    assert (out["speaker"] == "Donald Trump").sum() == 32


def test_migration_preserves_label_histograms_exactly(migrated):
    out, _ = migrated
    biden = out[out["speaker"] == "Joe Biden"]["label"].value_counts().to_dict()
    trump = out[out["speaker"] == "Donald Trump"]["label"].value_counts().to_dict()
    assert biden == {"C": 52, "N": 16, "R": 1}
    assert trump == {"C": 20, "N": 10, "R": 2}


def test_every_offset_reslices_to_the_target_surname_in_the_right_speech(migrated):
    """(char_start, char_end) is anchored to a real speech, not merely counted:
    slicing that speech's transcript must reproduce the stored mention, and the
    mention must be the target's surname."""
    out, df = migrated
    transcripts = dict(zip(df["doc_name"], df["transcript"]))
    for row in out.itertuples():
        surname = row.target.split()[-1]  # "Trump" / "Obama"
        assert row.mention == surname
        sliced = transcripts[row.doc_name][row.char_start:row.char_end]
        assert sliced == row.mention, (
            f"offset {row.char_start}:{row.char_end} in {row.doc_name} sliced "
            f"{sliced!r}, expected {row.mention!r}"
        )


def test_migration_manifest_cost_is_null_not_zero(migrated):
    """cost_usd MUST be None (unknown) — recording 0.0 for an interactive run
    whose spend was never captured would be a lie."""
    m = ann.read_manifest(ann.INVOCATION_TONE_RUN_ID)
    assert m.cost_usd is None
    assert m.n_requests == 0
    assert m.batch_id is None


def test_migration_manifest_preserves_original_method_verbatim(migrated):
    """The historical `method` claim is copied into notes unedited, and the
    prompt_hash is the hash of exactly that string."""
    import hashlib

    original = json.loads(FIXTURE_JSON.read_text())
    m = ann.read_manifest(ann.INVOCATION_TONE_RUN_ID)
    assert original["method"] in m.notes
    assert m.prompt_hash == "sha256:" + hashlib.sha256(original["method"].encode()).hexdigest()


def test_migration_is_idempotent_when_cached(migrated, monkeypatch):
    """A second call without force returns the cached parquet rather than
    re-deriving (and re-writing) it."""
    out, df = migrated
    again = ann.migrate_invocation_tone(df=df, force=False)
    assert len(again) == len(out)
    # The cached read comes back sorted by key while the fresh derivation is in
    # match order, so compare content, not row order.
    cols = ["doc_name", "char_start", "char_end", "label"]
    assert (set(map(tuple, again[cols].itertuples(index=False)))
            == set(map(tuple, out[cols].itertuples(index=False))))


def test_forced_rederive_with_deleted_source_signposts_git_recovery(redirect_annotation_dirs, monkeypatch):
    """This commit deletes the source JSON once the parquet supersedes it. A
    forced re-derivation (the audit path) must fail with a signpost to git
    recovery, not a bare FileNotFoundError deep in read_text()."""
    gone = ann.DATA_DIR / "llm_annotations" / "removed_after_migration.json"
    assert not gone.exists()
    monkeypatch.setattr(ann, "INVOCATION_TONE_JSON", gone)
    # INVOCATION_TONE_PATH is redirected to a fresh tmp dir (autouse), so force
    # cannot fall back to a cached parquet — it must reach the source read.
    with pytest.raises(FileNotFoundError, match="git history|git show") as exc:
        ann.migrate_invocation_tone(force=True)
    assert "recover" in str(exc.value)


def test_migration_bails_when_match_count_drifts(redirect_annotation_dirs, monkeypatch):
    """If the corpus no longer reproduces the recorded match count, the labels
    can't be trusted to line up — the migration must STOP, not hand-align."""
    json_path = redirect_annotation_dirs / "invocation_tone.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(FIXTURE_JSON.read_text())
    monkeypatch.setattr(ann, "INVOCATION_TONE_JSON", json_path)

    df = load()
    # Drop most of Biden's speeches so the Trump-mention scan can no longer find
    # the expected 69 matches. The migration must refuse rather than re-zip.
    biden = df[df["president"] == "Joe Biden"].head(1)
    shrunk = df[df["president"] != "Joe Biden"]
    import pandas as pd
    trimmed = pd.concat([shrunk, biden], ignore_index=True)
    with pytest.raises(AssertionError, match="do not hand-align|reconstruction found"):
        ann.migrate_invocation_tone(df=trimmed, force=True)
