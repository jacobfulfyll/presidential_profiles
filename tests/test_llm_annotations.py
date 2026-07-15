"""Storage layer: manifest round-trip, corpus fingerprint, and the loaders'
duplicate-key guards (regression #5, key integrity)."""

from __future__ import annotations

import pandas as pd
import pytest

from presidential_profiles import llm_annotations as ann


# ---------------------------------------------------------------------------
# manifests
# ---------------------------------------------------------------------------


def test_manifest_round_trips_through_disk():
    m = ann.Manifest(
        run_id="r1", model="claude-sonnet-5", prompt_version="v1",
        prompt_hash="sha256:abc", date="2026-07-14",
        n_requests=3, input_tokens=10, output_tokens=5, cost_usd=0.01,
        fields=["label"], annotation_files=["x.parquet"], notes="hi",
    )
    ann.write_manifest(m)
    assert ann.read_manifest("r1") == m


def test_manifest_preserves_null_cost_distinct_from_zero():
    """cost_usd=None (unknown) must survive the round-trip as None, never
    collapse to 0.0 — the whole point of the nullable field."""
    ann.write_manifest(ann.Manifest(
        run_id="r2", model="m", prompt_version="v", prompt_hash="h",
        date="2026-07-14", cost_usd=None,
    ))
    back = ann.read_manifest("r2")
    assert back.cost_usd is None


def test_read_manifest_tolerates_unknown_future_field(redirect_annotation_dirs):
    """Forward schema drift: a manifest written by a LATER version carries a
    field this dataclass doesn't know. read_manifest must ignore it and still
    return a valid Manifest, not raise TypeError — the layer's whole point is to
    stay readable later."""
    import json

    raw = {
        "run_id": "future", "model": "m", "prompt_version": "v", "prompt_hash": "h",
        "date": "2026-07-14", "n_requests": 7, "cost_usd": 0.5,
        "a_field_added_next_year": {"nested": True},  # unknown to this Manifest
    }
    ann.MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    ann.manifest_path("future").write_text(json.dumps(raw))

    m = ann.read_manifest("future")  # must not raise
    assert m.run_id == "future"
    assert m.n_requests == 7
    assert m.cost_usd == 0.5
    assert not hasattr(m, "a_field_added_next_year")  # dropped, not attached


def test_read_manifest_round_trip_still_returns_every_known_field(redirect_annotation_dirs):
    """The tolerance must not silently drop KNOWN fields: a full round-trip is
    still lossless."""
    m = ann.Manifest(
        run_id="full", model="claude-sonnet-5", prompt_version="v9", prompt_hash="sha256:z",
        date="2026-07-14", batch_id="b1", n_requests=4,
        input_tokens=11, output_tokens=6, cache_creation_input_tokens=2,
        cache_read_input_tokens=3, cost_usd=0.02,
        corpus_fingerprint={"n_speeches": 1}, fields=["label"],
        annotation_files=["y.parquet"], notes="round-trip",
    )
    ann.write_manifest(m)
    assert ann.read_manifest("full") == m


# ---------------------------------------------------------------------------
# corpus fingerprint
# ---------------------------------------------------------------------------


def _speeches(names):
    return pd.DataFrame({"doc_name": names})


def test_fingerprint_is_order_independent():
    """Sorted hash: the same doc_names in a different row order hash the same."""
    paras = pd.DataFrame({"doc_name": ["a", "a", "b"]})
    fp1 = ann.corpus_fingerprint(_speeches(["a", "b", "c"]), paras)
    fp2 = ann.corpus_fingerprint(_speeches(["c", "a", "b"]), paras)
    assert fp1["doc_name_sha256"] == fp2["doc_name_sha256"]


def test_fingerprint_moves_when_corpus_content_changes():
    paras = pd.DataFrame({"doc_name": ["a"]})
    fp1 = ann.corpus_fingerprint(_speeches(["a", "b"]), paras)
    fp2 = ann.corpus_fingerprint(_speeches(["a", "b", "c"]), paras)
    assert fp1["doc_name_sha256"] != fp2["doc_name_sha256"]


def test_fingerprint_reports_row_counts():
    fp = ann.corpus_fingerprint(_speeches(["a", "b"]), pd.DataFrame({"doc_name": ["a", "a", "b"]}))
    assert fp["n_speeches"] == 2
    assert fp["n_paragraphs"] == 3


def test_fingerprint_of_real_corpus_matches_recorded_provenance():
    """The migration manifest recorded a fingerprint of the shipped corpus.
    Recomputing it against the real parquet must reproduce it exactly — if this
    drifts, every annotation keyed against it is stale, and this test says so."""
    fp = ann.corpus_fingerprint()
    assert fp == {
        "n_speeches": 1057,
        "n_paragraphs": 36229,
        "doc_name_sha256": "23eff144ab94f3fa5cedefcacf7f21bc5c068518fe3cb2946c2a2a179fdb9bcc",
    }


# ---------------------------------------------------------------------------
# loaders — duplicate-key guards (regression #5)
# ---------------------------------------------------------------------------


def _write_raw(path, df):
    """Write a parquet directly, bypassing write_annotations' own key check, so
    the loader is tested in isolation against a table that is already bad."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_load_speech_annotations_raises_on_duplicate_doc_name(redirect_annotation_dirs):
    path = ann.annotation_path("dup_speech")
    _write_raw(path, pd.DataFrame({
        "doc_name": ["a", "a", "b"],
        "label": ["x", "y", "z"],
        "run_id": ["r", "r", "r"],
    }))
    with pytest.raises(ValueError, match="duplicate key"):
        ann.load_speech_annotations("dup_speech")


def test_load_speech_annotations_accepts_unique_doc_name(redirect_annotation_dirs):
    path = ann.annotation_path("ok_speech")
    _write_raw(path, pd.DataFrame({
        "doc_name": ["a", "b"], "label": ["x", "y"], "run_id": ["r", "r"],
    }))
    assert len(ann.load_speech_annotations("ok_speech")) == 2


def test_load_paragraph_annotations_raises_on_duplicate_grain(redirect_annotation_dirs):
    path = ann.annotation_path("dup_para")
    _write_raw(path, pd.DataFrame({
        "doc_name": ["a", "a", "a"],
        "para_idx": [0, 0, 1],  # (a,0) duplicated
        "label": ["x", "y", "z"],
        "run_id": ["r", "r", "r"],
    }))
    with pytest.raises(ValueError, match="duplicate key"):
        ann.load_paragraph_annotations("dup_para")


def test_paragraph_grain_allows_repeated_doc_name_but_speech_grain_does_not(redirect_annotation_dirs):
    """The two loaders guard INDEPENDENTLY at their own grain: a table with a
    repeated doc_name but unique (doc_name, para_idx) loads fine as paragraphs
    yet is rejected as speech-level."""
    path = ann.annotation_path("multi")
    _write_raw(path, pd.DataFrame({
        "doc_name": ["a", "a", "b"],
        "para_idx": [0, 1, 0],
        "label": ["x", "y", "z"],
        "run_id": ["r", "r", "r"],
    }))
    assert len(ann.load_paragraph_annotations("multi")) == 3
    with pytest.raises(ValueError, match="duplicate key"):
        ann.load_speech_annotations("multi")


def test_loader_reports_missing_key_columns(redirect_annotation_dirs):
    path = ann.annotation_path("nokey")
    _write_raw(path, pd.DataFrame({"label": ["x"], "run_id": ["r"]}))
    with pytest.raises(ValueError, match="missing columns"):
        ann.load_speech_annotations("nokey")


def test_load_invocation_tone_raises_on_duplicate_mention_key(redirect_annotation_dirs):
    """Mention grain is (doc_name, char_start) — its own third guard, distinct
    from the speech and paragraph loaders."""
    ann.INVOCATION_TONE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "doc_name": ["a", "a"],
        "char_start": [5, 5],  # duplicate mention key
        "label": ["C", "N"],
        "run_id": ["r", "r"],
    }).to_parquet(ann.INVOCATION_TONE_PATH, index=False)
    with pytest.raises(ValueError, match="char_start"):
        ann.load_invocation_tone()


def test_load_invocation_tone_accepts_distinct_mentions(redirect_annotation_dirs):
    ann.INVOCATION_TONE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "doc_name": ["a", "a"],
        "char_start": [5, 99],
        "label": ["C", "N"],
        "run_id": ["r", "r"],
    }).to_parquet(ann.INVOCATION_TONE_PATH, index=False)
    assert len(ann.load_invocation_tone()) == 2


# ---------------------------------------------------------------------------
# write_annotations enforces the key before disk
# ---------------------------------------------------------------------------


def test_write_annotations_refuses_bad_key_before_writing(redirect_annotation_dirs):
    df = pd.DataFrame({"doc_name": ["a", "a"], "label": ["x", "y"], "run_id": ["r", "r"]})
    with pytest.raises(ValueError, match="duplicate key"):
        ann.write_annotations("bad", df, unit="speech")
    assert not ann.annotation_path("bad").exists()


def test_write_annotations_round_trips_and_sorts(redirect_annotation_dirs):
    df = pd.DataFrame({
        "doc_name": ["b", "a"], "para_idx": [0, 0],
        "label": ["y", "x"], "run_id": ["r", "r"],
    })
    ann.write_annotations("good", df, unit="paragraph")
    back = ann.load_paragraph_annotations("good")
    assert list(back["doc_name"]) == ["a", "b"]  # sorted on the key
