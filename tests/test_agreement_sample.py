"""draw_sample (agreement.py) — the persisted era-stratified draw.

The committed JSON, not the seed, is the source of truth (a redraw must be a
deliberate, force-guarded act because it invalidates any Opus pass keyed to the
current membership). These tests drive draw_sample against an INJECTED synthetic
speeches frame and tmp_path only — never the real 1,057-speech corpus draw, and
never the committed data/llm_annotations/agreement_sample_v1.json.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from presidential_profiles import agreement as G


def _synthetic_speeches() -> pd.DataFrame:
    """Three era bins with deliberately chosen sizes:
      * bin 0 (year 1800):  4 docs  -> proportional 1, CLAMPED UP to min 3
      * bin 1 (year 1830): 14 docs  -> round-half-up int(0.25*14+0.5)=4 (floor=3)
      * bin 2 (year 1860): 40 docs  -> proportional 10
    So the draw is a real subset (not the whole frame) and both the min-3 clamp
    and the round-half-up rule are exercised on distinct bins."""
    rows = []
    rows += [{"doc_name": f"a{i:02d}", "year": 1800} for i in range(4)]
    rows += [{"doc_name": f"b{i:02d}", "year": 1830} for i in range(14)]
    rows += [{"doc_name": f"c{i:02d}", "year": 1860} for i in range(40)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_same_seed_and_corpus_yields_identical_membership_twice(tmp_path):
    speeches = _synthetic_speeches()
    a = G.draw_sample(path=tmp_path / "a.json", speeches=speeches,
                      draw_date="2026-07-21", force=True)
    b = G.draw_sample(path=tmp_path / "b.json", speeches=speeches,
                      draw_date="2026-07-21", force=True)
    assert a["doc_names"] == b["doc_names"]
    # non-vacuous: the draw is a genuine subset, not the whole frame
    assert a["n_sampled"] < a["n_corpus_speeches"]


def test_a_different_seed_can_change_membership(tmp_path):
    """Guards the seed is actually threaded into the RNG — a hardcoded/ignored
    seed would make this pass vacuously identical."""
    speeches = _synthetic_speeches()
    a = G.draw_sample(path=tmp_path / "a.json", speeches=speeches, seed=1, force=True)
    b = G.draw_sample(path=tmp_path / "b.json", speeches=speeches, seed=999, force=True)
    # the b-bin (14 docs, 4 sampled) and c-bin (40 docs, 10 sampled) have room to
    # differ across seeds; at least one selection must move
    assert a["doc_names"] != b["doc_names"]


# ---------------------------------------------------------------------------
# stratum sizing: min-3 clamp + proportional round-half-up
# ---------------------------------------------------------------------------


def test_tiny_bin_is_clamped_up_to_the_minimum_per_bin(tmp_path):
    out = G.draw_sample(path=tmp_path / "s.json", speeches=_synthetic_speeches(), force=True)
    bin0 = next(b for b in out["bins"] if b["era_bin"] == 0)
    assert bin0["n_in_bin"] == 4
    assert bin0["n_sampled"] == G.MIN_PER_BIN == 3  # 0.25*4=1 clamped up to 3


def test_proportional_bin_uses_round_half_up_not_floor(tmp_path):
    out = G.draw_sample(path=tmp_path / "s.json", speeches=_synthetic_speeches(), force=True)
    bin1 = next(b for b in out["bins"] if b["era_bin"] == 1)
    assert bin1["n_in_bin"] == 14
    # int(0.25*14 + 0.5) = int(4.0) = 4 ; a plain floor(0.25*14) would give 3
    assert bin1["n_sampled"] == 4


def test_large_bin_is_proportional(tmp_path):
    out = G.draw_sample(path=tmp_path / "s.json", speeches=_synthetic_speeches(), force=True)
    bin2 = next(b for b in out["bins"] if b["era_bin"] == 2)
    assert bin2["n_in_bin"] == 40
    assert bin2["n_sampled"] == 10


def test_bin_smaller_than_min_per_bin_samples_all_without_crashing(tmp_path):
    """The size formula is min(n_in, max(min_per_bin, round_half_up)). The OUTER
    min(n_in, ...) matters when a bin has FEWER docs than min_per_bin: without it
    the draw would ask rng.choice(2, size=3, replace=False) and raise. A 2-doc bin
    (below the min of 3) must therefore sample BOTH docs, not crash. (The real
    corpus has >=3 per bin, but draw_sample is a general, seed-driven function and
    a future/sub-corpus draw can hit this.)"""
    speeches = pd.DataFrame(
        [{"doc_name": f"a{i:02d}", "year": 1800} for i in range(2)]       # bin 0: 2 docs
        + [{"doc_name": f"b{i:02d}", "year": 1830} for i in range(5)]     # bin 1: 5 docs
    )
    out = G.draw_sample(path=tmp_path / "s.json", speeches=speeches, force=True)
    bin0 = next(b for b in out["bins"] if b["era_bin"] == 0)
    assert bin0["n_in_bin"] == 2
    assert bin0["n_sampled"] == 2  # clamped DOWN to the bin size, no ValueError
    # and its two docs are actually in the drawn membership
    assert {"a00", "a01"} <= set(out["doc_names"])


def test_total_sampled_matches_sum_of_bin_samples_and_doc_name_count(tmp_path):
    out = G.draw_sample(path=tmp_path / "s.json", speeches=_synthetic_speeches(), force=True)
    expected_total = sum(b["n_sampled"] for b in out["bins"])
    assert out["n_sampled"] == expected_total == len(out["doc_names"])
    # doc_names are sorted and unique
    assert out["doc_names"] == sorted(out["doc_names"])
    assert len(set(out["doc_names"])) == len(out["doc_names"])


# ---------------------------------------------------------------------------
# overwrite protection (the persisted file is the source of truth)
# ---------------------------------------------------------------------------


def test_refuses_to_overwrite_an_existing_sample_without_force(tmp_path):
    path = tmp_path / "s.json"
    G.draw_sample(path=path, speeches=_synthetic_speeches(), force=True)
    with pytest.raises(FileExistsError, match="source of truth"):
        G.draw_sample(path=path, speeches=_synthetic_speeches())  # force defaults False


def test_force_true_deliberately_overwrites(tmp_path):
    path = tmp_path / "s.json"
    G.draw_sample(path=path, speeches=_synthetic_speeches(), seed=1, force=True)
    before = json.loads(path.read_text())["seed"]
    G.draw_sample(path=path, speeches=_synthetic_speeches(), seed=42, force=True)
    after = json.loads(path.read_text())["seed"]
    assert before == 1 and after == 42  # the file was replaced


# ---------------------------------------------------------------------------
# provenance JSON: every field a downstream/truth run needs
# ---------------------------------------------------------------------------


def test_sample_json_records_full_provenance(tmp_path):
    path = tmp_path / "s.json"
    out = G.draw_sample(path=path, speeches=_synthetic_speeches(),
                        draw_date="2026-07-21", seed=20260721, force=True)
    on_disk = json.loads(path.read_text())
    assert on_disk == out  # what is returned is exactly what is persisted

    for field in ("version", "seed", "fraction", "min_per_bin", "era_anchor",
                  "era_bin_width", "selection_rule", "draw_date",
                  "n_corpus_speeches", "n_sampled", "corpus_fingerprint",
                  "bins", "doc_names", "speeches"):
        assert field in on_disk, f"missing provenance field {field!r}"

    assert on_disk["seed"] == 20260721
    assert on_disk["draw_date"] == "2026-07-21"
    assert on_disk["fraction"] == G.SAMPLE_FRACTION
    # fingerprint is the corpus content hash, not empty
    assert set(on_disk["corpus_fingerprint"]) >= {"n_speeches", "doc_name_sha256"}
    # each speech record carries its stratum for a clean downstream join
    assert on_disk["speeches"]
    first = on_disk["speeches"][0]
    assert set(first) >= {"doc_name", "era_bin", "era_label", "year"}


def test_selection_rule_names_the_committed_file_as_source_of_truth(tmp_path):
    out = G.draw_sample(path=tmp_path / "s.json", speeches=_synthetic_speeches(), force=True)
    assert "source of truth" in out["selection_rule"]


# ---------------------------------------------------------------------------
# load_sample / sample_doc_names round-trip and error surfaces
# ---------------------------------------------------------------------------


def test_load_sample_missing_file_errors_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="draw-sample"):
        G.load_sample(tmp_path / "absent.json")


def test_sample_doc_names_round_trips_a_drawn_file(tmp_path):
    path = tmp_path / "s.json"
    out = G.draw_sample(path=path, speeches=_synthetic_speeches(), force=True)
    assert G.sample_doc_names(path) == sorted(out["doc_names"])


def test_sample_doc_names_falls_back_to_speeches_list_when_no_doc_names_key(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"speeches": [{"doc_name": "z"}, {"doc_name": "a"}]}))
    assert G.sample_doc_names(path) == ["a", "z"]  # sorted


def test_sample_doc_names_errors_when_no_doc_names_present(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"doc_names": [], "speeches": []}))
    with pytest.raises(ValueError, match="no doc_names"):
        G.sample_doc_names(path)
