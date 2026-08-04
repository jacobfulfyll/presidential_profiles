"""Deterministic contracts for the public era-boundary analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import era_boundaries, era_boundaries_site, eras
from presidential_profiles.expansion_site import NAV


def _fingerprints() -> pd.DataFrame:
    rows = []
    for i, unit in enumerate(range(1784, 1880, 8)):
        for measure, group, value in [
            ("topic__Economy", "topic", float(i < 6)),
            ("marker_boosters", "marker", float(i >= 6)),
        ]:
            rows.append(
                {
                    "grain": "bin8",
                    "unit": str(float(unit)),
                    "measure": measure,
                    "axis_group": group,
                    "value_raw": value,
                    "n_paragraphs": 100,
                    "n_speeches": 10,
                }
            )
    return pd.DataFrame(rows)


def _markers(docs: list[str]) -> pd.DataFrame:
    rows = []
    for doc in docs:
        year = int(doc[1:])
        row = {"doc_name": doc, "n_words": 1000.0}
        for column in eras.MARKER_COLUMNS:
            row[column] = float(column == "boosters" and year >= 2008)
        rows.append(row)
    return pd.DataFrame(rows)


def _stats(docs: list[str]) -> pd.DataFrame:
    rows = []
    for doc in docs:
        row = {
            "doc_name": doc,
            "n_tokens": 1000.0,
            "fk_grade": 10.0,
            "words_per_sentence": 20.0,
            "i_count": 5.0,
            "we_count": 5.0,
        }
        row.update({column: 0.0 for column in eras.MODAL_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def _master() -> pd.DataFrame:
    rows = []
    for year in range(2000, 2016):
        rows.append(
            {
                "doc_name": f"d{year}",
                "para_idx": 0,
                "year": year,
                "era": "window",
                "president": "P",
                "bin8": (year // 8) * 8,
                "center_year": float(year),
                "speech_type": "annual_message",
                "topic__Economy": year >= 2008,
                "party_attack": False,
                "enemy_naming": False,
                "zero_sum": False,
                "proposal_values": "neither",
            }
        )
    return pd.DataFrame(rows)


def _two_axis_reference() -> pd.DataFrame:
    rows = []
    for unit, topic, marker in [("2000.0", 0.0, 0.0), ("2008.0", 1.0, 1.0)]:
        for measure, group, value in [
            ("topic__Economy", "topic", topic),
            ("marker_boosters", "marker", marker),
        ]:
            rows.append(
                {
                    "grain": "bin8",
                    "unit": unit,
                    "measure": measure,
                    "axis_group": group,
                    "value_raw": value,
                    "n_paragraphs": 8,
                    "n_speeches": 8,
                }
            )
    return pd.DataFrame(rows)


def test_cluster_stability_uses_only_observed_bin_starts():
    out = era_boundaries.build_cluster_stability(_fingerprints())
    observed = set(range(1784, 1880, 8))
    assert set(out["boundary_year"]).issubset(observed)
    assert set(out["condition"]) == {
        "full",
        "drop_topic",
        "drop_combat",
        "drop_register",
        "drop_marker",
        "drop_stat",
        "drop_genre",
        "drop_opponents",
    }
    assert out[(out["condition"] == "full") & (out["k"] == 9)].shape[0] == 8


def test_sliding_score_tests_the_only_possible_calendar_boundary():
    master = _master()
    docs = master["doc_name"].tolist()
    out = era_boundaries.sliding_transition_scores(
        master=master,
        fingerprints=_two_axis_reference(),
        markers=_markers(docs),
        stats=_stats(docs),
        taxonomy={"level2": [{"name": "Topic", "level1": "Economy"}]},
        window_years=8,
    )
    assert out["boundary_year"].tolist() == [2008]
    assert out.loc[0, "left_start"] == 2000
    assert out.loc[0, "right_end"] == 2015
    # Each raw difference is 1 and each two-bin population SD is .5.
    assert out.loc[0, "score"] == pytest.approx(np.sqrt(2 * 2.0**2))
    assert out.loc[0, "rank"] == 1


def test_consensus_combines_window_specific_rank_percentiles():
    sliding = pd.DataFrame(
        [
            {"boundary_year": year, "window_years": window, "score": score, "rank": rank}
            for window, values in (
                (4, [(1800, 9.0, 1), (1801, 5.0, 2), (1802, 1.0, 3)]),
                (8, [(1800, 1.0, 3), (1801, 5.0, 2), (1802, 9.0, 1)]),
            )
            for year, score, rank in values
        ]
    )
    out = era_boundaries.consensus_transition_scores(sliding)
    assert out["consensus"].tolist() == pytest.approx([0.5, 0.5, 0.5])
    assert out["consensus_rank"].tolist() == [1, 1, 1]


def test_persistent_regime_artifact_records_reviewed_near_optimum():
    analysis = era_boundaries.load()
    full = analysis["segmentation"]
    full = full[full["condition"] == "full"]
    assert full["segment_start"].tolist() == [
        1789, 1805, 1849, 1869, 1913, 1937, 1953, 1981, 2017
    ]
    assert analysis["meta"]["reviewed_grid_starts"] == [
        1789, 1809, 1849, 1869, 1913, 1933, 1953, 1981, 2017
    ]
    assert analysis["meta"]["reviewed_excess_objective_pct"] == pytest.approx(
        0.5121370210498188
    )


def test_public_page_explains_grid_artifact_and_presidential_calls():
    page = era_boundaries_site.render_era_boundaries(era_boundaries.load())
    assert "How we chose the eras" in page
    assert "Why did 1872 appear before?" in page
    assert "There was no presidential change in 1870" in page
    assert "1809 · Madison takes office" in page
    assert "Who owns a speech in an accession year?" in page
    assert "Jefferson’s April 1809 Albemarle County message" in page
    assert "contains only Grant in 1869" in page
    assert "governs story ownership only" in page
    assert "1869 · Grant takes office" in page
    assert "only 0.5% above the optimum" in page
    assert "Recommended scheme · live story ranges not yet migrated" in page
    assert "data/era-boundary-sliding-scores.csv" in page
    assert "data/era-boundary-segmentation.csv" in page
    for start in (1809, 1850, 1869, 1913, 1933, 1953, 1981, 2017):
        assert f'"year": {start}' in page
    assert '"year": 1878' not in page


def test_page_is_a_primary_navigation_destination():
    data_children = dict(NAV)["Data"]
    assert ("Era Choices", "era-boundaries.html") in data_children
