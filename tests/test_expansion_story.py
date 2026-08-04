"""Locked evidence and determinism contract for the 1816–1849 story layer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import expansion_story as E


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture(scope="module")
def real_inputs() -> dict[str, object]:
    return {
        "paragraphs": pd.read_parquet(DATA / "paragraphs.parquet"),
        "paragraph_issues": pd.read_parquet(
            DATA / "paragraph_issues.parquet"
        ),
        "speeches": pd.read_parquet(DATA / "speeches.parquet"),
        "primary_annotations": pd.read_parquet(
            DATA / "llm_annotations" / "paragraph_annotations.parquet"
        ),
        "secondary_annotations": pd.read_parquet(
            DATA
            / "llm_annotations"
            / "paragraph_annotations__opus4-8.parquet"
        ),
        "entities": pd.read_parquet(
            DATA / "llm_annotations" / "paragraph_entities.parquet"
        ),
        "taxonomy": json.loads(
            (DATA / "llm_annotations" / "taxonomy_v1.json").read_text()
        ),
    }


@pytest.fixture(scope="module")
def real_table(real_inputs) -> pd.DataFrame:
    return E.build_period_metrics(**real_inputs)


@pytest.fixture(scope="module")
def real_details(real_inputs) -> dict[str, object]:
    return E.derive_story_details(**real_inputs)


def _tiny_taxonomy() -> dict:
    labels = sorted({label for values in E.TOPIC_ROWS.values() for label in values})
    return {
        "level2": [{"name": label, "level1": "Test"} for label in labels]
    }


def _tiny_inputs() -> dict[str, object]:
    keys = [
        ("speech-a", 0),
        ("speech-a", 1),
        ("speech-b", 0),
    ]
    paragraphs = pd.DataFrame(
        {
            "doc_name": [row[0] for row in keys],
            "para_idx": [row[1] for row in keys],
            "text": ["alpha", "topic free", "beta"],
            "word_count": [1, 2, 1],
        }
    )
    paragraph_issues = paragraphs[E.KEYS].assign(year=[1816, 1816, 1817])
    topics = [
        [
            E.RELATIONS,
            E.RELATIONS.swapcase(),
            E.INDIAN_AFFAIRS,
        ],
        [],
        [E.PUBLIC_LANDS],
    ]
    primary = paragraphs[E.KEYS].assign(
        topics=topics, enemy_naming=[True, False, False]
    )
    secondary = paragraphs.iloc[[0]][E.KEYS].assign(
        topics=[[E.RELATIONS]], enemy_naming=[False]
    )
    speeches = pd.DataFrame(
        {
            "doc_name": ["speech-a", "speech-b"],
            "president": ["A", "B"],
            "year": [1816, 1817],
            "title": ["A", "B"],
        }
    )
    entities = pd.DataFrame(
        columns=["doc_name", "para_idx", "entity", "type", "stance"]
    )
    return {
        "paragraphs": paragraphs,
        "paragraph_issues": paragraph_issues,
        "speeches": speeches,
        "primary_annotations": primary,
        "secondary_annotations": secondary,
        "entities": entities,
        "taxonomy": _tiny_taxonomy(),
    }


def test_period_boundaries_and_preview_are_exact():
    assert E.PERIODS == (
        ("1816–20", 1816, 1820, False),
        ("1821–25", 1821, 1825, False),
        ("1826–30", 1826, 1830, False),
        ("1831–35", 1831, 1835, False),
        ("1836–40", 1836, 1840, False),
        ("1841–45", 1841, 1845, False),
        ("1846–49", 1846, 1849, False),
        ("1850–54", 1850, 1854, True),
    )
    assert E._period_label(1815) is None
    assert E._period_label(1816) == "1816–20"
    assert E._period_label(1849) == "1846–49"
    assert E._period_label(1854) == "1850–54"
    assert E._period_label(1855) is None


def test_full_key_set_validation_rejects_silent_inner_join_loss():
    inputs = _tiny_inputs()
    inputs["paragraph_issues"] = inputs["paragraph_issues"].iloc[:-1]
    with pytest.raises(ValueError, match="key sets diverge"):
        E.prepare_inputs(**inputs)


def test_taxonomy_normalization_unknown_labels_raise():
    inputs = _tiny_inputs()
    inputs["primary_annotations"] = inputs["primary_annotations"].copy()
    inputs["primary_annotations"].at[0, "topics"] = ["Invented topic"]
    with pytest.raises(ValueError, match="not in taxonomy_v1"):
        E.prepare_inputs(**inputs)


def test_duplicate_labels_and_composite_union_count_each_paragraph_once():
    master, _, _ = E.prepare_inputs(**_tiny_inputs())
    indicators = E._indicator_frame(
        master, "topics_primary", "enemy_naming_primary"
    )
    first = master.iloc[0]
    assert first["topics_primary"] == frozenset(
        {E.RELATIONS, E.INDIAN_AFFAIRS}
    )
    assert bool(indicators.iloc[0]["Territorial expansion"])
    assert bool(indicators.iloc[0]["↳ External acquisition"])
    assert bool(indicators.iloc[0]["↳ Native removal and settlement"])
    assert int(indicators["Territorial expansion"].sum()) == 2


def test_topic_free_paragraphs_stay_in_the_denominator():
    master, _, _ = E.prepare_inputs(**_tiny_inputs())
    indicators = E._indicator_frame(
        master, "topics_primary", "enemy_naming_primary"
    )
    estimate = E.bootstrap_period(
        indicators,
        ["Territorial expansion"],
        period_start=1816,
        n_draws=20,
    )
    assert estimate["n_paragraphs"] == 3
    assert estimate["n_labeled"][0] == 2
    assert estimate["observed"][0] == pytest.approx(2 / 3)


def test_bootstrap_resamples_speeches_not_paragraphs():
    # One 100-paragraph speech has no hits; one one-paragraph speech has a hit.
    # A paragraph bootstrap would put the upper bound near 3%. A cluster
    # bootstrap repeatedly draws the one-paragraph speech twice and reaches 100%.
    indicators = pd.DataFrame(
        {
            "doc_name": ["long"] * 100 + ["short"],
            "period": ["1816–20"] * 101,
            "x": [False] * 100 + [True],
        }
    )
    estimate = E.bootstrap_period(
        indicators, ["x"], period_start=1816, n_draws=500
    )
    assert estimate["observed"][0] == pytest.approx(1 / 101)
    assert estimate["hi"][0] == pytest.approx(1.0)


def test_bootstrap_seed_is_deterministic():
    indicators = pd.DataFrame(
        {
            "doc_name": np.repeat(["a", "b", "c", "d"], [2, 3, 4, 5]),
            "period": ["1816–20"] * 14,
            "x": [
                True,
                False,
                True,
                True,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
            ],
        }
    )
    left = E.bootstrap_period(
        indicators, ["x"], period_start=1816, n_draws=73, seed=123
    )
    right = E.bootstrap_period(
        indicators, ["x"], period_start=1816, n_draws=73, seed=123
    )
    np.testing.assert_array_equal(left["lo"], right["lo"])
    np.testing.assert_array_equal(left["hi"], right["hi"])


def test_confidence_and_paired_model_gates_remain_separate():
    one_speech = pd.DataFrame(
        {"doc_name": ["a", "a"], "period": ["1816–20"] * 2, "x": [False, False]}
    )
    suppressed = E.bootstrap_period(
        one_speech, ["x"], period_start=1816, n_draws=20
    )
    assert suppressed["ci_status"] == "suppressed_n_floor"
    assert np.isnan(suppressed["lo"][0])
    assert not suppressed["interval_unresolvable"][0]

    two_speeches = pd.DataFrame(
        {"doc_name": ["a", "b"], "period": ["1816–20"] * 2, "x": [False, False]}
    )
    unresolved = E.bootstrap_period(
        two_speeches, ["x"], period_start=1816, n_draws=20
    )
    assert unresolved["ci_status"] == "low_cluster_caution"
    assert unresolved["interval_unresolvable"][0]
    assert np.isnan(unresolved["lo"][0])

    def paired(n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "x__primary": [True] * n,
                "x__secondary": [False] * n,
            }
        )

    assert E._paired_disagreement(paired(49), ["x"])["available"] is False
    available = E._paired_disagreement(paired(50), ["x"])
    assert available["available"] is True
    assert available["half_width"][0] == pytest.approx(0.5)


def _cell(table: pd.DataFrame, period: str, row: str) -> pd.Series:
    return table[
        table["period"].eq(period) & table["display_row"].eq(row)
    ].iloc[0]


def test_real_numerical_acceptance_cells_are_rederived(real_table):
    assert len(real_table) == 72
    assert _cell(real_table, "1831–35", "Banking")["point"] * 100 == pytest.approx(
        23.3, abs=0.05
    )
    assert _cell(
        real_table, "1831–35", E.ADVERSARY_ROW
    )["point"] * 100 == pytest.approx(36.5, abs=0.05)
    assert _cell(
        real_table, "1846–49", "Territorial expansion"
    )["point"] * 100 == pytest.approx(56.4, abs=0.05)
    assert _cell(
        real_table, "1846–49", "↳ External acquisition"
    )["point"] * 100 == pytest.approx(49.2, abs=0.05)
    assert _cell(
        real_table, "1846–49", E.ADVERSARY_ROW
    )["point"] * 100 == pytest.approx(27.3, abs=0.05)
    assert _cell(
        real_table, "1850–54", "Slavery and sectionalism"
    )["point"] * 100 == pytest.approx(37.7, abs=0.05)


def test_every_published_interval_widens_or_preserves_sampling(real_table):
    applied = real_table[real_table["disagreement_band_applied"]]
    assert not applied.empty
    assert (applied["lo"] <= applied["lo_sampling"]).all()
    assert (applied["hi"] >= applied["hi_sampling"]).all()
    unavailable = real_table[
        real_table["disagreement_status"].eq(E.bands.THIN_PAIRED)
    ]
    assert set(unavailable["period"]) == {"1816–20", "1821–25", "1846–49"}
    assert unavailable["disagreement_half_width"].isna().all()


def test_exact_receipt_keys_and_excerpts_validate(real_inputs, real_details):
    receipts = real_details["receipts"]
    assert len(receipts) == len(E.RECEIPT_SPECS)
    assert {receipt["group"] for receipt in receipts} == {
        "monroe",
        "jackson",
        "polk",
    }
    paragraph_lookup = real_inputs["paragraphs"].set_index(E.KEYS)
    for receipt in receipts:
        key = (receipt["doc_name"], receipt["para_idx"])
        assert key in paragraph_lookup.index
        assert receipt["excerpt"] in paragraph_lookup.loc[key, "text"]


def test_jackson_character_metrics_are_derived(real_details):
    character = real_details["jackson_character"]
    assert character["bank_veto"]["n_enemy_naming"] == 30
    assert character["bank_veto"]["n_paragraphs"] == 56
    assert character["bank_veto"]["share"] == pytest.approx(30 / 56)
    assert character["nullification_proclamation"]["share"] == pytest.approx(
        31 / 60
    )
    assert character["removal_message"]["share"] == 0
    assert character["removal_message"]["n_paragraphs"] == 3


def test_period_entity_annotations_are_derived(real_details):
    annotations = real_details["entity_annotations"]
    domestic = annotations["1831–35"]
    foreign = annotations["1846–49"]
    assert domestic["dominant_type"] == "institution"
    assert domestic["dominant_type_share"] == pytest.approx(154 / 290)
    assert [row["entity"] for row in domestic["recurring_names"]] == [
        "Bank of the United States",
        "Senate",
    ]
    assert foreign["dominant_type"] == "nation"
    assert foreign["dominant_type_share"] == pytest.approx(149 / 193)
    assert foreign["recurring_names"][0]["entity"] == "Mexico"


def test_accessible_table_contains_every_cell_interval_support_and_status(
    real_table,
):
    rendered = E.accessible_table_html(real_table)
    assert rendered.count("<tr") == 10  # one header + eight topics + adversary
    assert "Adversary naming · enemy_naming=True" in rendered
    assert "95% interval" in rendered
    assert "paragraphs ·" in rendered and "speeches" in rendered
    assert "confidence:" in rendered and "model disagreement:" in rendered
    for _, row in real_table.iterrows():
        assert f"{row['point'] * 100:.1f}%" in rendered
    for label in E.TOPIC_ROWS:
        assert label in rendered
    for period, *_ in E.PERIODS:
        assert period in rendered
