"""Fail-closed reader, projection, rendering, and public-download contracts."""

from __future__ import annotations

from copy import deepcopy
import json

import pandas as pd
import pytest

from presidential_profiles import (
    era_profiles,
    foundation_audit,
    site,
    story_foundation as sf,
)


@pytest.fixture(scope="module")
def bundle() -> sf.StoryFoundationBundle:
    return sf.load_story_foundation()


@pytest.fixture(scope="module")
def governed_metadata():
    return (
        json.loads(sf.SPEAKER_META_PATH.read_text()),
        json.loads(sf.REFERENCE_META_PATH.read_text()),
        foundation_audit.check_foundation(),
    )


def _rehash(meta: dict) -> dict:
    meta["metadata_sha256"] = sf._metadata_hash(meta)
    return meta


def test_real_population_and_projection_receipt(bundle):
    receipt = sf.check_story_contract(bundle)
    assert receipt["population"] == {
        "retained_attribution_rows": 35_394,
        "eligible_presidential_paragraphs": 32_531,
        "excluded_paragraphs": 2_863,
        "cross_owner_presidential_paragraphs": 296,
        "appearances": 1_054,
    }
    assert receipt["era_denominators"] == [
        672, 3_847, 2_610, 6_721, 2_134, 1_409, 5_583, 6_144, 3_411,
    ]
    assert receipt["era_candidates"] == 45
    assert receipt["source_agreement"] == {
        "AI + NER": 29, "AI only": 16, "NER only": 0,
    }
    assert receipt["invocation_overlay"] == sf.EXPECTED_INVOCATION_OVERLAY
    projected = sf.project_distinctive_references(bundle)
    assert list(projected) == [spec.key for spec in era_profiles.ERA_PROFILE_SPECS]
    assert all(len(record["rows"]) == 5 for record in projected.values())
    assert sum(
        record["denominator"]["paragraphs"] for record in projected.values()
    ) == 32_531


def test_carter_reagan_and_excluded_speaker_regressions(bundle):
    doc = (
        "/the-presidency/presidential-speeches/"
        "october-28-1980-debate-ronald-reagan"
    )
    rows = bundle.paragraph_view[bundle.paragraph_view["doc_name"].eq(doc)]
    assert len(rows) == 151
    assert rows["attributed_speaker"].value_counts().to_dict() == {
        "Jimmy Carter": 44,
        "Ronald Reagan": 42,
    }
    assert rows["speaker_outcome"].value_counts().to_dict() == {
        "canonical_president": 86,
        "multiple_speakers": 43,
        "non_president": 22,
    }
    assert not rows.loc[
        rows["speaker_outcome"].ne("canonical_president"),
        "analysis_eligible",
    ].any()
    appearances = bundle.appearances[bundle.appearances["doc_name"].eq(doc)]
    assert set(appearances["attributed_speaker"]) == {
        "Jimmy Carter", "Ronald Reagan",
    }


def test_entity_support_deduplicates_paragraphs_and_keeps_empty_denominators(bundle):
    eligible = bundle.paragraph_view[bundle.paragraph_view["analysis_eligible"]]
    entity_keys = bundle.entity_mentions[
        bundle.entity_mentions["analysis_eligible"]
    ][["doc_name", "para_idx"]].drop_duplicates()
    assert len(entity_keys) < len(eligible)
    for row in bundle.era_distinctive.itertuples(index=False):
        support = bundle.entity_mentions[
            bundle.entity_mentions["analysis_eligible"]
            & bundle.entity_mentions["ai_entity"].notna()
            & bundle.entity_mentions["story_era_key"].eq(row.story_era_key)
            & bundle.entity_mentions["normalized_entity"].eq(
                row.normalized_entity
            )
        ]
        assert len(support[["doc_name", "para_idx"]].drop_duplicates()) == (
            row.era_paragraphs
        )
        assert row.era_paragraphs >= 5
        assert row.era_documents >= 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("identity", "metadata identity"),
        ("fingerprint", "corpus_fingerprint identity"),
        ("run", "attribution_run identity"),
        ("generation", "annotation_generation identity"),
        ("schema", "schema drift"),
        ("hash", "artifact hash drift"),
        ("threshold", "threshold metadata drift"),
    ],
)
def test_metadata_and_provenance_drift_is_refused(
    governed_metadata, mutation, message
):
    speaker, reference, audit = map(deepcopy, governed_metadata)
    if mutation == "identity":
        speaker["annotation_generation"] = "tampered"
    elif mutation == "fingerprint":
        reference["canonical_corpus_fingerprint"] = "tampered"
        _rehash(reference)
    elif mutation == "run":
        reference["attribution_run_id"] = "tampered"
        _rehash(reference)
    elif mutation == "generation":
        reference["annotation_generation"] = "tampered"
        _rehash(reference)
    elif mutation == "schema":
        reference["entity_schema_version"] = "future-schema"
        _rehash(reference)
    elif mutation == "hash":
        reference["artifacts"][sf.ENTITY_MENTIONS_PATH.name] = "sha256:bad"
        _rehash(reference)
    elif mutation == "threshold":
        reference["thresholds"]["min_source_documents"] = 1
        _rehash(reference)
    with pytest.raises(sf.StoryFoundationError, match=message):
        sf._validate_metadata(speaker, reference, audit)


@pytest.mark.parametrize(
    ("attribute", "label"),
    [
        ("paragraph_view", "paragraph view has duplicate keys"),
        ("entity_mentions", "entity mentions has duplicate keys"),
        ("era_distinctive", "era candidate ranks has duplicate keys"),
    ],
)
def test_duplicate_governed_keys_are_refused(bundle, attribute, label):
    frames = {
        "paragraph_view": bundle.paragraph_view,
        "appearances": bundle.appearances,
        "entity_mentions": bundle.entity_mentions,
        "era_distinctive": bundle.era_distinctive,
    }
    frames[attribute] = pd.concat(
        [frames[attribute], frames[attribute].iloc[[0]]], ignore_index=True
    )
    with pytest.raises(sf.StoryFoundationError, match=label):
        sf._validate_frames(
            frames["paragraph_view"],
            frames["appearances"],
            frames["entity_mentions"],
            frames["era_distinctive"],
        )


def test_candidate_sort_uses_every_deterministic_tie_break():
    rows = pd.DataFrame([
        {"normalized_entity": "z", "log_odds": 3.0, "era_paragraphs": 7, "era_documents": 2},
        {"normalized_entity": "y", "log_odds": 4.0, "era_paragraphs": 5, "era_documents": 2},
        {"normalized_entity": "x", "log_odds": 3.0, "era_paragraphs": 8, "era_documents": 2},
        {"normalized_entity": "w", "log_odds": 3.0, "era_paragraphs": 7, "era_documents": 3},
        {"normalized_entity": "a", "log_odds": 3.0, "era_paragraphs": 7, "era_documents": 2},
    ])
    assert sf._sort_candidates(rows)["normalized_entity"].tolist() == [
        "y", "x", "w", "a", "z",
    ]


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("evidence_para_idx", -1, "does not resolve"),
        ("evidence_excerpt", "wrong text", "disagrees with entity row"),
        ("source_badge", "NER only", "prohibited NER-only"),
    ],
)
def test_candidate_evidence_and_badge_drift_is_refused(
    bundle, column, value, message
):
    candidates = bundle.era_distinctive.copy()
    candidates.loc[candidates.index[0], column] = value
    with pytest.raises(sf.StoryFoundationError, match=message):
        sf._validate_frames(
            bundle.paragraph_view,
            bundle.appearances,
            bundle.entity_mentions,
            candidates,
        )


def test_reference_landscape_receipts_escape_every_string(bundle):
    profiles_by_era = era_profiles.load_all_era_profiles()
    profile = deepcopy(profiles_by_era["founding"])
    row = next(
        item["highlight"]
        for item in profile["reference_landscape"]["types"]
        if item["highlight"] is not None
    )
    row["label"] = '<script data-x="name">'
    row["source_agreement"]["badge"] = 'AI only <badge>'
    row["evidence"].update({
        "excerpt": '<img src=x onerror="bad"> long evidence',
        "ai_mention": '<AI & mention>',
        "ner_mention": '<NER mention>',
        "actual_speaker": '<Actual Speaker>',
        "document_owner": '<Owner & Co>',
        "cross_owner": True,
        "title": '<Document "Title">',
        "source_url": 'https://example.test/?q="bad"&x=<x>',
        "doc_name": '<doc>',
    })
    body = site._era_profile_html(profile, "escape-profile")
    assert '<script data-x="name">' not in body
    assert '<img src=x onerror="bad">' not in body
    assert "&lt;script data-x=&quot;name&quot;&gt;" in body
    assert "AI only &lt;badge&gt;" in body
    assert "Actual speaker: &lt;Actual Speaker&gt;" in body
    assert "source document cataloged under &lt;Owner &amp; Co&gt;" in body
    assert '<details class="era-reference-highlight" open>' not in body


def test_excerpt_truncation_is_word_bounded():
    text = "word " * 100
    excerpt = site._word_boundary_excerpt(text, 360)
    assert len(excerpt) <= 360
    assert excerpt.endswith("…")
    assert excerpt.removesuffix("…").endswith("word")


def test_public_downloads_and_json_projection_are_exact(bundle, tmp_path):
    paths = sf.write_story_downloads(bundle, tmp_path)
    assert len(paths) == 4
    download_receipt = sf._check_public_downloads(bundle, tmp_path)
    assert download_receipt["era_rows"] == 45
    assert download_receipt["appearance_rows"] == 1_054
    projected = sf.project_distinctive_references(bundle)
    data_dir = tmp_path / "data"
    (data_dir / "era_profiles.json").write_text(json.dumps({
        "schema_version": "era-profile-v6",
        "profiles": {
            key: {"distinctive_references": value}
            for key, value in projected.items()
        },
    }))
    (data_dir / "era_visualizations.json").write_text(json.dumps({
        "schema_version": "era-visualizations-v9",
    }))
    (data_dir / "era_contextualizations.json").write_text(json.dumps({
        "schema_version": "era-contextualizations-v11",
        "contextualizations": {
            "founding": {"support": {
                "invocation_overlay": sf.EXPECTED_INVOCATION_OVERLAY,
            }},
        },
    }))
    assert sf._check_public_json(bundle, tmp_path)["profiles"] == 9
    source_hash = sf._sha256_file(sf.ENTITY_MENTIONS_PATH)
    assert sf._sha256_file(
        tmp_path / "data" / "story" / sf.PUBLIC_ENTITIES_NAME
    ) == source_hash
