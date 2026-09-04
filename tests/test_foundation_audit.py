"""Contracts for the speaker/reference-entity foundation."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from presidential_profiles import era_profiles
from presidential_profiles import foundation_audit as foundation


def _synthetic_speaker_inputs():
    doc = "/the-presidency/presidential-speeches/example"
    paragraphs = pd.DataFrame(
        [
            {"doc_name": doc, "para_idx": index, "text": text, "word_count": len(text.split())}
            for index, text in enumerate(
                [
                    "Carter addresses Congress.",
                    "Reagan answers Carter.",
                    "The moderator asks a question.",
                    "Carter concludes.",
                ]
            )
        ]
    )
    speeches = pd.DataFrame(
        [
            {
                "doc_name": doc,
                "date": pd.Timestamp("1980-10-28"),
                "title": "Example debate",
                "year": 1980,
                "president": "Jimmy Carter",
            }
        ]
    )
    outcomes = [
        ("Jimmy Carter", "jimmy-carter", "canonical_president"),
        ("Ronald Reagan", "ronald-reagan", "canonical_president"),
        (None, None, "non_president"),
        ("Jimmy Carter", "jimmy-carter", "canonical_president"),
    ]
    rows = []
    for paragraph, (speaker, profile, outcome) in zip(
        paragraphs.to_dict("records"), outcomes, strict=True
    ):
        audited = outcome == "canonical_president"
        rows.append(
            {
                "doc_name": doc,
                "para_idx": paragraph["para_idx"],
                "source_text_sha256": foundation._sha256_text(paragraph["text"]),
                "document_owner": "Jimmy Carter",
                "document_owner_profile_id": "jimmy-carter",
                "attributed_speaker": speaker,
                "attributed_speaker_profile_id": profile,
                "speaker_outcome": outcome,
                "speaker_reason_code": "explicit_speaker_cue",
                "speaker_reason": "fixture",
                "evidence_para_indices_json": "[]",
                "classification_source": "fixture",
                "review_status": "fixture",
                "document_class": "multi_speaker",
                "speech_type": "campaign_or_debate",
                "canonical_document_owner": True,
                "speaker_audited_all": audited,
                "debate_excluded": False,
                "single_president_documents": False,
                "annual_message_strict": False,
                "spec_version": "speaker-attribution-v1",
                "run_id": "fixture",
            }
        )
    return paragraphs, speeches, pd.DataFrame(rows)


def test_speaker_view_builds_actual_speaker_appearances_and_exclusion_receipts():
    paragraphs, speeches, attribution = _synthetic_speaker_inputs()
    view, appearances, coverage = foundation.build_speaker_views(
        paragraphs, speeches, attribution
    )
    assert len(view) == 4
    assert view["analysis_eligible"].sum() == 3
    assert view["cross_owner_paragraph"].sum() == 1
    assert view.loc[view["para_idx"].eq(2), "exclusion_reason"].item() == "non_president"
    assert set(appearances["attributed_speaker"]) == {"Jimmy Carter", "Ronald Reagan"}
    carter = appearances.set_index("attributed_speaker").loc["Jimmy Carter"]
    reagan = appearances.set_index("attributed_speaker").loc["Ronald Reagan"]
    assert json.loads(carter["para_indices_json"]) == [0, 3]
    assert "Reagan answers Carter." not in carter["appearance_text"]
    assert reagan["cross_owner_appearance"]
    assert coverage.set_index("speaker_outcome").loc["non_president", "n_paragraphs"] == 1


def test_speaker_view_refuses_diverging_key_sets():
    paragraphs, speeches, attribution = _synthetic_speaker_inputs()
    with pytest.raises(foundation.FoundationError, match="key-set mismatch"):
        foundation.build_speaker_views(
            paragraphs.iloc[:-1], speeches, attribution
        )


def test_aliases_are_safe_and_historical_successors_remain_distinct():
    assert foundation.canonical_entity("  U.S.A. ") == "united states"
    assert foundation.canonical_entity("The   United States") == "united states"
    for left, right in foundation.DELIBERATELY_UNMERGED:
        assert foundation.canonical_entity(left) != foundation.canonical_entity(right)


def test_hybrid_matching_is_paragraph_local_and_keeps_nullable_stance():
    paragraphs, speeches, attribution = _synthetic_speaker_inputs()
    view, _, _ = foundation.build_speaker_views(paragraphs, speeches, attribution)
    ai = pd.DataFrame(
        [
            {
                "doc_name": view.iloc[0].doc_name,
                "para_idx": 0,
                "ai_entity": "U.S.A.",
                "ai_type": "nation",
                "ai_stance": "neutral",
                "ai_run_id": "primary",
                "ai_spec_version": "v1",
                "ai_item_index": 0,
                "ai_promotion_channel": "primary",
                "ai_label_id": "one",
                "normalized_entity_raw": "u.s.a.",
                "normalized_entity": "united states",
            },
            {
                "doc_name": view.iloc[0].doc_name,
                "para_idx": 1,
                "ai_entity": "Congress",
                "ai_type": "institution",
                "ai_stance": "favorable",
                "ai_run_id": "primary",
                "ai_spec_version": "v1",
                "ai_item_index": 0,
                "ai_promotion_channel": "primary",
                "ai_label_id": "two",
                "normalized_entity_raw": "congress",
                "normalized_entity": "congress",
            },
        ]
    )
    ner_metadata = {
        column: view.iloc[0][column]
        for column in [
            "speaker_outcome",
            "analysis_eligible",
            "attributed_speaker",
            "attributed_speaker_profile_id",
            "document_owner",
            "document_owner_profile_id",
            "cross_owner_paragraph",
            "year",
            "story_era_key",
            "story_era",
            "title",
            "source_url",
            "speech_type",
        ]
    }
    ner = pd.DataFrame(
        [
            {
                "doc_name": view.iloc[0].doc_name,
                "para_idx": 0,
                **ner_metadata,
                "ner_text": "United States",
                "start_char": 0,
                "end_char": 13,
                "ner_label": "GPE",
                "display_type": "place_or_political_entity",
                "normalized_entity_raw": "united states",
                "normalized_entity": "united states",
            },
            {
                "doc_name": view.iloc[0].doc_name,
                "para_idx": 0,
                **ner_metadata,
                "ner_text": "Carter",
                "start_char": 14,
                "end_char": 20,
                "ner_label": "PERSON",
                "display_type": "person",
                "normalized_entity_raw": "carter",
                "normalized_entity": "carter",
            },
        ]
    )
    hybrid = foundation.build_hybrid_entities(view, ai, ner)
    by = hybrid.set_index(["para_idx", "normalized_entity"])
    assert by.loc[(0, "united states"), "source_badge"] == "AI + NER"
    assert by.loc[(1, "congress"), "source_badge"] == "AI only"
    assert by.loc[(0, "carter"), "source_badge"] == "NER only"
    assert pd.isna(by.loc[(0, "carter"), "ai_stance"])
    # Congress in a different paragraph must not match a paragraph-0 NER span.
    assert not by.loc[(1, "congress"), "source_agreement"]


def test_local_ner_preserves_offsets_and_attribution_metadata():
    spacy = pytest.importorskip("spacy")
    paragraphs, speeches, attribution = _synthetic_speaker_inputs()
    view, _, _ = foundation.build_speaker_views(paragraphs, speeches, attribution)
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "PERSON", "pattern": "Carter"}])
    mentions = foundation.run_ner(view, nlp=nlp)
    first = mentions.iloc[0]
    assert first["ner_text"] == "Carter"
    assert first["text"] if "text" in first else True
    source = view.loc[
        view["doc_name"].eq(first["doc_name"])
        & view["para_idx"].eq(first["para_idx"]),
        "text",
    ].item()
    assert source[first["start_char"] : first["end_char"]] == "Carter"
    assert first["speaker_outcome"] == "canonical_president"


def test_era_ranking_uses_all_eligible_paragraphs_and_publishes_five_per_era():
    paragraph_rows = []
    entity_rows = []
    for era_index, spec in enumerate(era_profiles.ERA_PROFILE_SPECS):
        for para_idx in range(10):
            doc_name = f"/{spec.key}/doc-{para_idx % 2}"
            paragraph_rows.append(
                {
                    "doc_name": doc_name,
                    "para_idx": para_idx,
                    "text": f"paragraph {para_idx}",
                    "analysis_eligible": True,
                    "story_era_key": spec.key,
                    "story_era": spec.label,
                }
            )
        for entity_index in range(5):
            for para_idx in range(5):
                doc_name = f"/{spec.key}/doc-{para_idx % 2}"
                entity_rows.append(
                    {
                        "doc_name": doc_name,
                        "para_idx": para_idx,
                        "analysis_eligible": True,
                        "story_era_key": spec.key,
                        "story_era": spec.label,
                        "normalized_entity": f"entity-{era_index}-{entity_index}",
                        "ai_entity": f"Entity {era_index}-{entity_index}",
                        "ai_type": "group",
                        "ai_stance": "neutral",
                        "ner_text": f"Entity {era_index}-{entity_index}",
                        "source_agreement": True,
                        "source_badge": "AI + NER",
                        "title": "Fixture",
                        "source_url": "https://example.test",
                        "attributed_speaker": "George Washington",
                        "document_owner": "George Washington",
                        "cross_owner_paragraph": False,
                        "text": f"evidence {para_idx}",
                    }
                )
    paragraphs = pd.DataFrame(paragraph_rows)
    hybrid = pd.DataFrame(entity_rows)
    ranked = foundation.rank_era_entities(paragraphs, hybrid)
    assert len(ranked) == 45
    assert ranked.groupby("story_era_key").size().eq(5).all()
    assert set(ranked["era_denominator_paragraphs"]) == {10}
    assert set(ranked["era_paragraphs"]) == {5}
    assert set(ranked["era_documents"]) == {2}


def test_consumer_inventory_declares_units_denominators_and_status():
    payload = foundation._consumer_payload()
    required = {
        "consumer",
        "current_unit",
        "replacement_unit",
        "denominator",
        "migration_status",
    }
    assert len(payload["consumers"]) >= 15
    assert all(required <= set(row) for row in payload["consumers"])
    assert any(row["consumer"] == "combat.president_conflict_v2" for row in payload["consumers"])


def test_real_foundation_fast_acceptance_contract():
    result = foundation.check_foundation()
    assert result["status"] == "accepted"
    assert result["retained_attribution_rows"] == 35_394
    assert result["eligible_presidential_paragraphs"] == 32_531
    assert result["excluded_paragraphs"] == 2_863
    assert result["cross_owner_presidential_paragraphs"] == 296
    regression = result["carter_reagan_regression"]
    assert regression["carter_paragraphs"] == 44
    assert regression["reagan_paragraphs"] == 42
    assert regression["non_president_paragraphs"] == 22
    assert regression["multiple_speaker_paragraphs"] == 43
