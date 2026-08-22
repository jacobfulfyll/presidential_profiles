"""Website adapters for the frozen AI annotation layer."""

import pandas as pd
import pytest

from presidential_profiles import (
    ai_labels,
    compare_site,
    methodology_site,
    profiles,
    profiles_site,
)


@pytest.fixture
def tiny_ai_data():
    taxonomy = {
        "level1": [
            {"name": "Policy", "definition": "Policy matters.", "kind": "policy"},
            {"name": "Values", "definition": "Non-policy values.", "kind": "non-policy"},
        ],
        "level2": [
            {"name": "Alpha Topic", "definition": "Alpha.", "level1": "Policy"},
            {"name": "Beta Topic", "definition": "Beta.", "level1": "Policy"},
            {"name": "Civic Values", "definition": "Values.", "level1": "Values"},
        ],
    }
    speeches = pd.DataFrame([
        {"doc_name": "a", "president": "President A", "year": 1900},
        {"doc_name": "b", "president": "President B", "year": 1901},
        {"doc_name": "c", "president": "President C", "year": 1902},
    ])
    annotations = pd.DataFrame([
        {"doc_name": "a", "para_idx": 0,
         "topics": ["Alpha Topic", "alpha topic"], "party_attack": True,
         "enemy_naming": False, "zero_sum": False, "proposal_values": "proposal"},
        {"doc_name": "a", "para_idx": 1,
         "topics": [], "party_attack": False,
         "enemy_naming": True, "zero_sum": False, "proposal_values": "values"},
        {"doc_name": "b", "para_idx": 0,
         "topics": ["Beta Topic"], "party_attack": False,
         "enemy_naming": False, "zero_sum": True, "proposal_values": "mixed"},
        {"doc_name": "c", "para_idx": 0,
         "topics": ["Civic Values"], "party_attack": False,
         "enemy_naming": False, "zero_sum": False, "proposal_values": "neither"},
    ])
    speech_annotations = pd.DataFrame([
        {"doc_name": "a", "speech_type": "public_remarks_or_address",
         "audience": "general_public", "medium": "spoken_address"},
        {"doc_name": "b", "speech_type": "state_of_the_union_or_annual_message",
         "audience": "congress", "medium": "written_message"},
        {"doc_name": "c", "speech_type": "campaign_or_debate",
         "audience": "general_public", "medium": "debate"},
    ])
    entities = pd.DataFrame([
        {"doc_name": "a", "para_idx": 1, "entity": "Rival",
         "type": "person", "stance": "adversarial"},
    ])
    return ai_labels.build_ai_data(
        speeches=speeches, annotations=annotations,
        speech_annotations=speech_annotations, entities=entities,
        taxonomy=taxonomy,
    )


def test_topic_case_variants_and_duplicates_count_once(tiny_ai_data):
    a = tiny_ai_data["by_president"]["President A"]
    alpha = next(t for t in a["top_topics"] if t["name"] == "Alpha Topic")
    assert alpha["n"] == 1
    assert alpha["share"] == pytest.approx(50.0)
    assert tiny_ai_data["summary"]["n_assignments"] == 3


def test_empty_topic_paragraph_stays_in_denominator(tiny_ai_data):
    summary = tiny_ai_data["summary"]
    assert summary["empty_topics"] == 1
    assert summary["empty_share"] == pytest.approx(25.0)
    assert summary["mean_topics"] == pytest.approx(0.75)


def test_president_payload_carries_every_ai_surface(tiny_ai_data):
    a = tiny_ai_data["by_president"]["President A"]
    assert a["flags"] == {
        "party_attack": 50.0, "enemy_naming": 50.0, "zero_sum": 0.0,
    }
    assert a["proposal_values"]["proposal"] == 50.0
    assert a["proposal_values"]["values"] == 50.0
    assert a["speech_types"][0]["name"] == "Public remarks / address"
    assert a["adversaries"] == [{"name": "Rival", "n": 1}]


def test_explorer_exposes_every_ai_topic(tiny_ai_data):
    series = ai_labels.explorer_topic_series(tiny_ai_data, min_year_paragraphs=1)
    assert set(series) == {
        "AI topic · Alpha Topic", "AI topic · Beta Topic", "AI topic · Civic Values",
    }
    # A has one Alpha paragraph out of two in 1900.
    assert series["AI topic · Alpha Topic"]["y"] == [1900, 1901, 1902]
    assert series["AI topic · Alpha Topic"]["v"] == [50.0, 0.0, 0.0]


def test_topic_bucket_series_uses_paragraph_weighted_10_and_20_year_buckets(
    tiny_ai_data,
):
    ten = ai_labels.topic_bucket_series(
        tiny_ai_data, bucket_years=10, min_paragraphs=1
    )
    twenty = ai_labels.topic_bucket_series(
        tiny_ai_data, bucket_years=20, min_paragraphs=1
    )
    # Four paragraphs share the 1900 bucket; one carries Alpha Topic.
    assert ten["Alpha Topic"] == {"x": [1900], "v": [25.0], "n": [4]}
    assert twenty["Alpha Topic"] == {"x": [1900], "v": [25.0], "n": [4]}
    with pytest.raises(ValueError, match="10 or 20"):
        ai_labels.topic_bucket_series(tiny_ai_data, bucket_years=5)


def test_methodology_is_an_article_with_vector_images(tiny_ai_data):
    page = methodology_site.render_methodology(tiny_ai_data)
    assert "How a speech becomes a label" in page
    assert page.count("<svg") >= 2
    assert "The complete frozen taxonomy" in page
    assert "Alpha Topic" in page
    assert "second opinion" in page.lower()


def test_compare_payload_uses_real_v3_profile_model_and_keeps_thin_state(tiny_ai_data):
    row = {
        "party": "Test", "first_year": 1900, "last_year": 1901,
        "n_speeches": 1, "n_words": 1000,
        "certainty": 0.6, "hype": 2, "mechanism": 3,
        "nrc_hope": 4, "nrc_fear": 5, "fk_grade": 6,
        "us_them": 2, "self_reference": 0.3, "ttr": 0.5,
        "religiosity": 1,
    }
    for key, _ in profiles.RADAR_AXES:
        row[f"pct_{key}"] = 50
    data = {
        "scores": pd.DataFrame([row], index=["President A"]),
        "issues": pd.DataFrame([{
            "n_paragraphs": 2, "share_Issue": 0.5, "rel_Issue": 2.0,
        }], index=["President A"]),
        "issue_cards": {"President A": {
            "n_paragraphs": 2, "cards": [], "voice": [], "low_confidence": True,
        }},
        "ai": tiny_ai_data,
        "distinctive": pd.DataFrame([{
            "president": "President A", "term": "word", "z": 2.0, "rank": 0,
        }]),
        "signatures": {"President A": [{
            "title": "Speech", "year": 1900, "url": "example-address",
        }]},
        "invokes": {"President A": []},
        "invoked_by": {"President A": None},
        "voice_neighbors": {"President A": []},
        "agenda_neighbors": {"President A": []},
        "invocation_v2": {"President A": []},
        "feature_neighbors": {"President A": {}},
    }
    view = profiles_site.profile_view_model("President A", data, ["Issue"])
    shared = profiles_site.profile_public_payload(view)
    compare_payload = compare_site.build_payload(
        data,
        ["Issue"],
        profile_views={"President A": view},
    )
    compared = compare_payload["presidents"]["President A"]

    assert shared["schema_version"] == "president-profile-v3"
    assert tuple(compared) == (
        "president", "display_name", "slug", "party", "years", "sample",
        "measures", "agenda", "neighbors", "evidence",
    )
    assert compared["president"] == shared["president"]
    assert compared["slug"] == shared["slug"]
    assert compared["years"] == shared["years"]
    assert compared["party"] == shared["party"]
    assert "raw_stats" not in compared
    assert "issue_evidence" not in compared
    assert "signature_speeches" not in compared
    assert compared["sample"]["thin_record"] is True
    assert compared["sample"]["speech_count_label"] == "1 speech"
    assert len(compared["measures"]["corpus"]) == len(profiles.RADAR_AXES)
    assert len(compared["measures"]["ai"]) == 6
    assert all(
        row[2] is None
        and row[3] == "N/A · Insufficient record"
        for layer in ("corpus", "ai")
        for row in compared["measures"][layer]
    )
