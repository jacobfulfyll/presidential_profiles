"""Website adapters for the frozen AI annotation layer."""

import pandas as pd
import pytest

from presidential_profiles import ai_labels, compare_site, methodology_site, profiles


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


def test_compare_payload_preserves_every_shared_profile_field(monkeypatch):
    shared = {
        "schema_version": "president-profile-v2",
        "president": "President A", "slug": "president-a", "party": "Test",
        "years": {"first": 1900, "last": 1901},
        "sample": {"n_speeches": 6, "n_words": 1000, "thin_record": False,
                   "warning": None},
        "rhetorical_radar": [{"key": "x"}], "raw_stats": {"x": 1},
        "legacy_issue_attention": [
            {"key": "Issue", "label": "Issue", "share": 10.0,
             "era_relative_difference": 2.0}
        ],
        "issue_evidence": {"cards": []}, "ai": None,
        "distinctive_vocabulary": [{"term": "word"}],
        "signature_speeches": [{"title": "Speech", "url": "https://example.test"}],
        "legacy_invocations": {"invokes": [], "invoked_by": None},
        "classified_invocations": [], "voice_neighbors": [],
        "agenda_neighbors": [], "context_specific": {
            "profile_only": [], "compare_only": [],
        },
    }
    monkeypatch.setattr(
        compare_site.profiles_site, "public_profile_payload",
        lambda president, data, issues: shared,
    )
    row = {
        "party": "Test", "first_year": 1900, "last_year": 1901,
        "n_speeches": 6, "certainty": 1, "hype": 2, "mechanism": 3,
        "nrc_hope": 4, "nrc_fear": 5, "fk_grade": 6,
    }
    for key, _ in profiles.RADAR_AXES:
        row[f"pct_{key}"] = 50
    data = {
        "scores": pd.DataFrame([row], index=["President A"]),
        "voice_neighbors": {}, "agenda_neighbors": {},
    }
    compared = compare_site.build_payload(data, ["Issue"])["President A"]
    for key, value in shared.items():
        assert compared[key] == value
