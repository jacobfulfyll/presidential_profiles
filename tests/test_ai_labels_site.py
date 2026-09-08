"""Website adapters for the frozen AI annotation layer."""

import json
import re

import pandas as pd
import pytest

from presidential_profiles import (
    ai_labels,
    compare_site,
    grammar,
    methodology_site,
    profiles,
    profiles_site,
    quality_audit,
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


def test_methodology_is_layered_semantic_article_with_no_js_fallbacks(tiny_ai_data):
    page = methodology_site.render_methodology(tiny_ai_data)
    assert "From speech to auditable paragraph judgment" in page
    assert 'data-metric="annotation_pipeline_lineage"' in page
    assert 'href="metrics.html#annotation_pipeline_lineage"' in page
    assert "The 30-second trust verdict" in page
    assert "One paragraph, end to end" in page
    assert "Structured output" in page
    assert "Normalization" in page
    assert "Semantic key + receipt" in page
    assert "Downstream chart" in page
    assert "Actual request unit" in page
    assert "Several consecutive paragraphs from one speech" in page
    assert "The population passport" in page
    assert "35,825 primary-LLM topic-bearing paragraphs from 36,229 source paragraphs" in page
    assert "404 no-topic paragraphs excluded by the preregistered estimand" in page
    assert "Separate from the 8,570-paragraph second-model reproducibility audit" in page
    assert "Persisted paired second-model sample only" not in page
    assert page.count("data-substantive-chart") >= 7
    assert page.count("<table") >= 3
    assert page.count('class="table-scroll" role="region"') >= 5
    assert page.count('tabindex="0"') >= 5
    assert "Scroll table horizontally →" in page
    assert "scroll-margin-top:190px" in page
    assert "<svg" not in page
    assert "<script" not in page
    assert "complete frozen taxonomy" in page.lower()
    assert "Alpha Topic" in page
    assert "cross-model reproducibility" in page.lower()
    assert "human validity" in page.lower()
    for href in (
        "index.html",
        "summary.html",
        "presidents/index.html",
        "compare.html",
        "issues/index.html",
        "explorer.html",
        "era-boundaries.html",
        "data-quality.html",
        "label-models.html",
        "methodology.html",
        "data/validation-protocols-v1/manifest_v1.json",
    ):
        assert f'href="{href}"' in page
    figures = re.findall(r"<figure\b[^>]*data-substantive-chart[^>]*>", page)
    assert figures
    assert page.count("<figure") == len(figures)
    assert all('data-metric="' in figure for figure in figures)
    assert all('data-evidence="' in figure for figure in figures)
    assert all('aria-labelledby="' in figure for figure in figures)
    ids = re.findall(r'\bid="([^"]+)"', page)
    assert len(ids) == len(set(ids))
    for label in (
        "Claim", "Population", "Method", "Uncertainty", "Artifact", "Status",
        "Limitation", "Downstream use",
    ):
        assert f"<dt>{label}</dt>" in page
    assert page.count('<article class="receipt">') == 6
    for label in (
        "Inputs", "Outputs", "Schema / hash identity", "Command", "Relevant tests"
    ):
        assert page.count(f"<dt>{label}</dt>") == 6
    assert "sha256:" in page


def test_methodology_separates_agreement_metrics_and_unavailable_intervals(
    tiny_ai_data,
):
    rows = pd.DataFrame([
        {"field": "topics", "metric": "jaccard", "era_bin": -1,
         "era_label": "overall", "value": .68, "n": 12},
        {"field": "party_attack", "metric": "cohen_kappa", "era_bin": -1,
         "era_label": "overall", "value": .71, "n": 12},
        {"field": "proposal_values", "metric": "exact_match", "era_bin": -1,
         "era_label": "overall", "value": .72, "n": 12},
    ])
    page = methodology_site.render_methodology(
        tiny_ai_data,
        agreement_evidence=rows,
        agreement_source="synthetic agreement",
    )
    assert "One audit, several non-interchangeable measures" in page
    assert "κ 0.71" in page
    assert "68.0%" in page
    assert "Interval unavailable in the current frozen evidence" in page
    assert "Metric families are separated" in page


def test_methodology_accepts_story_era_clustered_intervals(tiny_ai_data):
    rows = pd.DataFrame([
        {"field": "topics", "metric": "jaccard_normalized",
         "era_key": "overall", "era_label": "overall", "era_order": -1,
         "value": .68, "ci_low": .64, "ci_high": .72, "n_units": 12,
         "support_unit": "paragraph", "n_paragraphs": 12, "n_speeches": 2,
         "prevalence_primary": None, "prevalence_second": None,
         "interval_status": "speech_cluster_bootstrap"},
        {"field": "topics", "metric": "jaccard_normalized",
         "era_key": "founding", "era_label": "Establishing the republic",
         "era_order": 0, "value": .70, "ci_low": .61, "ci_high": .78,
         "n_units": 6, "support_unit": "paragraph", "n_paragraphs": 6,
         "n_speeches": 1, "prevalence_primary": None,
         "prevalence_second": None,
         "interval_status": "speech_cluster_bootstrap"},
    ])
    page = methodology_site.render_methodology(
        tiny_ai_data,
        agreement_evidence=rows,
        agreement_source="data/quality/agreement_by_field_and_era.csv",
    )
    assert 'data-metric="intermodel_agreement_by_story_era"' in page
    assert "governed Story eras" in page
    assert "95% speech-clustered interval 64.0%–72.0%" in page
    assert "Establishing the republic" in page
    assert "era × field heatmap" in page
    assert 'class="agreement-cell"' in page


def _write_quality_agreement_projection(directory):
    table = pd.DataFrame([
        {
            "field": "topics",
            "metric": "jaccard_normalized",
            "era_key": "overall",
            "era_label": "overall",
            "era_order": -1,
            "value": 0.68,
            "ci_low": 0.64,
            "ci_high": 0.72,
            "n_units": 12,
            "support_unit": "paragraph",
            "n_paragraphs": 12,
            "n_speeches": 2,
            "prevalence_primary": None,
            "prevalence_second": None,
            "interval_status": "speech_cluster_bootstrap",
        }
    ], columns=methodology_site.QUALITY_AGREEMENT_COLUMNS)
    csv_path = directory / quality_audit.PUBLIC_TABLES["agreement_by_field_and_era"]
    payload = table.to_csv(index=False, lineterminator="\n").encode("utf-8")
    csv_path.write_bytes(payload)
    manifest = {
        "schema_version": quality_audit.MANIFEST_SCHEMA,
        "contract_version": quality_audit.CONTRACT_VERSION,
        "generation_status": "deterministic_local_no_api_calls",
        "units": {"agreement": "proportion_0_to_1"},
        "validations": {"test_fixture": "passed"},
        "table_schemas": {
            csv_path.name: {
                "columns": list(table.columns),
                "dtypes": {column: str(table[column].dtype) for column in table},
                "semantic_key": quality_audit.PUBLIC_KEYS[
                    "agreement_by_field_and_era"
                ],
            }
        },
        "artifacts": {
            csv_path.name: {
                "schema_version": quality_audit.CONTRACT_VERSION,
                "rows": len(table),
                "bytes": len(payload),
                "sha256": methodology_site._sha256_bytes(payload),
            }
        },
    }
    manifest["metadata_sha256"] = methodology_site._manifest_metadata_hash(manifest)
    (directory / quality_audit.MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return csv_path, payload


def test_methodology_reads_quality_projection_only_with_governed_manifest(tmp_path):
    csv_path, _ = _write_quality_agreement_projection(tmp_path)
    table, source = methodology_site.load_agreement_evidence(csv_path)
    assert source == str(csv_path)
    assert table.loc[0, "value"] == pytest.approx(0.68)
    assert table.loc[0, "_era_scheme"] == "story"


def test_methodology_rejects_missing_or_tampered_quality_receipts(tmp_path):
    csv_path, payload = _write_quality_agreement_projection(tmp_path)
    (tmp_path / quality_audit.MANIFEST_NAME).unlink()
    with pytest.raises(ValueError, match="manifest is missing"):
        methodology_site.load_agreement_evidence(csv_path)

    csv_path, payload = _write_quality_agreement_projection(tmp_path)
    csv_path.write_bytes(payload + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        methodology_site.load_agreement_evidence(csv_path)


def test_methodology_publishes_small_grammar_receipts_and_compatibility_alias(
    tmp_path,
):
    targets = methodology_site.publish_grammar_artifacts(tmp_path)
    assert targets["era_csv"].read_bytes() == grammar.ERA_POS_PATH.read_bytes()
    assert targets["manifest"].read_bytes() == grammar.MANIFEST_PATH.read_bytes()
    assert (
        targets["compatibility_csv"].read_bytes()
        == targets["era_csv"].read_bytes()
    )
    assert not list(tmp_path.rglob("*.parquet"))


def test_population_contract_rejects_negative_missingness_and_wrong_totals():
    base = {
        "canonical_paragraphs": 10,
        "eligible_paragraphs": 8,
        "excluded_paragraphs": 2,
        "cross_owner_paragraphs": 1,
        "sample_expected": 5,
        "sample_paired": 4,
    }
    methodology_site._validate_population_data(base)
    with pytest.raises(ValueError, match="exceeds its expected population"):
        methodology_site._validate_population_data({**base, "sample_paired": 6})
    with pytest.raises(ValueError, match="do not reconcile"):
        methodology_site._validate_population_data({**base, "excluded_paragraphs": 3})
    with pytest.raises(ValueError, match="must be positive"):
        methodology_site._validate_population_data(
            {**base, "model_comparison_paragraphs": 0}
        )
    with pytest.raises(ValueError, match="exclusion population does not reconcile"):
        methodology_site._validate_population_data({
            **base,
            "model_comparison_paragraphs": 8,
            "model_comparison_excluded_no_topic": 1,
            "model_comparison_source_paragraphs": 10,
        })


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
    compared = compare_payload["presidents"]["president-a"]

    assert shared["schema_version"] == "president-profile-v3"
    assert tuple(compared) == (
        "president_id", "president", "display_name", "slug", "party", "years",
        "source_document_speech_count", "source_document_support_state",
        "actual_speaker_appearance_count", "actual_speaker_support_state",
        "measures", "profile_url", "profile_data_url", "portrait_url",
        "short_name",
    )
    assert compared["short_name"] == "A"
    assert compared["president"] == shared["president"]
    assert compared["slug"] == shared["slug"]
    assert compared["years"] == shared["years"]
    assert compared["party"] == shared["party"]
    assert "raw_stats" not in compared
    assert "issue_evidence" not in compared
    assert "signature_speeches" not in compared
    assert compared["source_document_support_state"] == "thin"
    assert compared["actual_speaker_support_state"] == "unavailable"
    assert compared["source_document_speech_count"] == 1
    assert len(compared["measures"]["corpus"]) == len(profiles.RADAR_AXES)
    assert len(compared["measures"]["ai"]) == 6
    assert all(
        row["percentile"] is None
        and row["rank_display"] == "N/A · Insufficient record"
        for layer in ("corpus", "ai")
        for row in compared["measures"][layer]
    )
