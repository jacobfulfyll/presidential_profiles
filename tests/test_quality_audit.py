"""Governed Data Quality calculations, publication, and accessible rendering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import pytest

from presidential_profiles import era_profiles, expansion_site, quality_audit


DATA = Path(__file__).parents[1] / "data"


@pytest.fixture(scope="module")
def quality_bundle() -> quality_audit.QualityAuditBundle:
    # A smaller draw count keeps the focused suite fast. Point estimates and all
    # source/population contracts are independent of the bootstrap draw count.
    return quality_audit.build_quality_audit(DATA, bootstrap_draws=200)


def _rewrite_manifest(directory: Path, manifest: dict) -> None:
    manifest["metadata_sha256"] = quality_audit._metadata_hash(manifest)
    (directory / quality_audit.MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _refresh_artifact_receipt(directory: Path, manifest: dict, filename: str) -> None:
    payload = (directory / filename).read_bytes()
    manifest["artifacts"][filename]["bytes"] = len(payload)
    manifest["artifacts"][filename]["sha256"] = quality_audit._sha256_bytes(payload)


def test_corrected_headlines_are_computed_from_governed_populations(quality_bundle):
    summary = quality_bundle.summary
    assert (summary["paired"], summary["sampled"], summary["missing"]) == (8_570, 9_048, 478)
    assert summary["affected_speeches"] == 28
    assert (summary["entity_matched"], summary["entity_union"]) == (4_147, 8_704)
    assert summary["entity_match_rate"] == pytest.approx(0.47644761029411764)
    assert (summary["entity_stance_matches"], summary["entity_matched"]) == (3_653, 4_147)
    assert summary["entity_stance_agreement"] == pytest.approx(0.8808777429467085)
    assert summary["entity_jaccard_bearing"] == pytest.approx(0.5323722950176439)
    assert summary["entity_jaccard_all"] == pytest.approx(0.7559823691153913)
    assert summary["entity_bearing_paragraphs"] == 4_472
    assert summary["normalized_topic_assignments"] == 52_133
    assert summary["multi_label_paragraphs"] == 14_507
    assert summary["unlabeled_paragraphs"] == 404


def test_population_ledger_exposes_document_owner_and_actual_speaker_layers(quality_bundle):
    summary = quality_bundle.summary
    assert summary["retained_documents"] == 1_053
    assert summary["retained_paragraphs"] == 35_394
    assert summary["eligible_paragraphs"] == 32_531
    assert summary["excluded_paragraphs"] == 2_863
    assert summary["cross_owner_paragraphs"] == 296
    assert summary["speaker_appearances"] == 1_054
    ledger = quality_bundle.tables["population_ledger"].set_index("population_id")
    assert ledger.loc["source_documents", "unit"] == "documents"
    assert ledger.loc["speaker_appearances", "unit"] == "appearances"
    assert ledger.loc["second_model_paired_paragraphs", "count"] == 8_570


def test_all_public_era_tables_follow_approved_story_chronology(quality_bundle):
    expected = [spec.key for spec in era_profiles.ERA_PROFILE_SPECS]
    for table_name in (
        "second_model_by_era",
        "era_coverage",
        "taxonomy_label_behavior",
    ):
        table = quality_bundle.tables[table_name]
        assert table["era_key"].tolist() == expected
        assert table["era_order"].tolist() == list(range(9))
    zero_sum = quality_bundle.tables["zero_sum_by_era"]
    assert zero_sum["era_key"].tolist() == ["overall", *expected]


def test_entity_audit_is_key_complete_over_only_paired_paragraphs(quality_bundle):
    audit = quality_bundle.tables["entity_name_agreement"]
    assert len(audit) == 8_570
    assert not audit.duplicated(quality_audit.PARAGRAPH_KEY).any()
    assert int(audit[["primary_only", "second_only", "matched"]].sum().sum()) == 8_704
    assert int(audit["matched"].sum()) == 4_147
    # Both-empty paragraphs remain explicit and account for the difference
    # between all-paragraph and entity-bearing Jaccard.
    assert int((~audit["entity_bearing"]).sum()) == 4_098


def test_case_studies_are_deterministic_symmetric_and_not_adjudicated(quality_bundle):
    zero_sum = quality_bundle.summary["_charts"]["zero_sum_cases"]
    assert [(row["zero_sum_primary"], row["zero_sum_second"]) for row in zero_sum] == [
        (True, False),
        (False, True),
    ]
    entities = quality_bundle.tables["entity_case_examples"].set_index("case_id")
    assert entities.index.tolist() == [
        "matched_after_normalization",
        "primary_only_detection",
        "second_only_detection",
    ]
    assert entities["case_kind"].tolist() == ["matched", "primary_only", "second_only"]
    assert entities["interpretation_status"].eq(
        "illustrative_cross_model_comparison_not_adjudicated"
    ).all()
    matched = entities.loc["matched_after_normalization"]
    assert matched["entity_primary"] != matched["entity_second"]
    assert matched["entity_normalized"]
    assert not entities.loc["primary_only_detection", "entity_second"]
    assert not entities.loc["second_only_detection", "entity_primary"]


def test_entity_builder_drops_rows_outside_shared_semantic_keys():
    keys = pd.DataFrame([{"doc_name": "paired", "para_idx": 0}])
    primary = pd.DataFrame(
        [
            {"doc_name": "paired", "para_idx": 0, "entity": "  Rival ", "stance": "adversarial"},
            {"doc_name": "outside", "para_idx": 0, "entity": "Noise", "stance": "neutral"},
        ]
    )
    second = pd.DataFrame(
        [{"doc_name": "paired", "para_idx": 0, "entity": "rival", "stance": "adversarial"}]
    )
    audit, matched, by_doc = quality_audit._entity_frames(primary, second, keys)
    assert len(audit) == 1
    assert audit.iloc[0]["matched"] == 1
    assert len(matched) == 1
    assert by_doc.iloc[0].to_dict() == {
        "doc_name": "paired",
        "n_union": 1,
        "n_matched": 1,
        "n_stance_agree": 1,
    }


def test_uncertainty_export_uses_percentage_points_and_separate_grains(quality_bundle):
    table = quality_bundle.tables["uncertainty_decomposition"].set_index("surface")
    assert table.loc["corex_issues", "period_kind"] == "year5"
    assert table.loc["corex_issues", "mean_sampling_width_pp"] == pytest.approx(8.695279, rel=1e-6)
    assert table.loc["llm_topics", "period_kind"] == "era"
    assert table.loc["llm_topics", "mean_sampling_width_pp"] == pytest.approx(2.115224, rel=1e-6)
    assert table.loc["llm_topics", "mean_combined_width_pp"] == pytest.approx(2.822877, rel=1e-6)
    assert (table["mean_sampling_width_pp"] >= 0).all()


def test_uncertainty_unit_and_nonfinite_mutations_fail_closed():
    base = pd.DataFrame(
        [
            {
                "surface": "example",
                "series": "a",
                "period_kind": "era",
                "lo_sampling": 0.1,
                "hi_sampling": 0.2,
                "lo": 0.05,
                "hi": 0.25,
                "ci_components": "sampling+other",
                "disagreement_status": "measured",
                "disagreement_half_width": 0.025,
            }
        ]
    )
    result = quality_audit._build_uncertainty_decomposition(base)
    assert result.iloc[0]["mean_sampling_width_pp"] == pytest.approx(10.0)
    wrong_unit = base.assign(hi_sampling=20.0)
    with pytest.raises(quality_audit.QualityAuditError, match="unit mismatch"):
        quality_audit._build_uncertainty_decomposition(wrong_unit)
    nonfinite = base.assign(hi=float("inf"))
    with pytest.raises(quality_audit.QualityAuditError, match="non-finite"):
        quality_audit._build_uncertainty_decomposition(nonfinite)


def test_agreement_export_has_metric_specific_clustered_intervals(quality_bundle):
    table = quality_bundle.tables["agreement_by_field_and_era"]
    assert list(table.columns) == [
        "field", "metric", "era_key", "era_label", "era_order", "value",
        "ci_low", "ci_high", "n_units", "support_unit", "n_paragraphs",
        "n_speeches", "prevalence_primary", "prevalence_second",
        "interval_status",
    ]
    overall = table[table["era_key"].eq("overall")].set_index(["field", "metric"])
    assert overall.loc[("entities", "name_match_rate"), "value"] == pytest.approx(
        4_147 / 8_704
    )
    assert overall.loc[("entities", "stance_agreement"), "value"] == pytest.approx(
        3_653 / 4_147
    )
    assert overall.loc[("topics", "jaccard_normalized"), "value"] == pytest.approx(
        0.7044511629, rel=1e-6
    )
    assert set(overall["interval_status"]) == {"speech_cluster_bootstrap"}
    assert (overall["ci_low"] <= overall["value"]).all()
    assert (overall["value"] <= overall["ci_high"]).all()
    assert quality_bundle.manifest["agreement_v1_validation"]["status"] == "passed"


def test_duplicate_and_divergent_key_mutations_fail_closed():
    duplicate = pd.DataFrame(
        [
            {"doc_name": "a", "para_idx": 0},
            {"doc_name": "a", "para_idx": 0},
        ]
    )
    with pytest.raises(quality_audit.QualityAuditError, match="duplicate keys"):
        quality_audit._require_unique(duplicate, quality_audit.PARAGRAPH_KEY, "mutant")
    left = pd.DataFrame([{"doc_name": "a", "para_idx": 0}])
    right = pd.DataFrame([{"doc_name": "b", "para_idx": 0}])
    with pytest.raises(quality_audit.QualityAuditError, match="key sets diverge"):
        quality_audit._require_same_keys(left, right, "mutant")


def test_publication_is_hashed_versioned_and_removes_retired_pos_file(
    quality_bundle, tmp_path
):
    retired = tmp_path / "era_part_of_speech.csv"
    retired.write_text("stale\n", encoding="utf-8")
    manifest = quality_audit.write_publication(quality_bundle, tmp_path)
    assert not retired.exists()
    assert manifest["schema_version"] == "data-quality-public-manifest-v2"
    assert manifest["generation_status"] == "deterministic_local_no_api_calls"
    assert manifest["human_validity_status"] == "not_measured"
    assert set(manifest["artifacts"]) == set(quality_audit.PUBLIC_TABLES.values())
    assert set(manifest["table_schemas"]) == set(quality_audit.PUBLIC_TABLES.values())
    assert manifest["table_schemas"]["entity_name_agreement.csv"]["semantic_key"] == [
        "doc_name", "para_idx"
    ]
    for filename, receipt in manifest["artifacts"].items():
        payload = (tmp_path / filename).read_bytes()
        assert receipt["sha256"] == "sha256:" + hashlib.sha256(payload).hexdigest()
        assert receipt["bytes"] == len(payload)
    persisted = json.loads((tmp_path / quality_audit.MANIFEST_NAME).read_text())
    semantic = {key: value for key, value in persisted.items() if key != "metadata_sha256"}
    observed = "sha256:" + hashlib.sha256(
        quality_audit._canonical_json(semantic).encode("utf-8")
    ).hexdigest()
    assert persisted["metadata_sha256"] == observed


def test_publication_validator_rejects_stale_source_and_population_receipts(
    quality_bundle, tmp_path
):
    source_dir = tmp_path / "source-receipt"
    quality_audit.write_publication(quality_bundle, source_dir)
    manifest = json.loads((source_dir / quality_audit.MANIFEST_NAME).read_text())
    first_source = next(iter(manifest["source_artifacts"].values()))
    first_source["sha256"] = "sha256:" + "0" * 64
    _rewrite_manifest(source_dir, manifest)
    with pytest.raises(quality_audit.QualityAuditError, match="source receipt .* stale"):
        quality_audit.validate_publication(source_dir)

    population_dir = tmp_path / "population-receipt"
    quality_audit.write_publication(quality_bundle, population_dir)
    manifest = json.loads((population_dir / quality_audit.MANIFEST_NAME).read_text())
    manifest["populations"]["paired"] -= 1
    _rewrite_manifest(population_dir, manifest)
    with pytest.raises(quality_audit.QualityAuditError, match="headline population regression"):
        quality_audit.validate_publication(population_dir)


def test_publication_validator_rejects_negative_missingness_and_schema_key_mutations(
    quality_bundle, tmp_path
):
    missing_dir = tmp_path / "negative-missingness"
    quality_audit.write_publication(quality_bundle, missing_dir)
    filename = quality_audit.PUBLIC_TABLES["second_model_by_speech"]
    frame = pd.read_csv(missing_dir / filename)
    frame.loc[0, "missing_paragraphs"] = -1
    (missing_dir / filename).write_bytes(quality_audit._csv_bytes(frame))
    manifest = json.loads((missing_dir / quality_audit.MANIFEST_NAME).read_text())
    _refresh_artifact_receipt(missing_dir, manifest, filename)
    _rewrite_manifest(missing_dir, manifest)
    with pytest.raises(quality_audit.QualityAuditError, match="negative missingness"):
        quality_audit.validate_publication(missing_dir)

    key_dir = tmp_path / "missing-key"
    quality_audit.write_publication(quality_bundle, key_dir)
    filename = quality_audit.PUBLIC_TABLES["entity_name_agreement"]
    manifest = json.loads((key_dir / quality_audit.MANIFEST_NAME).read_text())
    manifest["table_schemas"][filename]["semantic_key"] = ["doc_name"]
    _rewrite_manifest(key_dir, manifest)
    with pytest.raises(quality_audit.QualityAuditError, match="semantic key mismatch"):
        quality_audit.validate_publication(key_dir)

    schema_dir = tmp_path / "missing-schema-column"
    quality_audit.write_publication(quality_bundle, schema_dir)
    filename = quality_audit.PUBLIC_TABLES["population_ledger"]
    frame = pd.read_csv(schema_dir / filename).drop(columns="definition")
    (schema_dir / filename).write_bytes(quality_audit._csv_bytes(frame))
    manifest = json.loads((schema_dir / quality_audit.MANIFEST_NAME).read_text())
    manifest["table_schemas"][filename]["columns"] = list(frame.columns)
    _refresh_artifact_receipt(schema_dir, manifest, filename)
    _rewrite_manifest(schema_dir, manifest)
    with pytest.raises(quality_audit.QualityAuditError, match="schema columns mismatch"):
        quality_audit.validate_publication(schema_dir)


def test_quality_page_is_layered_semantic_and_javascript_independent(quality_bundle):
    page = expansion_site._quality_page(dict(quality_bundle.summary))
    assert "Can I trust the data?" in page
    assert "30-second verdict" in page
    assert "Two-minute visual story" in page
    assert "Technical evidence" in page
    assert "primary-only" in page
    assert "second-only" in page
    assert "false-positive" not in page
    assert "human-validated accuracy" in page
    assert "What a disagreement looks like" in page
    assert "From source\ntext to a normalized comparison key" in page
    assert page.count("Illustrative, not adjudicated.") == 2
    assert "data/quality/entity_case_examples.csv" in page
    assert "Grammar change" not in page
    assert "era_part_of_speech.csv" not in page
    assert "Plotly" not in page
    figures = re.findall(r"<figure\b[^>]*data-substantive-chart[^>]*>", page)
    assert len(figures) == 9
    assert all(re.search(r'data-metric="[^"]+"', figure) for figure in figures)
    assert all(re.search(r'data-evidence="[^"]+"', figure) for figure in figures)
    assert all("aria-labelledby=" in figure for figure in figures)
    assert page.count("<table") >= 6
    assert page.count('class="table-scroll"') == 6
    assert page.count('role="region"') >= 6
    assert "Scroll table horizontally →" in page
    assert "#receipts code,.case-fields code{overflow-wrap:anywhere" in page
    for label in (
        "Claim", "Population", "Method", "Uncertainty", "Artifact", "Status",
        "Limitation", "Downstream use",
    ):
        assert f"<dt>{label}</dt>" in page


def test_quality_module_static_no_network_guard():
    quality_audit.assert_no_network_dependencies()
