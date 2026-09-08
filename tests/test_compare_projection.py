from __future__ import annotations

import csv
import io

import pytest

from presidential_profiles import ai_labels
from presidential_profiles import compare_projection as compare
from presidential_profiles import profiles
from presidential_profiles import profiles_site
from presidential_profiles import speaker_topic_network
from presidential_profiles import topic_quality


@pytest.fixture(scope="module")
def compare_inputs():
    profile_data = profiles.build_profile_data()
    profile_data["ai"] = ai_labels.build_ai_data()
    profile_data["feature_neighbors"] = profiles.feature_neighbors(
        profile_data, profile_data["ai"]
    )
    display_issues = topic_quality.display_issues(profile_data["issue_meta"]["issues"])
    views = profiles_site.build_profile_view_models(
        profile_data, display_issues, require_complete=True
    )
    network = speaker_topic_network.load_network_bundle()
    return network, profile_data, views, display_issues


@pytest.fixture(scope="module")
def projection(compare_inputs):
    return compare.build_projection(*compare_inputs)


def test_projection_has_exact_inventory_and_dense_cardinalities(projection):
    assert projection.agenda_index["schema_version"] == compare.AGENDA_INDEX_SCHEMA
    assert projection.evidence_index["schema_version"] == compare.EVIDENCE_INDEX_SCHEMA
    assert projection.manifest["schema_version"] == compare.MANIFEST_SCHEMA
    assert len(projection.files) == 22
    assert len(projection.topic_shards) == 17
    assert len(projection.manifest["inventory"]) == 21
    assert len(projection.agenda_index["presidents"]) == 45
    assert len(projection.agenda_index["level1_topics"]) == 17
    assert len(projection.agenda_index["broad_cells"]) == 45 * 17
    assert sum(len(shard["topics"]) for shard in projection.topic_shards.values()) == 50
    assert sum(len(shard["cells"]) for shard in projection.topic_shards.values()) == 45 * 50
    assert len(projection.agenda_index["legacy_issues"]) == 16
    assert len(projection.agenda_index["legacy_cells"]) == 45 * 16


def test_ai_values_reconcile_to_plan3_and_explicit_zero_states(
    projection, compare_inputs
):
    network = compare_inputs[0]
    compare.validate_projection(projection, network_bundle=network)
    rows = list(csv.DictReader(io.StringIO(
        projection.files[compare.AI_VALUES_FILE].decode("utf-8")
    )))
    assert len(rows) == compare.EXPECTED_AI_ROWS
    assert len({(row["president_profile_id"], row["topic_id"]) for row in rows}) == len(rows)
    source_keys = set(
        network.edges.loc[
            network.edges["scope_type"].eq("corpus")
            & network.edges["scope_id"].eq("all-corpus"),
            ["president_profile_id", "topic_id"],
        ].itertuples(index=False, name=None)
    )
    for row in rows:
        key = (row["president_profile_id"], row["topic_id"])
        if key not in source_keys:
            assert row["topic_paragraph_count"] == "0"
            assert row["speaker_paragraph_share"] == "0.0"
            assert row["state"] in {"supported_zero", "thin_zero"}


def test_support_states_preserve_thin_presidents_and_edges(projection):
    states = {
        cell["state"]
        for cell in projection.agenda_index["broad_cells"]
    }
    states.update(
        cell["state"]
        for shard in projection.topic_shards.values()
        for cell in shard["cells"]
    )
    assert {"supported", "thin_edge", "thin_president", "supported_zero", "thin_zero"} <= states
    assert "unavailable" not in states
    thin = {
        row["president_profile_id"]
        for row in projection.agenda_index["president_support"]
        if row["president_support_status"] == "thin"
    }
    assert thin == {"james-a-garfield", "william-harrison", "zachary-taylor"}


def test_legacy_values_are_source_document_owner_values(projection, compare_inputs):
    _, profile_data, _, _ = compare_inputs
    catalog = {row["issue_id"]: row for row in projection.agenda_index["legacy_issues"]}
    cells = {
        (row["president_profile_id"], row["issue_id"]): row
        for row in projection.agenda_index["legacy_cells"]
    }
    issue = next(row for row in catalog.values() if row["issue_label"] == "War & military")
    cell = cells[("abraham-lincoln", issue["issue_id"])]
    source = profile_data["issues"].loc["Abraham Lincoln", f"share_{issue['source_issue_key']}"]
    assert cell["paragraph_share"] == source
    assert cell["source_document_paragraph_count"] == int(
        profile_data["issues"].loc["Abraham Lincoln", "n_paragraphs"]
    )
    broad = {
        (row["president_profile_id"], row["topic_id"]): row
        for row in projection.agenda_index["broad_cells"]
    }
    war_topic = next(
        row for row in projection.agenda_index["level1_topics"]
        if row["topic_label"] == "War & Military Affairs"
    )
    assert broad[("abraham-lincoln", war_topic["topic_id"])]["speaker_paragraph_share"] != source


def test_crosswalk_preserves_declared_links_and_unmapped_topics(projection):
    assert tuple(projection.agenda_index["unmapped_level2_labels"]) == compare.KNOWN_UNMAPPED_LEVEL2
    relationships = projection.agenda_index["crosswalk_relationships"]
    assert len(relationships) > 50
    counts: dict[str, int] = {}
    for row in relationships:
        counts[row["topic_id"]] = counts.get(row["topic_id"], 0) + 1
    assert any(count > 1 for count in counts.values())


def test_evidence_uses_only_declared_existing_families(projection):
    assert len(projection.evidence_index["presidents"]) == 45
    lincoln = next(
        row for row in projection.evidence_index["presidents"]
        if row["president_profile_id"] == "abraham-lincoln"
    )
    assert lincoln["footprint"]["source_document_speech_count"] == 15
    assert len(lincoln["adversarial_entities"]) == 3
    assert len(lincoln["presidential_invocations"]) <= 6
    assert len(lincoln["distinctive_vocabulary"]) == 6
    assert len(lincoln["signature_speeches"]) == 3
    assert all(
        len(group["receipts"]) <= 3
        for group in lincoln["adversarial_entities"] + lincoln["presidential_invocations"]
    )
    assert "legacy_invocations" not in lincoln


def test_manifest_hashes_budgets_and_publication_are_exact(projection, compare_inputs, tmp_path):
    for filename, receipt in projection.manifest["inventory"].items():
        value = projection.files[filename]
        assert receipt["bytes"] == len(value)
        assert receipt["sha256"] == compare._sha256_bytes(value)
    compare.write_public_projection(
        projection, tmp_path, network_bundle=compare_inputs[0]
    )
    compare.validate_publication(
        projection, tmp_path, network_bundle=compare_inputs[0]
    )
    assert {path.name for path in (tmp_path / "data" / "compare").iterdir()} == set(projection.files)


def test_two_projection_builds_are_byte_identical(projection, compare_inputs):
    assert compare.build_projection(*compare_inputs).files == projection.files
