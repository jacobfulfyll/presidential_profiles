from __future__ import annotations

import csv
import copy
import io
import json
from pathlib import Path

import pytest

from presidential_profiles import metrics
from presidential_profiles import speaker_topic_network as network
from presidential_profiles import summary_topic_network as summary
from presidential_profiles.summary_topic_network_assets import (
    SUMMARY_TOPIC_NETWORK_CSS,
    SUMMARY_TOPIC_NETWORK_JS,
)


@pytest.fixture(scope="module")
def accepted_bundle():
    return network.load_network_bundle()


@pytest.fixture(scope="module")
def projection(accepted_bundle):
    return summary.build_projection(accepted_bundle)


def test_projection_has_locked_all_corpus_population_and_support(projection):
    assert projection.index["schema_version"] == summary.INDEX_SCHEMA
    assert projection.index["scope"] == {
        "scope_type": "corpus",
        "scope_id": "all-corpus",
        "scope_label": "All eligible corpus",
        "topic_level": "level1",
    }
    assert projection.index["edge_width_measure"] == "speaker_paragraph_share"
    assert projection.index["counts"] == {
        "eligible_actual_president_paragraphs": 32_531,
        "presidents": 45,
        "supported_presidents": 42,
        "thin_presidents": 3,
        "level1_topics": 17,
        "observed_level1_edges": 700,
        "supported_level1_edges": 595,
        "default_visible_level1_edges": 427,
    }
    thin = {
        row["president_name"]
        for row in projection.index["president_support"]
        if row["president_support_status"] == "thin"
    }
    assert thin == {"James A. Garfield", "William Harrison", "Zachary Taylor"}


def test_projection_is_column_filtered_but_every_value_matches_plan3(
    projection, accepted_bundle
):
    summary.validate_projection(projection, accepted_bundle)
    source = accepted_bundle.edges.loc[
        accepted_bundle.edges["scope_type"].eq("corpus")
        & accepted_bundle.edges["scope_id"].eq("all-corpus")
        & accepted_bundle.edges["topic_level"].eq("level1")
    ].set_index("edge_id")
    for edge in projection.index["edges"]:
        row = source.loc[edge["edge_id"]]
        assert set(edge) == set(summary.EDGE_FIELDS)
        for field, value in edge.items():
            assert value == (row.name if field == "edge_id" else row[field])
    forbidden = {
        "weight", "confidence", "importance", "influence", "similarity",
        "layout", "x", "y",
    }
    assert not forbidden.intersection(projection.index["edges"][0])


def test_changed_or_missing_projected_rows_fail_closed(projection, accepted_bundle):
    changed_index = copy.deepcopy(projection.index)
    changed_index["edges"][0]["speaker_paragraph_share"] += 0.01
    changed_index["index_sha256"] = summary._self_hash(changed_index, "index_sha256")
    changed_files = dict(projection.files)
    changed_files[summary.INDEX_FILE] = summary._json_bytes(changed_index)
    changed = summary.SummaryTopicNetworkProjection(
        changed_index, projection.shards, projection.manifest, changed_files
    )
    with pytest.raises(summary.SummaryTopicNetworkError, match="rows or values"):
        summary.validate_projection(changed, accepted_bundle)


def test_seventeen_shards_cover_level2_once_and_reconcile_evidence(
    projection, accepted_bundle
):
    assert projection.manifest["counts"] == {
        "files_excluding_manifest": 19,
        "topic_shards": 17,
        "level1_edges": 700,
        "level2_edges": 1_464,
        "level2_topics": 50,
    }
    child_ids = [
        child["topic_id"]
        for shard in projection.shards.values()
        for child in shard["child_topics"]
    ]
    assert len(child_ids) == len(set(child_ids)) == 50
    projected_edge_ids = {
        edge["edge_id"] for shard in projection.shards.values() for edge in shard["edges"]
    }
    assert len(projected_edge_ids) == 1_464
    assert projected_edge_ids == set(
        accepted_bundle.edges.loc[
            accepted_bundle.edges["scope_type"].eq("corpus")
            & accepted_bundle.edges["topic_level"].eq("level2"),
            "edge_id",
        ]
    )
    for shard in projection.shards.values():
        valid = {edge["edge_id"] for edge in shard["edges"]}
        valid.update(
            edge["edge_id"] for edge in projection.index["edges"]
            if edge["topic_id"] == shard["parent_topic"]["topic_id"]
        )
        assert {receipt["edge_id"] for receipt in shard["evidence"]} <= valid
        assert all(
            child["parent_topic_id"] == shard["parent_topic"]["topic_id"]
            for child in shard["child_topics"]
        )


def test_manifest_inventory_hashes_and_byte_budgets_are_exact(
    projection, accepted_bundle
):
    assert len(projection.files) == 20
    assert set(projection.files) == {
        summary.INDEX_FILE, summary.MANIFEST_FILE, summary.RECURRING_TABLE_FILE,
        *projection.shards,
    }
    index_receipt = projection.manifest["files"][summary.INDEX_FILE]
    assert index_receipt["bytes"] <= summary.INDEX_RAW_MAX
    assert index_receipt["gzip_bytes"] <= summary.INDEX_GZIP_MAX
    for path, receipt in projection.manifest["files"].items():
        value = projection.files[path]
        assert receipt["bytes"] == len(value)
        assert receipt["sha256"] == summary._sha256_bytes(value)
        if path.startswith("topics/"):
            assert receipt["bytes"] <= summary.SHARD_RAW_MAX
            assert receipt["gzip_bytes"] <= summary.SHARD_GZIP_MAX

    rows = list(csv.DictReader(io.StringIO(
        projection.files[summary.RECURRING_TABLE_FILE].decode("utf-8")
    )))
    assert len(rows) == 427
    assert list(rows[0]) == list(accepted_bundle.edges.columns)
    assert {row["edge_id"] for row in rows} == {
        edge["edge_id"] for edge in projection.index["edges"]
        if edge["default_visible"] is True
        and edge["president_support_status"] == "supported"
    }
    assert {row["topic_level"] for row in rows} == {"level1"}
    assert {row["default_visible"] for row in rows} == {"True"}


def test_publication_round_trip_rejects_extra_and_changed_files(
    projection, accepted_bundle, tmp_path
):
    summary.write_public_projection(
        accepted_bundle, tmp_path, projection=projection
    )
    summary.write_renderer_assets(tmp_path)
    assert summary.validate_publication(accepted_bundle, tmp_path)["files"] == 20
    output = tmp_path / "data" / summary.PUBLIC_DIR_NAME
    (output / "unexpected.json").write_text("{}\n")
    with pytest.raises(summary.SummaryTopicNetworkError, match="inventory"):
        summary.validate_publication(accepted_bundle, tmp_path)
    (output / "unexpected.json").unlink()
    index_path = output / summary.INDEX_FILE
    payload = json.loads(index_path.read_text())
    payload["counts"]["presidents"] = 44
    index_path.write_bytes(summary._json_bytes(payload))
    with pytest.raises(summary.SummaryTopicNetworkError):
        summary.validate_publication(accepted_bundle, tmp_path)


def test_two_in_memory_builds_are_byte_identical(accepted_bundle, projection):
    assert summary.build_projection(accepted_bundle).files == projection.files


def test_server_shell_keeps_topic_controls_and_only_the_exact_csv_download(projection):
    page = summary.render_summary_section(projection)
    assert 'id="summary-topics"' in page
    assert "05 · Topics" in page
    assert "Presidents and their recurring topics" in page
    assert "Which broad topics recur" not in page  # the question lives in JSON, not duplicate prose
    assert 'href="#summary-topic-group-' not in page
    assert page.count('data-summary-topic-choice="') == 17
    assert "Choose up to eight broad topics" in page
    assert ">Choose Topics</h3>" in page
    assert "Build a topic field" not in page
    assert 'class="summary-topic-control-heading"' in page
    assert "regardless of the order you click them" not in page
    assert 'data-node-size-measure="selected_topic_paragraph_memberships"' in page
    assert "President topic-gravity field" in page
    assert "One paragraph can count under more than one topic" in page
    assert 'role="tablist"' not in page
    assert page.count('<option value="">') == 1
    assert page.count('<option value="') == 43
    assert 'data-summary-topic-president-filter' in page
    assert 'data-summary-topic-president-row' in page
    assert "Compare presidents" in page
    assert "Recurring relationships only" not in page
    assert "Choose up to eight supported presidents" not in page
    assert "Choose up to eight topics. No presidents are drawn yet." not in page
    assert "Include lower-support relationships" not in page
    assert "Exact relationship readout" not in page
    assert "deterministic audit examples" not in page.lower()
    assert "Where these topics come from" not in page
    assert "nine era-isolated AI proposals" not in page
    assert 'data-summary-topic-stage-readout' in page
    assert 'data-summary-topic-size-key' in page
    assert "<noscript>" in page
    assert "exact recurring-topic table" in page.lower()
    assert "How to read this topic-gravity field" not in page
    assert 'class="summary-topic-how"' not in page
    assert 'class="summary-topic-exact"' not in page
    assert "<table" not in page
    assert "importance, intent, influence, or policy success" in page
    assert page.count("<a ") == 1
    assert (
        f'data/{summary.PUBLIC_DIR_NAME}/{summary.RECURRING_TABLE_FILE}' in page
    )
    for removed_link in (
        "data/speaker-topic-network/network_v1.json",
        "data/speaker-topic-network/edges_v1.csv",
        "data-quality.html",
        "methodology.html",
        "feedback.html",
    ):
        assert removed_link not in page


def test_renderer_state_machine_accessibility_and_loading_contracts():
    js = SUMMARY_TOPIC_NETWORK_JS
    for state_field in (
        "selectedTopicIds", "selectedPresidentIds", "previewPresidentId",
        "pinnedPresidentId", "previewTopicId",
    ):
        assert f"{state_field}:" in js
    for removed_field in (
        "detailLevel", "includeThin", "previewEdgeId", "pinnedEdgeId",
        "evidenceEdgeId", "returnFocusElement",
    ):
        assert removed_field not in js
    assert "const MAX_TOPICS = 8" in js
    assert "const MAX_PRESIDENTS = 8" in js
    assert "IntersectionObserver" in js and 'rootMargin: "600px 0px"' in js
    assert "loadShard" not in js and "shardCache" not in js
    assert "anchorPositions(topics, width, height)" in js
    assert "gravityLayout(aggregates, topics, mobile, sizeDomainMax)" in js
    assert "semanticX" in js and "semanticY" in js
    assert "Number(edge.speaker_paragraph_share)" in js
    assert "row.memberships += Number(edge.topic_paragraph_count)" in js
    assert "Math.sqrt(aggregate.memberships / maxMemberships)" in js
    assert "selected-topic paragraph memberships" in js
    assert "aggregates.slice(0, mobile ? 18 : 45)" in js
    assert 'event.key !== "Enter" && event.key !== " "' in js
    assert "edge.default_visible === true" in js
    assert 'edge.president_support_status === "supported"' in js
    assert "Choose up to eight topics. No presidents are drawn yet." in js
    assert "for (let i = 0; i < 190; i += 1)" in js
    assert "presidentAggregates(applyPresidentFilter = true)" in js
    assert "renderPresidentFilter()" in js
    assert "addPresidentFilter(presidentId)" in js
    assert "removePresidentFilter(presidentId)" in js
    assert "updateStageReadout()" in js
    assert "updateSizeKey()" in js
    assert "index.level1_topics.filter(topic => selected.has(topic.topic_id))" in js
    assert "history.pushState" not in js and "history.replaceState" not in js
    assert ".innerHTML" not in js and "innerHTML" not in js
    assert "textContent" in js and "replaceChildren" in js
    assert "1.5 + 5.5" in js and "speaker_paragraph_share" in js
    assert len((js.strip() + "\n").encode()) <= summary.RENDERER_JS_RAW_MAX


def test_scoped_css_has_mobile_touch_overflow_and_reduced_motion_contracts():
    css = SUMMARY_TOPIC_NETWORK_CSS
    assert "@media(max-width:760px)" in css
    assert "min-height:44px" in css
    assert "overflow:clip" in css
    assert "overflow-x:auto" in css
    assert "@media(prefers-reduced-motion:reduce)" in css
    assert ".reduced-motion .summary-topic-section" in css
    assert ".summary-topic-president:focus .president-focus" in css
    assert ".summary-topic-anchor:focus .anchor-focus" in css
    assert ".summary-topic-edge.is-active" in css
    assert ".summary-topic-president-filter" in css
    assert ".summary-topic-president-chips" in css
    assert ".summary-topic-stage-tools" in css
    assert ".summary-topic-browser{display:grid;grid-template-columns:minmax(0,1fr)" in css
    assert ".summary-topic-picker{display:flex;flex-wrap:nowrap" in css
    assert "overflow-x:auto" in css
    assert ".summary-topic-control-heading" in css
    assert ".summary-topic-president-row" in css


def test_chart_registry_maps_only_primary_edge_width_measure():
    assert metrics.CHART_METRICS["summary_topic_relationships"] == {
        "actual_speaker_topic_presence"
    }
    assert "network_edge" not in metrics.CHART_METRICS["summary_topic_relationships"]


def test_generated_summary_replaces_audit_and_ends_after_topics():
    page = (Path(__file__).parents[1] / "docs" / "summary.html").read_text()
    assert 'href="#summary-topics">Topics by president</a>' in page
    assert 'id="summary-topics"' in page
    assert "Presidents and their recurring topics" in page
    assert 'id="summary-audit"' not in page
    assert "What survives" not in page
    assert "synthesis-matrix" not in page
    assert f'assets/{summary.ASSET_JS_NAME}' in page
    assert f'assets/{summary.ASSET_CSS_NAME}' in page
    for label in ("01 · Voice", "02 · Conflict", "03 · Time", "04 · Emotional register"):
        assert label in page
    assert "Open the extreme-speeches evidence cards" not in page
    assert 'id="records_appendix"' not in page
    assert ".summary-document .summary-topic-section { margin-bottom:0; }" in page
    assert ".summary-document footer { margin-top:8px; }" in page
