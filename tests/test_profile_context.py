"""Focused contract tests for the separate profile-context-v1 projection."""

from __future__ import annotations

from collections import Counter
import copy
import csv
import html
import io
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from presidential_profiles import (
    actual_speaker_invocation_network as invocation_network,
    profile_context,
    speaker_topic_network,
    story_foundation,
)


@pytest.fixture(scope="module")
def context_inputs():
    foundation = story_foundation.load_story_foundation()
    topics = speaker_topic_network.load_network_bundle(foundation=foundation)
    invocations = invocation_network.build_network_bundle(foundation, topics)
    views = {}
    for path in sorted((profile_context.REPO_ROOT / "docs/data/presidents").glob("*.json")):
        view = json.loads(path.read_text(encoding="utf-8"))
        views[view["president"]] = view
    assert len(views) == 45
    return topics, invocations, foundation, views


@pytest.fixture(scope="module")
def context_projection(context_inputs):
    return profile_context.build_projection(*context_inputs)


def _clone_projection(projection):
    return profile_context.ProfileContextProjection(
        index=copy.deepcopy(projection.index),
        shards={
            relative: copy.deepcopy(shard)
            for relative, shard in projection.shards.items()
        },
        manifest=copy.deepcopy(projection.manifest),
        files=dict(projection.files),
    )


def _reseal_manifest(projection):
    projection.manifest["manifest_sha256"] = profile_context._self_hash(
        projection.manifest, "manifest_sha256"
    )
    projection.files[profile_context.MANIFEST_FILE] = profile_context._json_bytes(
        projection.manifest
    )


def _reseal_index(projection):
    projection.index["index_sha256"] = profile_context._self_hash(
        projection.index, "index_sha256"
    )
    value = profile_context._json_bytes(projection.index)
    projection.files[profile_context.INDEX_FILE] = value
    old = projection.manifest["files"][profile_context.INDEX_FILE]
    projection.manifest["files"][profile_context.INDEX_FILE] = (
        profile_context._file_receipt(
            value, old["schema_version"], old["rows"]
        )
    )
    _reseal_manifest(projection)


def _reseal_shard(projection, relative):
    shard = projection.shards[relative]
    shard["shard_sha256"] = profile_context._self_hash(shard, "shard_sha256")
    value = profile_context._json_bytes(shard)
    projection.files[relative] = value
    index_receipt = next(
        row for row in projection.index["president_shards"]
        if row["filename"] == relative
    )
    index_receipt["sha256"] = profile_context._sha256_bytes(value)
    projection.index["index_sha256"] = profile_context._self_hash(
        projection.index, "index_sha256"
    )
    index_value = profile_context._json_bytes(projection.index)
    projection.files[profile_context.INDEX_FILE] = index_value
    for path, file_value in (
        (relative, value),
        (profile_context.INDEX_FILE, index_value),
    ):
        old = projection.manifest["files"][path]
        projection.manifest["files"][path] = profile_context._file_receipt(
            file_value, old["schema_version"], old["rows"]
        )
    _reseal_manifest(projection)


@pytest.fixture
def fast_validator(monkeypatch, context_inputs, context_projection):
    """Keep mutation tests focused on projection validation, not source I/O."""
    identity = copy.deepcopy(context_projection.index["source_identity"])
    legacy = {
        shard["president"]["president_name"]: copy.deepcopy(
            shard["legacy_issue_evidence"]
        )
        for shard in context_projection.shards.values()
    }
    monkeypatch.setattr(
        profile_context, "_source_identity", lambda *args: copy.deepcopy(identity)
    )
    monkeypatch.setattr(
        profile_context, "_legacy_evidence_by_president",
        lambda *args: copy.deepcopy(legacy),
    )
    monkeypatch.setattr(story_foundation, "check_story_contract", lambda *args: None)
    monkeypatch.setattr(speaker_topic_network, "validate_bundle", lambda *args: None)
    monkeypatch.setattr(invocation_network, "validate_bundle", lambda *args: None)

    def validate(projection):
        profile_context.validate_projection(projection, *context_inputs)

    return validate


def test_exact_inventory_contracts_and_budgets(context_projection):
    projection = context_projection
    assert projection.index["contract_version"] == profile_context.CONTRACT_VERSION
    assert projection.manifest["contract_version"] == profile_context.CONTRACT_VERSION
    assert len(projection.files) == 49
    assert len(projection.shards) == 45
    assert set(projection.files) == {
        profile_context.INDEX_FILE,
        profile_context.MANIFEST_FILE,
        profile_context.INVOCATION_EDGE_CSV,
        profile_context.INVOCATION_EVIDENCE_CSV,
        *projection.shards,
    }
    assert all(
        shard["contract_version"] == profile_context.CONTRACT_VERSION
        for shard in projection.shards.values()
    )
    assert all(
        shard["source_identity"] == projection.index["source_identity"]
        for shard in projection.shards.values()
    )
    index_value = projection.files[profile_context.INDEX_FILE]
    assert len(index_value) <= profile_context.INDEX_RAW_MAX
    assert profile_context._gzip_size(index_value) <= profile_context.INDEX_GZIP_MAX
    shard_values = [
        value for name, value in projection.files.items()
        if name.startswith("presidents/")
    ]
    assert max(map(len, shard_values)) <= profile_context.SHARD_RAW_MAX
    assert max(map(profile_context._gzip_size, shard_values)) <= profile_context.SHARD_GZIP_MAX
    assert sum(map(len, shard_values)) <= profile_context.ALL_SHARDS_RAW_MAX
    assert sum(map(profile_context._gzip_size, shard_values)) <= profile_context.ALL_SHARDS_GZIP_MAX
    edge_csv = projection.files[profile_context.INVOCATION_EDGE_CSV]
    evidence_csv = projection.files[profile_context.INVOCATION_EVIDENCE_CSV]
    assert len(edge_csv) <= profile_context.INVOCATION_EDGE_CSV_RAW_MAX
    assert profile_context._gzip_size(edge_csv) <= profile_context.INVOCATION_EDGE_CSV_GZIP_MAX
    assert len(evidence_csv) <= profile_context.INVOCATION_EVIDENCE_CSV_RAW_MAX
    assert (
        profile_context._gzip_size(evidence_csv)
        <= profile_context.INVOCATION_EVIDENCE_CSV_GZIP_MAX
    )


def test_exact_network_records_and_cross_shard_parity(context_projection):
    projection = context_projection
    index = projection.index
    assert len(index["presidents"]) == 45
    assert len(index["level1_topics"]) == 17
    assert len(index["topic_edges"]) == 595
    assert len(index["invocation_edges"]) == 303
    assert set(index["topic_edges"][0]) == set(profile_context.TOPIC_EDGE_FIELDS)
    assert set(index["invocation_edges"][0]) == set(invocation_network.EDGE_FIELDS)

    topic_edges = [
        row for shard in projection.shards.values() for row in shard["topic_edges"]
    ]
    topic_receipts = [
        row for shard in projection.shards.values() for row in shard["topic_receipts"]
    ]
    assert len(topic_edges) == len({row["edge_id"] for row in topic_edges}) == 700
    assert len(topic_receipts) == len({row["receipt_id"] for row in topic_receipts}) == 1_945
    assert set(topic_edges[0]) == set(profile_context.TOPIC_EDGE_FIELDS)
    assert set(topic_receipts[0]) == set(profile_context.TOPIC_RECEIPT_FIELDS)
    assert all(
        row["source_url"].startswith(invocation_network.MILLER_PREFIX)
        for row in topic_receipts
    )

    edge_refs = Counter()
    evidence_refs = Counter()
    for shard in projection.shards.values():
        relationships = shard["invocations"]
        for edge_id in (
            relationships["outgoing_edge_ids"] + relationships["incoming_edge_ids"]
        ):
            edge_refs[edge_id] += 1
            rows = relationships["receipts_by_edge"][edge_id]
            positions = relationships["selected_receipt_positions_by_edge"][edge_id]
            assert positions == sorted(set(positions))
            assert 1 <= len(positions) <= 3
            assert profile_context.invocation_receipt_examples(shard, edge_id) == [
                rows[position] for position in positions
            ]
            evidence_refs.update(row["candidate_id"] for row in rows)
    assert len(edge_refs) == 303 and set(edge_refs.values()) == {2}
    assert len(evidence_refs) == 1_243 and set(evidence_refs.values()) == {2}


def test_topic_selections_are_precomputed_supported_and_thin_safe(context_projection):
    projection = context_projection
    supported = {
        edge["edge_id"] for edge in projection.index["topic_edges"]
    }
    fdr = projection.shards["presidents/franklin-d-roosevelt_v1.json"]
    default = fdr["topic_selections"]["default"]
    expanded = fdr["topic_selections"]["expanded"]
    assert default["edge_ids"]
    assert set(default["edge_ids"]).issubset(supported)
    assert set(expanded["edge_ids"]).issubset(supported)
    assert len(default["mobile_topic_peer_node_ids"]) <= 6
    assert default["node_ids"][0] == "franklin-d-roosevelt"

    for slug in ("william-harrison", "zachary-taylor", "james-a-garfield"):
        shard = projection.shards[f"presidents/{slug}_v1.json"]
        assert shard["president_support"]["president_support_status"] == "thin"
        assert shard["topic_selections"]["default"]["edge_ids"] == []
        assert shard["topic_selections"]["expanded"]["edge_ids"] == []
        observed = shard["topic_selections"]["observed_thin"]
        assert observed["comparison_allowed"] is False
        assert observed["edge_style"] == "dashed"
        assert len(observed["edge_ids"]) <= 5


def test_legacy_cards_preserve_v3_and_add_keyed_audit_receipts(
    context_inputs, context_projection,
):
    *_, views = context_inputs
    cards = []
    for shard in context_projection.shards.values():
        legacy = shard["legacy_issue_evidence"]
        assert legacy["schema_version"] == profile_context.LEGACY_EVIDENCE_SCHEMA
        assert legacy["population"] == "Source documents assigned to the president."
        president = shard["president"]["president_name"]
        assert [card["legacy_v3"] for card in legacy["cards"]] == views[president][
            "issue_evidence"
        ]["cards"]
        cards.extend(legacy["cards"])
    assert len(cards) == 163
    quoted = [card for card in cards if card["receipts"]]
    rate_only = [card for card in cards if not card["receipts"]]
    assert len(quoted) == 161
    assert len(rate_only) == 2
    for card in quoted:
        receipt = card["receipts"][0]
        assert receipt["selection_role"] == "primary"
        assert html.escape(receipt["excerpt"]) == card["legacy_v3"]["quote"]
        assert receipt["source_url"].startswith(invocation_network.MILLER_PREFIX)
        assert receipt["speaker_eligibility_state"] in {
            "eligible", "excluded", "not_in_accepted_speaker_overlay",
        }
        assert 1 <= len(card["receipts"]) <= 3
        assert len({row["doc_name"] for row in card["receipts"]}) == len(
            card["receipts"]
        )
        assert card["limitation"] == profile_context.LEGACY_LIMITATION
        assert card["why_shown"]["thresholds_passed"]
    for card in rate_only:
        assert card["no_qualifying_excerpt_selected"] is True
        assert card["no_qualifying_excerpt_copy"]


def test_invocation_downloads_are_exact_and_keep_late_trump_biden_rows(
    context_projection,
):
    projection = context_projection
    edge_rows = list(csv.DictReader(io.StringIO(
        projection.files[profile_context.INVOCATION_EDGE_CSV].decode("utf-8")
    )))
    evidence_rows = list(csv.DictReader(io.StringIO(
        projection.files[profile_context.INVOCATION_EVIDENCE_CSV].decode("utf-8")
    )))
    assert len(edge_rows) == 303
    assert len(evidence_rows) == 1_243
    assert list(edge_rows[0]) == list(invocation_network.EDGE_FIELDS)
    assert list(evidence_rows[0]) == list(invocation_network.EVIDENCE_FIELDS)
    late = [
        row for row in evidence_rows
        if row["source_president_name"] == "Donald Trump"
        and row["target_president_name"] == "Joe Biden"
        and row["speech_date"] >= "2025-01-20"
    ]
    assert late
    assert all(row["target_status"] == "former_president" for row in late)
    assert "independent legal-term calendar" in projection.index["policies"][
        "invocation_target_status_limitation"
    ]


def test_complete_projection_is_byte_deterministic(context_inputs, context_projection):
    repeat = profile_context.build_projection(*context_inputs)
    assert context_projection.files == repeat.files


def test_publication_scoped_swap_cleans_stale_and_rolls_back(
    tmp_path, monkeypatch, context_inputs, context_projection,
):
    topics, invocations, foundation, views = context_inputs
    output = tmp_path / "site/data/profile-context"
    output.mkdir(parents=True)
    (output / "stale.txt").write_text("remove only inside owned output", encoding="utf-8")
    outside = tmp_path / "site/keep.txt"
    outside.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        profile_context, "build_projection", lambda *args, **kwargs: context_projection
    )
    written = profile_context.write_public_projection(
        topics, invocations, foundation, views, tmp_path / "site",
        projection=context_projection,
    )
    assert len(written) == 49
    assert not (output / "stale.txt").exists()
    assert outside.read_text(encoding="utf-8") == "keep"
    before = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*") if path.is_file()
    }

    original_validate = profile_context.validate_publication
    calls = 0

    def fail_after_candidate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ProfileContextErrorForTest("post-swap failure")
        return original_validate(*args, **kwargs)

    class ProfileContextErrorForTest(RuntimeError):
        pass

    monkeypatch.setattr(profile_context, "validate_publication", fail_after_candidate)
    with pytest.raises(ProfileContextErrorForTest):
        profile_context.write_public_projection(
            topics, invocations, foundation, views, tmp_path / "site",
            projection=context_projection,
        )
    after = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*") if path.is_file()
    }
    assert after == before


def test_source_hash_and_unsafe_url_refuse_before_projection(
    monkeypatch, context_inputs,
):
    topics, invocations, _, views = context_inputs
    altered_topics = copy.deepcopy(topics)
    altered_topics.topic_nodes.loc[
        altered_topics.topic_nodes.index[0], "topic_definition"
    ] += " Altered."
    with pytest.raises(
        profile_context.ProfileContextError,
        match="loaded speaker-topic bundle differs from disk",
    ):
        profile_context._rehash_governed_sources(altered_topics, invocations)
    monkeypatch.setattr(profile_context, "_sha256_file", lambda path: "sha256:stale")
    with pytest.raises(profile_context.ProfileContextError, match="hash is stale"):
        profile_context._source_identity(topics, invocations, views)
    with pytest.raises(profile_context.ProfileContextError, match="unsafe topic"):
        broken = topics.edge_evidence.head(1).copy()
        broken["source_url"] = "javascript:alert(1)"
        profile_context._safe_topic_receipts(broken)


@pytest.mark.parametrize(
    ("constant", "measurement"),
    (
        ("INDEX_RAW_MAX", "index_raw"),
        ("INDEX_GZIP_MAX", "index_gzip"),
        ("SHARD_RAW_MAX", "shard_raw"),
        ("SHARD_GZIP_MAX", "shard_gzip"),
        ("ALL_SHARDS_RAW_MAX", "all_shards_raw"),
        ("ALL_SHARDS_GZIP_MAX", "all_shards_gzip"),
        ("INVOCATION_EDGE_CSV_RAW_MAX", "edge_csv_raw"),
        ("INVOCATION_EDGE_CSV_GZIP_MAX", "edge_csv_gzip"),
        ("INVOCATION_EVIDENCE_CSV_RAW_MAX", "evidence_csv_raw"),
        ("INVOCATION_EVIDENCE_CSV_GZIP_MAX", "evidence_csv_gzip"),
    ),
)
def test_every_raw_and_gzip_budget_is_enforced(
    monkeypatch, context_projection, constant, measurement,
):
    projection = context_projection
    index_value = projection.files[profile_context.INDEX_FILE]
    shard_values = [projection.files[path] for path in projection.shards]
    edge_csv = projection.files[profile_context.INVOCATION_EDGE_CSV]
    evidence_csv = projection.files[profile_context.INVOCATION_EVIDENCE_CSV]
    measurements = {
        "index_raw": len(index_value),
        "index_gzip": profile_context._gzip_size(index_value),
        "shard_raw": max(map(len, shard_values)),
        "shard_gzip": max(map(profile_context._gzip_size, shard_values)),
        "all_shards_raw": sum(map(len, shard_values)),
        "all_shards_gzip": sum(
            map(profile_context._gzip_size, shard_values)
        ),
        "edge_csv_raw": len(edge_csv),
        "edge_csv_gzip": profile_context._gzip_size(edge_csv),
        "evidence_csv_raw": len(evidence_csv),
        "evidence_csv_gzip": profile_context._gzip_size(evidence_csv),
    }
    monkeypatch.setattr(
        profile_context, constant, measurements[measurement] - 1
    )
    with pytest.raises(profile_context.ProfileContextError, match="budget exceeded"):
        profile_context._validate_byte_budgets(projection)


def test_record_comparison_is_order_sensitive_and_checks_unique_keys():
    source = pd.DataFrame(
        [{"record_id": "a", "value": 1}, {"record_id": "b", "value": 2}]
    )
    with pytest.raises(profile_context.ProfileContextError, match="differs"):
        profile_context._assert_records_equal(
            [{"record_id": "b", "value": 2}, {"record_id": "a", "value": 1}],
            source,
            ("record_id", "value"),
            ("record_id",),
            "test rows",
        )
    with pytest.raises(profile_context.ProfileContextError, match="not unique"):
        profile_context._assert_records_equal(
            [{"record_id": "a", "value": 1}, {"record_id": "a", "value": 2}],
            source,
            ("record_id", "value"),
            ("record_id",),
            "test rows",
        )


def test_validator_rejects_exact_key_drift_and_literal_nan_identifiers(
    context_projection, fast_validator,
):
    extra_index_key = _clone_projection(context_projection)
    extra_index_key.index["unexpected"] = True
    with pytest.raises(profile_context.ProfileContextError, match="index key drift"):
        fast_validator(extra_index_key)

    nested_key = _clone_projection(context_projection)
    relative = "presidents/franklin-d-roosevelt_v1.json"
    nested_key.shards[relative]["topic_selections"]["default"]["unexpected"] = []
    _reseal_shard(nested_key, relative)
    with pytest.raises(
        profile_context.ProfileContextError, match="default topic selection key drift"
    ):
        fast_validator(nested_key)

    literal_nan = _clone_projection(context_projection)
    relationships = literal_nan.shards[relative]["invocations"]
    edge_id = next(iter(relationships["receipts_by_edge"]))
    relationships["receipts_by_edge"][edge_id][0]["candidate_id"] = "nan"
    with pytest.raises(profile_context.ProfileContextError, match="literal nan identifier"):
        fast_validator(literal_nan)


@pytest.mark.parametrize(
    "mutation",
    ("outgoing_order", "default_direction", "endpoint_partition"),
)
def test_validator_recomputes_invocation_direction_order_and_default(
    context_projection, fast_validator, mutation,
):
    projection = _clone_projection(context_projection)
    if mutation == "outgoing_order":
        relative, shard = next(
            (path, value) for path, value in projection.shards.items()
            if len(value["invocations"]["outgoing_edge_ids"]) >= 2
        )
        outgoing = shard["invocations"]["outgoing_edge_ids"]
        outgoing[0], outgoing[1] = outgoing[1], outgoing[0]
        message = "direction/order drift"
    elif mutation == "default_direction":
        relative, shard = next(
            (path, value) for path, value in projection.shards.items()
            if value["invocations"]["default_direction"] is not None
        )
        shard["invocations"]["default_direction"] = None
        message = "default direction drift"
    else:
        relative, shard = next(
            (path, value) for path, value in projection.shards.items()
            if value["invocations"]["outgoing_edge_ids"]
            and value["invocations"]["incoming_edge_ids"]
        )
        outgoing = shard["invocations"]["outgoing_edge_ids"]
        incoming = shard["invocations"]["incoming_edge_ids"]
        outgoing[0], incoming[0] = incoming[0], outgoing[0]
        message = "direction/order drift"
    _reseal_shard(projection, relative)
    with pytest.raises(profile_context.ProfileContextError, match=message):
        fast_validator(projection)


@pytest.mark.parametrize(
    "mutation",
    (
        "numerator", "claim", "threshold", "method", "attribution",
        "receipt", "primary_pairing",
    ),
)
def test_validator_reconstructs_every_enriched_legacy_field(
    context_projection, fast_validator, mutation,
):
    projection = _clone_projection(context_projection)
    relative, shard = next(
        (path, value) for path, value in projection.shards.items()
        if any(card["receipts"] for card in value["legacy_issue_evidence"]["cards"])
    )
    card = next(
        card for card in shard["legacy_issue_evidence"]["cards"]
        if card["receipts"]
    )
    if mutation == "numerator":
        card["exact_evidence"]["issue_paragraph_count"] += 1
    elif mutation == "claim":
        card["claim"]["text"] += " Altered."
    elif mutation == "threshold":
        card["why_shown"]["absolute_threshold_multiple"] += 0.01
    elif mutation == "method":
        card["method"]["stance_method"] = "invented_method"
    elif mutation == "attribution":
        card["receipts"][0]["actual_speaker_profile_id"] = "not-a-president"
    elif mutation == "receipt":
        card["receipts"][0]["excerpt"] += " Altered."
    else:
        card["legacy_v3"]["quote"] += " Altered."
    _reseal_shard(projection, relative)
    with pytest.raises(
        profile_context.ProfileContextError,
        match="enriched legacy issue evidence drift",
    ):
        fast_validator(projection)


@pytest.mark.parametrize(
    "mutation",
    ("source_identity", "counts", "budgets", "receipt_schema", "receipt_rows", "receipt_keys"),
)
def test_validator_rejects_manifest_identity_counts_budgets_and_receipt_drift(
    context_projection, fast_validator, mutation,
):
    projection = _clone_projection(context_projection)
    if mutation == "source_identity":
        projection.manifest["source_identity"]["contract_version"] = "stale"
        message = "manifest source identity drift"
    elif mutation == "counts":
        projection.manifest["counts"]["files_excluding_manifest"] -= 1
        message = "manifest count/key drift"
    elif mutation == "budgets":
        projection.manifest["budgets"]["index_raw_max"] -= 1
        message = "manifest budget/key drift"
    else:
        receipt = projection.manifest["files"][profile_context.INDEX_FILE]
        if mutation == "receipt_schema":
            receipt["schema_version"] = "wrong-schema"
        elif mutation == "receipt_rows":
            receipt["rows"] += 1
        else:
            receipt["unexpected"] = True
        _reseal_manifest(projection)
        message = r"manifest(/file)? receipt"
    with pytest.raises(profile_context.ProfileContextError, match=message):
        fast_validator(projection)


def test_validator_binds_shard_objects_index_receipts_and_exact_bytes(
    context_projection, fast_validator,
):
    relative = "presidents/franklin-d-roosevelt_v1.json"

    wrong_index_receipt = _clone_projection(context_projection)
    receipt = next(
        row for row in wrong_index_receipt.index["president_shards"]
        if row["filename"] == relative
    )
    receipt["sha256"] = "sha256:" + ("0" * 64)
    _reseal_index(wrong_index_receipt)
    with pytest.raises(profile_context.ProfileContextError, match="index hash drift"):
        fast_validator(wrong_index_receipt)

    object_bytes_mismatch = _clone_projection(context_projection)
    shard = object_bytes_mismatch.shards[relative]
    shard["counts"]["focal_topic_edges"] += 1
    shard["shard_sha256"] = profile_context._self_hash(shard, "shard_sha256")
    with pytest.raises(profile_context.ProfileContextError, match="object/byte drift"):
        fast_validator(object_bytes_mismatch)

    file_bytes_mismatch = _clone_projection(context_projection)
    file_bytes_mismatch.files[relative] += b" "
    with pytest.raises(profile_context.ProfileContextError, match="object/byte drift"):
        fast_validator(file_bytes_mismatch)


def test_publication_uses_unique_transaction_paths_without_touching_collisions(
    tmp_path, monkeypatch, context_inputs, context_projection,
):
    site = tmp_path / "site"
    parent = site / "data"
    parent.mkdir(parents=True)
    pid = os.getpid()
    collision_paths = (
        parent / f".profile-context-backup-{pid}",
        parent / f".profile-context-broken-{pid}",
        parent / f".profile-context-validation-{pid}",
        parent / ".profile-context-transaction-existing",
    )
    for path in collision_paths:
        path.mkdir()
        (path / "sentinel.txt").write_text("owned", encoding="utf-8")
    transactions_before = set(parent.glob(".profile-context-transaction-????????"))
    monkeypatch.setattr(profile_context, "validate_projection", lambda *args: None)
    monkeypatch.setattr(
        profile_context, "build_projection", lambda *args, **kwargs: context_projection
    )
    monkeypatch.setattr(
        profile_context, "validate_publication", lambda *args, **kwargs: {}
    )
    profile_context.write_public_projection(
        *context_inputs, site_dir=site, projection=context_projection
    )
    assert all(
        (path / "sentinel.txt").read_text(encoding="utf-8") == "owned"
        for path in collision_paths
    )
    assert set(parent.glob(".profile-context-transaction-????????")) == transactions_before


def test_failed_rollback_preserves_the_only_complete_backup(
    tmp_path, monkeypatch, context_inputs, context_projection,
):
    site = tmp_path / "site"
    output = site / "data/profile-context"
    output.mkdir(parents=True)
    (output / "original.txt").write_text("recover me", encoding="utf-8")
    monkeypatch.setattr(profile_context, "validate_projection", lambda *args: None)
    monkeypatch.setattr(
        profile_context, "build_projection", lambda *args, **kwargs: context_projection
    )
    validation_calls = 0

    def fail_post_swap(*args, **kwargs):
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 2:
            raise RuntimeError("post-swap validation failed")
        return {}

    monkeypatch.setattr(profile_context, "validate_publication", fail_post_swap)
    original_replace = os.replace

    def fail_restore(source, destination):
        if Path(source).name == "backup" and Path(destination) == output:
            raise OSError("simulated restore failure")
        return original_replace(source, destination)

    monkeypatch.setattr(profile_context.os, "replace", fail_restore)
    with pytest.raises(
        profile_context.ProfileContextError,
        match="rollback failed; preserved recovery data at",
    ) as error:
        profile_context.write_public_projection(
            *context_inputs, site_dir=site, projection=context_projection
        )
    transaction = Path(str(error.value).rsplit(" at ", 1)[1])
    assert transaction.is_dir()
    assert (transaction / "backup/original.txt").read_text(encoding="utf-8") == "recover me"
    assert (transaction / "failed-candidate" / profile_context.MANIFEST_FILE).is_file()
