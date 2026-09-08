from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from presidential_profiles import actual_speaker_invocation_network as network
from presidential_profiles import speaker_topic_network, story_foundation


EXPECTED_EDGE_FIELDS = (
    "edge_id",
    "edge_type",
    "source_president_profile_id",
    "source_president_name",
    "target_president_profile_id",
    "target_president_name",
    "raw_mentions",
    "reference_paragraph_count",
    "distinct_speeches",
    "function_counts",
    "stance_counts",
    "first_speech_date",
    "last_speech_date",
    "evidence_row_count",
    "evidence_status",
)
EXPECTED_EVIDENCE_FIELDS = (
    "candidate_id",
    "edge_id",
    "source_president_profile_id",
    "source_president_name",
    "target_president_profile_id",
    "target_president_name",
    "doc_name",
    "para_idx",
    "speech_date",
    "story_era_id",
    "story_era_label",
    "speech_title",
    "source_url",
    "raw_mention",
    "function",
    "stance",
    "evidence_span",
    "rationale",
    "quotation_status",
    "target_status",
    "rubric_version",
    "evidence_status",
    "annotation_speaker",
    "source_document_owner_profile_id",
    "source_document_owner",
    "cross_owner",
)
EXPECTED_ACCEPTANCE = {
    "candidate_rows": 5_610,
    "classification_rows": 2_032,
    "accepted_evidence_rows": 2_032,
    "former_president_source_rows": 1_447,
    "unresolved_keys": 48,
    "ineligible_rows": 156,
    "reassigned_speakers": 27,
    "retained_rows": 1_243,
    "reference_paragraphs": 915,
    "source_speeches": 405,
    "directed_edges": 303,
    "source_presidents": 38,
    "target_presidents": 44,
    "actual_speaker_self_rows": 0,
}


@pytest.fixture(scope="module")
def real_context():
    foundation = story_foundation.load_story_foundation()
    topic_bundle = speaker_topic_network.load_network_bundle(foundation=foundation)

    replayed_candidate_ids: list[str] = []
    original_classifier = network.invocations.classify_candidate

    def traced_classifier(candidate: pd.Series) -> dict[str, str]:
        replayed_candidate_ids.append(str(candidate["candidate_id"]))
        return original_classifier(candidate)

    network.invocations.classify_candidate = traced_classifier
    try:
        bundle = network.build_network_bundle(foundation, topic_bundle)
    finally:
        network.invocations.classify_candidate = original_classifier

    classifications = pd.read_parquet(network.invocations.CLASSIFICATIONS)
    accepted = pd.read_parquet(story_foundation.INVOCATION_EVIDENCE_PATH)
    accepted_with_rubric = accepted.merge(
        classifications.loc[:, ["candidate_id", "rubric_version"]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    retained, overlay = story_foundation.overlay_invocation_evidence(
        foundation, accepted_with_rubric
    )
    return SimpleNamespace(
        bundle=bundle,
        foundation=foundation,
        topic_bundle=topic_bundle,
        classifications=classifications,
        retained=retained,
        overlay=overlay,
        replayed_candidate_ids=replayed_candidate_ids,
    )


def test_contract_versions_fields_and_fixed_enum_order_are_exact(real_context):
    bundle = real_context.bundle
    assert network.CONTRACT_VERSION == "actual-speaker-invocation-network-v1"
    assert network.EDGE_SCHEMA == "actual-speaker-invocation-edge-v1"
    assert network.EVIDENCE_SCHEMA == "actual-speaker-invocation-evidence-v1"
    assert network.EDGE_FIELDS == EXPECTED_EDGE_FIELDS
    assert network.EVIDENCE_FIELDS == EXPECTED_EVIDENCE_FIELDS
    assert tuple(bundle.edges.columns) == EXPECTED_EDGE_FIELDS
    assert tuple(bundle.evidence.columns) == EXPECTED_EVIDENCE_FIELDS
    assert network.FUNCTIONS == (
        "legacy/inheritance",
        "institutional precedent",
        "policy inheritance",
        "historical comparison",
        "contemporary rivalry",
        "ceremonial/biographical",
        "other/unclear",
    )
    assert network.STANCES == (
        "positive", "negative", "mixed", "neutral", "unclear"
    )
    assert set(network.FUNCTIONS) == network.invocations.FUNCTIONS
    assert set(network.STANCES) == network.invocations.STANCES

    edge_records = network.edge_records(bundle)
    evidence_records = network.evidence_records(bundle)
    assert list(edge_records[0]) == list(EXPECTED_EDGE_FIELDS)
    assert list(evidence_records[0]) == list(EXPECTED_EVIDENCE_FIELDS)
    json.dumps(
        {"edges": edge_records, "evidence": evidence_records},
        allow_nan=False,
        sort_keys=True,
    )


def test_real_bundle_matches_every_pinned_population_count(real_context):
    bundle = real_context.bundle
    assert network.EXPECTED_ACCEPTANCE == EXPECTED_ACCEPTANCE
    assert bundle.acceptance == EXPECTED_ACCEPTANCE
    assert len(bundle.edges) == 303
    assert len(bundle.evidence) == 1_243
    assert bundle.edges["edge_id"].is_unique
    assert bundle.evidence["candidate_id"].is_unique
    assert bundle.evidence[["doc_name", "para_idx"]].drop_duplicates().shape[0] == 915
    assert bundle.evidence["doc_name"].nunique() == 405
    assert bundle.edges["source_president_name"].nunique() == 38
    assert bundle.edges["target_president_name"].nunique() == 44
    assert int(bundle.edges["raw_mentions"].sum()) == 1_243
    assert int(bundle.edges["evidence_row_count"].sum()) == 1_243
    assert real_context.overlay == {
        "source_rows": 1_447,
        "unresolved_keys": 48,
        "ineligible_rows": 156,
        "reassigned_speakers": 27,
        "retained_rows": 1_243,
    }


def test_real_source_identity_and_every_classifier_replay_are_recorded(real_context):
    identity = real_context.bundle.source_identity
    assert identity["contract_version"] == network.CONTRACT_VERSION
    assert identity["rubric_version"] == network.invocations.RUBRIC_VERSION
    assert identity["speaker_foundation"] == (
        real_context.topic_bundle.meta["input_provenance"]["story_foundation"]
    )
    assert identity["president_catalog_metadata_sha256"] == (
        real_context.topic_bundle.meta["metadata_sha256"]
    )
    assert {
        name: receipt["sha256"]
        for name, receipt in identity["accepted_sources"].items()
    } == network.EXPECTED_SOURCE_HASHES
    assert all(
        not receipt["path"].startswith("/") and receipt["bytes"] > 0
        for receipt in identity["accepted_sources"].values()
    )
    assert len(identity["candidate_fingerprint"]) == 64
    assert len(identity["classifier_prompt_sha256"]) == 64

    replayed = real_context.replayed_candidate_ids
    classified = real_context.classifications["candidate_id"].astype(str).tolist()
    assert len(replayed) == 2_032
    assert len(set(replayed)) == 2_032
    assert set(replayed) == set(classified)


def test_every_edge_has_zero_filled_ordered_counts_and_reconciles(real_context):
    bundle = real_context.bundle
    grouped = {
        edge_id: rows
        for edge_id, rows in bundle.evidence.groupby("edge_id", sort=False)
    }
    for edge in bundle.edges.itertuples(index=False):
        rows = grouped[edge.edge_id]
        assert tuple(edge.function_counts) == network.FUNCTIONS
        assert tuple(edge.stance_counts) == network.STANCES
        assert all(
            isinstance(value, int) and value >= 0
            for value in edge.function_counts.values()
        )
        assert all(isinstance(value, int) and value >= 0 for value in edge.stance_counts.values())
        assert 0 in edge.function_counts.values()
        assert 0 in edge.stance_counts.values()
        assert sum(edge.function_counts.values()) == edge.raw_mentions == len(rows)
        assert sum(edge.stance_counts.values()) == edge.raw_mentions
        assert edge.evidence_row_count == len(rows)
        assert edge.reference_paragraph_count == len(
            rows[["doc_name", "para_idx"]].drop_duplicates()
        )
        assert edge.distinct_speeches == rows["doc_name"].nunique()
        assert edge.first_speech_date == rows["speech_date"].min()
        assert edge.last_speech_date == rows["speech_date"].max()
        assert edge.function_counts == {
            value: int(rows["function"].eq(value).sum())
            for value in network.FUNCTIONS
        }
        assert edge.stance_counts == {
            value: int(rows["stance"].eq(value).sum())
            for value in network.STANCES
        }


def test_edge_and_evidence_ordering_use_all_declared_tie_breaks(real_context):
    bundle = real_context.bundle
    display_order = real_context.topic_bundle.president_nodes.set_index(
        "president_profile_id"
    )["display_order"].astype(int).to_dict()
    observed_edge_keys = [
        (
            -int(row.reference_paragraph_count),
            -int(row.raw_mentions),
            -int(row.distinct_speeches),
            display_order[row.target_president_profile_id],
            row.edge_id,
        )
        for row in bundle.edges.itertuples(index=False)
    ]
    assert observed_edge_keys == sorted(observed_edge_keys)

    observed_evidence_keys = list(
        bundle.evidence[
            ["edge_id", "speech_date", "doc_name", "para_idx", "candidate_id"]
        ].itertuples(index=False, name=None)
    )
    assert observed_evidence_keys == sorted(observed_evidence_keys)


def test_real_aggregation_is_stable_under_input_and_catalog_permutation(real_context):
    baseline = real_context.bundle
    for seed in (17, 93):
        retained = real_context.retained.sample(frac=1, random_state=seed)
        president_nodes = real_context.topic_bundle.president_nodes.sample(
            frac=1, random_state=seed + 1
        )
        edges, evidence = network._build_records(retained, president_nodes)
        pd.testing.assert_frame_equal(edges, baseline.edges)
        pd.testing.assert_frame_equal(evidence, baseline.evidence)


@pytest.mark.parametrize(
    ("doc_name", "supplied"),
    [
        (
            "january-20-2025-inaugural-address",
            network.MILLER_PREFIX + "january-20-2025-inaugural-address",
        ),
        (
            "/the-presidency/presidential-speeches/"
            "january-20-2025-inaugural-address",
            network.MILLER_PREFIX + "january-20-2025-inaugural-address",
        ),
        (
            "the-presidency/presidential-speeches/"
            "january-20-2025-inaugural-address",
            network.MILLER_PREFIX + "january-20-2025-inaugural-address",
        ),
        (
            "january-20-2025-inaugural-address",
            "https://millercenter.orgjanuary-20-2025-inaugural-address",
        ),
    ],
)
def test_miller_urls_normalize_only_governed_document_forms(doc_name, supplied):
    assert network.normalized_miller_url(doc_name, supplied) == (
        network.MILLER_PREFIX + "january-20-2025-inaugural-address"
    )


@pytest.mark.parametrize(
    ("doc_name", "supplied"),
    [
        ("january-20-2025-inaugural-address", ""),
        (
            "january-20-2025-inaugural-address",
            "http://millercenter.org/the-presidency/presidential-speeches/"
            "january-20-2025-inaugural-address",
        ),
        (
            "january-20-2025-inaugural-address",
            "https://example.test/the-presidency/presidential-speeches/"
            "january-20-2025-inaugural-address",
        ),
        (
            "january-20-2025-inaugural-address",
            network.MILLER_PREFIX + "different-speech",
        ),
        (
            "january-20-2025-inaugural-address",
            network.MILLER_PREFIX + "january-20-2025-inaugural-address?download=1",
        ),
        (
            "../january-20-2025-inaugural-address",
            network.MILLER_PREFIX + "january-20-2025-inaugural-address",
        ),
    ],
)
def test_miller_urls_refuse_missing_mismatched_or_unsafe_values(doc_name, supplied):
    with pytest.raises(network.ActualSpeakerInvocationNetworkError):
        network.normalized_miller_url(doc_name, supplied)


def test_every_real_receipt_has_its_keyed_canonical_miller_url(real_context):
    receipts = real_context.bundle.evidence[
        ["doc_name", "source_url"]
    ].drop_duplicates()
    assert len(receipts) == 405
    for row in receipts.itertuples(index=False):
        assert row.source_url.startswith(network.MILLER_PREFIX)
        assert network.normalized_miller_url(row.doc_name, row.source_url) == row.source_url


def test_trump_to_biden_later_corpus_speeches_are_retained(real_context):
    bundle = real_context.bundle
    rows = bundle.evidence[
        bundle.evidence["source_president_name"].eq("Donald Trump")
        & bundle.evidence["target_president_name"].eq("Joe Biden")
    ]
    assert len(rows) == 39
    assert rows["target_status"].eq("former_president").all()
    assert rows["speech_date"].min() == "2025-01-20"
    assert rows["speech_date"].max() == "2026-02-25"
    assert {date[:4] for date in rows["speech_date"]} == {"2025", "2026"}

    edge = bundle.edges[
        bundle.edges["source_president_name"].eq("Donald Trump")
        & bundle.edges["target_president_name"].eq("Joe Biden")
    ].iloc[0]
    assert edge["raw_mentions"] == 39
    assert edge["reference_paragraph_count"] == 36
    assert edge["distinct_speeches"] == 6
    orders = real_context.topic_bundle.president_nodes.set_index("president_name")[
        "display_order"
    ]
    assert orders["Joe Biden"] > orders["Donald Trump"]


def test_actual_speaker_edges_exclude_self_and_preserve_reassignments(real_context):
    evidence = real_context.bundle.evidence
    assert not evidence["source_president_profile_id"].eq(
        evidence["target_president_profile_id"]
    ).any()
    assert not evidence["source_president_name"].eq(
        evidence["target_president_name"]
    ).any()
    assert int(
        evidence["annotation_speaker"].ne(evidence["source_president_name"]).sum()
    ) == 27


def test_contract_excludes_legacy_normalized_confidence_and_majority_fields(real_context):
    prohibited = {
        "mentions_per_100_speeches",
        "normalized_rate",
        "confidence",
        "majority_function",
        "majority_stance",
    }
    assert prohibited.isdisjoint(real_context.bundle.edges.columns)
    assert prohibited.isdisjoint(real_context.bundle.evidence.columns)
    assert set(real_context.bundle.edges.columns) == set(EXPECTED_EDGE_FIELDS)
    assert set(real_context.bundle.evidence.columns) == set(EXPECTED_EVIDENCE_FIELDS)


@pytest.mark.parametrize(
    "field",
    [
        "mentions_per_100_speeches",
        "normalized_rate",
        "confidence",
        "majority_function",
        "majority_stance",
    ],
)
def test_validator_refuses_prohibited_extra_edge_fields(real_context, field):
    edges = real_context.bundle.edges.assign(**{field: 0})
    with pytest.raises(
        network.ActualSpeakerInvocationNetworkError, match="schema drift"
    ):
        network.validate_bundle(replace(real_context.bundle, edges=edges))


def test_validator_refuses_unsafe_receipt_urls_and_self_edges(real_context):
    evidence = real_context.bundle.evidence.copy()
    evidence.at[0, "source_url"] = "javascript:alert(1)"
    with pytest.raises(
        network.ActualSpeakerInvocationNetworkError,
        match="unsafe invocation source URL",
    ):
        network.validate_bundle(replace(real_context.bundle, evidence=evidence))

    evidence = real_context.bundle.evidence.copy()
    evidence.at[0, "source_president_profile_id"] = evidence.at[
        0, "target_president_profile_id"
    ]
    with pytest.raises(
        network.ActualSpeakerInvocationNetworkError, match="self edge"
    ):
        network.validate_bundle(replace(real_context.bundle, evidence=evidence))


@pytest.mark.parametrize("mutation", ["missing_zero", "count_drift"])
def test_validator_refuses_incomplete_or_unreconciled_function_counts(
    real_context, mutation
):
    edges = real_context.bundle.edges.copy()
    counts = deepcopy(edges.at[0, "function_counts"])
    if mutation == "missing_zero":
        zero_key = next(key for key, value in counts.items() if value == 0)
        counts.pop(zero_key)
        expected = "function-count order drift"
    else:
        counts[network.FUNCTIONS[0]] += 1
        expected = "does not reconcile"
    edges.at[0, "function_counts"] = counts
    with pytest.raises(network.ActualSpeakerInvocationNetworkError, match=expected):
        network.validate_bundle(replace(real_context.bundle, edges=edges))


@pytest.mark.parametrize(
    ("candidate_id", "annotation_speaker", "actual_speaker", "target"),
    [
        (
            "176e5bda444441d3a0a6c5e3",
            "John F. Kennedy",
            "Richard M. Nixon",
            "Theodore Roosevelt",
        ),
        (
            "558833d49d6e707b7a7df01b",
            "Jimmy Carter",
            "Gerald Ford",
            "Harry S. Truman",
        ),
        (
            "2c95921cf2770d6fcc91dec6",
            "Jimmy Carter",
            "Ronald Reagan",
            "Gerald Ford",
        ),
    ],
)
def test_representative_debate_rows_use_the_accepted_actual_speaker_overlay(
    real_context, candidate_id, annotation_speaker, actual_speaker, target
):
    row = real_context.bundle.evidence.set_index("candidate_id").loc[candidate_id]
    assert row["annotation_speaker"] == annotation_speaker
    assert row["source_document_owner"] == annotation_speaker
    assert row["source_president_name"] == actual_speaker
    assert row["target_president_name"] == target
    assert bool(row["cross_owner"]) is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target_status", "serving_or_contemporary", "non-former"),
        ("rubric_version", "invocation-v2-rubric-drift", "rubric-version"),
        ("evidence_status", "rejected", "evidence-status"),
        ("source_president_profile_id", "unknown-profile", "president identity"),
        ("source_document_owner_profile_id", None, "null or empty"),
    ],
)
def test_validator_refuses_status_rubric_and_identifier_drift(
    real_context, field, value, message
):
    evidence = real_context.bundle.evidence.copy()
    evidence.at[0, field] = value
    with pytest.raises(network.ActualSpeakerInvocationNetworkError, match=message):
        network.validate_bundle(replace(real_context.bundle, evidence=evidence))


def test_validator_refuses_prefix_matching_but_noncanonical_url(real_context):
    evidence = real_context.bundle.evidence.copy()
    evidence.at[0, "source_url"] = evidence.at[0, "source_url"] + "?download=1"
    with pytest.raises(
        network.ActualSpeakerInvocationNetworkError,
        match="unsafe|non-canonical",
    ):
        network.validate_bundle(replace(real_context.bundle, evidence=evidence))


@pytest.mark.parametrize("mutation", ["annotation", "owner", "cross_owner"])
def test_validator_refuses_speaker_owner_attribution_drift(real_context, mutation):
    evidence = real_context.bundle.evidence.copy()
    row = evidence.iloc[0]
    if mutation == "annotation":
        replacement = next(
            name
            for name in network.corpus.PARTY
            if name != row["source_document_owner"]
        )
        evidence.at[0, "annotation_speaker"] = replacement
    elif mutation == "owner":
        replacement_id = next(
            profile_id
            for _, profile_id, _ in real_context.bundle._president_catalog
            if profile_id != row["source_document_owner_profile_id"]
        )
        evidence.at[0, "source_document_owner_profile_id"] = replacement_id
    else:
        evidence.at[0, "cross_owner"] = not bool(row["cross_owner"])
    with pytest.raises(
        network.ActualSpeakerInvocationNetworkError,
        match="identity|attribution",
    ):
        network.validate_bundle(replace(real_context.bundle, evidence=evidence))


def test_validator_refuses_edge_type_and_edge_evidence_identity_drift(real_context):
    edges = real_context.bundle.edges.copy()
    edges.at[0, "edge_type"] = "legacy_document_owner_edge"
    with pytest.raises(network.ActualSpeakerInvocationNetworkError, match="edge-type"):
        network.validate_bundle(replace(real_context.bundle, edges=edges))

    edges = real_context.bundle.edges.copy()
    edges.at[0, "source_president_name"] = "George Washington"
    with pytest.raises(
        network.ActualSpeakerInvocationNetworkError,
        match="edge president identity|edge/evidence",
    ):
        network.validate_bundle(replace(real_context.bundle, edges=edges))


def test_validator_refuses_non_content_derived_edge_id(real_context):
    edges = real_context.bundle.edges.copy()
    evidence = real_context.bundle.evidence.copy()
    old_id = edges.at[0, "edge_id"]
    new_id = "invocation_edge_" + "0" * 64
    edges.at[0, "edge_id"] = new_id
    evidence.loc[evidence["edge_id"].eq(old_id), "edge_id"] = new_id
    evidence = evidence.sort_values(
        ["edge_id", "speech_date", "doc_name", "para_idx", "candidate_id"],
        kind="stable",
    ).reset_index(drop=True)
    with pytest.raises(
        network.ActualSpeakerInvocationNetworkError,
        match="content identity",
    ):
        network.validate_bundle(
            replace(real_context.bundle, edges=edges, evidence=evidence)
        )


@pytest.mark.parametrize("mutation", ["scalar_float", "map_bool"])
def test_validator_refuses_non_integer_count_types(real_context, mutation):
    edges = real_context.bundle.edges.copy()
    if mutation == "scalar_float":
        edges["raw_mentions"] = edges["raw_mentions"].astype(object)
        edges.at[0, "raw_mentions"] = float(edges.at[0, "raw_mentions"])
        message = "integer-count type"
    else:
        index = next(
            index
            for index, counts in enumerate(edges["function_counts"])
            if 1 in counts.values()
        )
        counts = deepcopy(edges.at[index, "function_counts"])
        key = next(key for key, value in counts.items() if value == 1)
        counts[key] = True
        edges.at[index, "function_counts"] = counts
        message = "function-count integer type"
    with pytest.raises(network.ActualSpeakerInvocationNetworkError, match=message):
        network.validate_bundle(replace(real_context.bundle, edges=edges))


@pytest.mark.parametrize("frame_name", ["edges", "evidence"])
def test_validator_refuses_deterministic_row_order_drift(real_context, frame_name):
    frame = getattr(real_context.bundle, frame_name).iloc[::-1].reset_index(drop=True)
    with pytest.raises(
        network.ActualSpeakerInvocationNetworkError,
        match="deterministic ordering",
    ):
        network.validate_bundle(replace(real_context.bundle, **{frame_name: frame}))


def test_validator_refuses_source_identity_hash_drift(real_context):
    identity = deepcopy(real_context.bundle.source_identity)
    identity["accepted_sources"]["candidates"]["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(
        network.ActualSpeakerInvocationNetworkError,
        match="accepted-source hash|source identity projection",
    ):
        network.validate_bundle(replace(real_context.bundle, source_identity=identity))


@pytest.mark.parametrize("enum_name", ["FUNCTIONS", "STANCES"])
def test_validator_refuses_runtime_enum_drift(real_context, monkeypatch, enum_name):
    current = getattr(network.invocations, enum_name)
    monkeypatch.setattr(network.invocations, enum_name, set(current) | {"runtime drift"})
    with pytest.raises(
        network.ActualSpeakerInvocationNetworkError,
        match="runtime invocation .* enum drift",
    ):
        network.validate_bundle(real_context.bundle)


def test_builder_authenticates_passed_foundation_provenance(real_context):
    provenance = deepcopy(real_context.foundation.provenance)
    provenance["source_hashes"]["paragraph_view"] = "sha256:" + "0" * 64
    foundation = replace(real_context.foundation, provenance=provenance)
    with pytest.raises(
        network.ActualSpeakerInvocationNetworkError,
        match="does not match the accepted topic bundle",
    ):
        network.build_network_bundle(foundation, real_context.topic_bundle)


def test_builder_authenticates_exact_in_memory_foundation_rows(real_context):
    paragraphs = real_context.foundation.paragraph_view.copy()
    retained = real_context.retained.iloc[0]
    mask = paragraphs["doc_name"].eq(retained["doc_name"]) & paragraphs[
        "para_idx"
    ].eq(int(retained["para_idx"]))
    assert int(mask.sum()) == 1
    paragraphs.loc[mask, "document_owner_profile_id"] = "unknown-profile"
    foundation = replace(real_context.foundation, paragraph_view=paragraphs)
    with pytest.raises(
        network.ActualSpeakerInvocationNetworkError,
        match="in-memory speaker foundation differs",
    ):
        network.build_network_bundle(foundation, real_context.topic_bundle)
