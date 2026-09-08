from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from presidential_profiles import attention, metrics
from presidential_profiles import speaker_topic_network as network


def _topic_rows(keys: list[tuple[str, int]], topics: list[list[str]] | None = None) -> pd.DataFrame:
    topics = topics or [["A"] for _ in keys]
    return pd.DataFrame(
        [
            {"doc_name": doc, "para_idx": idx, "raw_topics": value}
            for (doc, idx), value in zip(keys, topics, strict=True)
        ]
    )


def test_exact_keyed_join_is_independent_of_row_order():
    foundation = pd.DataFrame(
        [{"doc_name": "b", "para_idx": 2, "value": 2}, {"doc_name": "a", "para_idx": 1, "value": 1}]
    )
    topics = _topic_rows([("a", 1), ("b", 2)]).iloc[::-1]
    joined = network.exact_topic_join(foundation, topics)
    assert list(joined[["doc_name", "para_idx"]].itertuples(index=False, name=None)) == [
        ("a", 1), ("b", 2)
    ]
    assert joined["value"].tolist() == [1, 2]


@pytest.mark.parametrize("side", ["foundation", "topics"])
def test_exact_keyed_join_rejects_duplicate_keys(side):
    foundation = pd.DataFrame([{"doc_name": "a", "para_idx": 1}])
    topics = _topic_rows([("a", 1)])
    if side == "foundation":
        foundation = pd.concat([foundation, foundation], ignore_index=True)
    else:
        topics = pd.concat([topics, topics], ignore_index=True)
    with pytest.raises(network.SpeakerTopicNetworkError, match="duplicate keys"):
        network.exact_topic_join(foundation, topics)


def test_exact_keyed_join_rejects_missing_and_extra_keys():
    foundation = pd.DataFrame([{"doc_name": "a", "para_idx": 1}])
    topics = _topic_rows([("b", 1)])
    with pytest.raises(network.SpeakerTopicNetworkError, match="key-set mismatch"):
        network.exact_topic_join(foundation, topics)


def _synthetic_joined() -> tuple[pd.DataFrame, dict, str, str]:
    taxonomy = attention.load_taxonomy()
    parent = attention.level1_parents(taxonomy)
    first = taxonomy["level2"][0]["name"]
    second = next(name for name, value in parent.items() if value == parent[first] and name != first)
    rows = []
    raw_topics = []
    for index, spec in enumerate(network.era_profiles.ERA_PROFILE_SPECS):
        rows.append(
            {
                "doc_name": f"eligible-{index}",
                "para_idx": index + 1,
                "text": f"Evidence paragraph {index}",
                "analysis_eligible": True,
                "attributed_speaker": "George Washington",
                "attributed_speaker_profile_id": "president-george-washington",
                "document_owner": "John Adams" if index == 0 else "George Washington",
                "document_owner_profile_id": "president-john-adams" if index == 0 else "president-george-washington",
                "cross_owner_paragraph": index == 0,
                "story_era_key": spec.key,
                "story_era": spec.label,
                "date": f"{max(spec.start_year, 1789):04d}-01-01",
                "title": f"Speech {index}",
                "source_url": f"https://example.test/{index}",
            }
        )
        raw_topics.append(
            [first, first.swapcase(), second]
            if index == 0
            else ([] if index == 1 else ([first, second] if index == 2 else [first]))
        )
    exclusion_reasons = ["multiple_speakers", "joint_shared", "unresolved", "non_president", "scaffolding"]
    for offset, reason in enumerate(exclusion_reasons, start=20):
        spec = network.era_profiles.ERA_PROFILE_SPECS[0]
        rows.append(
            {
                "doc_name": f"excluded-{reason}",
                "para_idx": offset,
                "text": "Excluded evidence",
                "analysis_eligible": False,
                "attributed_speaker": None,
                "attributed_speaker_profile_id": None,
                "document_owner": "George Washington",
                "document_owner_profile_id": "president-george-washington",
                "cross_owner_paragraph": False,
                "story_era_key": spec.key,
                "story_era": spec.label,
                "date": "1790-01-01",
                "title": reason,
                "source_url": "https://example.test/excluded",
            }
        )
        raw_topics.append([first])
    foundation = pd.DataFrame(rows)
    topics = _topic_rows(
        list(foundation[list(network.PARAGRAPH_KEY)].itertuples(index=False, name=None)), raw_topics
    )
    for column, value in {
        "promotion_id": "prom-1", "label_id": "label-1",
        "topic_annotation_run_id": "run-1", "topic_annotation_spec_version": "spec-v1",
        "spec_sha256": "sha256:" + "1" * 64, "promotion_rule_id": "primary-v1",
    }.items():
        topics[column] = value
    return network.exact_topic_join(foundation, topics), taxonomy, first, second


def test_normalization_preserves_actual_speaker_appearance_era_and_cross_owner():
    joined, taxonomy, first, second = _synthetic_joined()
    paragraphs, memberships, counts = network._normalize_joined(joined, taxonomy)
    assert len(paragraphs) == 9
    assert set(paragraphs["actual_speaker"]) == {"George Washington"}
    cross = paragraphs.loc[paragraphs["cross_owner"]].iloc[0]
    assert cross["source_document_owner"] == "John Adams"
    assert cross["appearance_id"] == network._appearance_id(
        cross["doc_name"], cross["president_profile_id"]
    )
    assert dict(paragraphs.groupby("story_era_id").size()) == {
        spec.key: 1 for spec in network.era_profiles.ERA_PROFILE_SPECS
    }
    first_row = paragraphs.iloc[0]
    assert json.loads(first_row["normalized_level2_topics_json"]) == [first, second]
    assert first_row["derived_level1_topic_count"] == 1
    assert counts["raw_level2_assignments"] - counts["normalized_level2_memberships"] == 1
    assert paragraphs.iloc[1]["topic_free"]
    assert not memberships["doc_name"].str.startswith("excluded-").any()
    assert len(memberships[memberships["doc_name"].eq("eligible-0")]) == 3


def test_unknown_taxonomy_label_fails_even_on_an_excluded_record():
    joined, taxonomy, _, _ = _synthetic_joined()
    joined.at[joined.index[-1], "raw_topics"] = ["Not a canonical topic"]
    with pytest.raises(network.SpeakerTopicNetworkError, match="not in taxonomy"):
        network._normalize_joined(joined, taxonomy)


def test_topic_free_denominators_and_multilabel_shares_are_non_additive():
    joined, taxonomy, _, _ = _synthetic_joined()
    paragraphs, memberships, _ = network._normalize_joined(joined, taxonomy)
    president = pd.DataFrame(
        [{
            "schema_version": network.PRESIDENT_NODE_SCHEMA,
            "contract_version": network.CONTRACT_VERSION,
            "president_profile_id": "president-george-washington",
            "president_name": "George Washington", "party": "Unaffiliated",
            "display_order": 1, "node_type": "president",
        }]
    )
    topic_nodes = network._topic_nodes(taxonomy)
    president_support, _, edges = network._support_tables(
        paragraphs, memberships, president, topic_nodes
    )
    corpus_support = president_support[president_support["scope_type"].eq("corpus")].iloc[0]
    assert corpus_support["eligible_president_paragraph_count"] == 9
    assert corpus_support["eligible_president_appearance_count"] == 9
    corpus_edges = edges[edges["scope_type"].eq("corpus")]
    level2_sum = corpus_edges[corpus_edges["topic_level"].eq("level2")]["speaker_paragraph_share"].sum()
    assert level2_sum > 1


@pytest.mark.parametrize(
    ("appearances", "expected"),
    [(0, "no_record"), (1, "thin"), (4, "thin"), (5, "supported")],
)
def test_president_support_thresholds(appearances, expected):
    assert network.president_support_status(appearances) == expected


@pytest.mark.parametrize(
    ("president_status", "paragraphs", "appearances", "support", "visible"),
    [
        ("thin", 100, 20, "thin_president", False),
        ("supported", 4, 2, "thin_edge", False),
        ("supported", 5, 1, "thin_edge", False),
        ("supported", 5, 2, "supported", False),
        ("supported", 20, 5, "supported", True),
    ],
)
def test_edge_and_default_visibility_thresholds(
    president_status, paragraphs, appearances, support, visible
):
    assert network.edge_support_status(president_status, paragraphs, appearances) == support
    assert network.is_default_visible(president_status, paragraphs, appearances) is visible


def test_zero_denominators_are_null_not_zero():
    assert network._safe_ratio(0, 0) is None
    assert network._safe_ratio(0, 5) == 0


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, []),
        (1, [("first", 0)]),
        (2, [("first", 0), ("last", 1)]),
        (3, [("first", 0), ("middle", 1), ("last", 2)]),
        (4, [("first", 0), ("middle", 1), ("last", 3)]),
    ],
)
def test_evidence_selection_is_deterministic_and_deduplicated(n, expected):
    assert network.evidence_selection_positions(n) == expected


def test_evidence_receipts_use_chronology_and_lowest_paragraph_in_appearance():
    edge = pd.DataFrame(
        [{
            "edge_id": "edge-1", "scope_type": "corpus", "scope_id": "all-corpus",
            "scope_label": "All eligible corpus", "president_profile_id": "p1",
            "president_name": "President One", "topic_id": "t1",
            "topic_level": "level2", "topic_label": "Topic One",
            "evidence_receipt_count": 0,
        }]
    )
    rows = []
    for date, doc, para in [
        ("1800-01-01", "a", 2), ("1800-01-01", "a", 1),
        ("1810-01-01", "b", 8), ("1820-01-01", "c", 4),
        ("1830-01-01", "d", 3),
    ]:
        rows.append(
            {
                "president_profile_id": "p1", "topic_id": "t1",
                "story_era_id": "founding", "speech_date": date,
                "doc_name": doc, "para_idx": para, "appearance_id": f"appearance-{doc}",
                "actual_speaker": "President One", "source_document_owner_profile_id": "owner",
                "source_document_owner": "Document Owner", "cross_owner": True,
                "story_era_label": "Founding", "speech_title": doc.upper(),
                "source_url": f"https://example.test/{doc}", "evidence_excerpt": f"excerpt-{doc}-{para}",
            }
        )
    edges, receipts = network._evidence_receipts(edge, pd.DataFrame(rows))
    assert edges.iloc[0]["evidence_receipt_count"] == 3
    assert list(receipts[["selection_role", "doc_name", "para_idx"]].itertuples(index=False, name=None)) == [
        ("first", "a", 1), ("middle", "b", 8), ("last", "d", 3)
    ]


def test_malformed_and_internally_stale_promotion_pointers_are_rejected(tmp_path):
    root = tmp_path / "materialized"
    root.mkdir()
    (root / "current").write_text("not-a-generation\n")
    with pytest.raises(network.SpeakerTopicNetworkError, match="malformed"):
        network.resolve_active_topics(materialized_root=root)
    generation = "a" * 64
    (root / "current").write_text(generation + "\n")
    path = root / "generations" / generation
    path.mkdir(parents=True)
    (path / "generation.json").write_text("{}\n")
    with pytest.raises(network.SpeakerTopicNetworkError, match="content hash drift"):
        network.resolve_active_topics(materialized_root=root)


@pytest.mark.parametrize("protected", network.PROTECTED_OUTPUT_ROOTS)
def test_protected_output_paths_are_rejected(protected):
    with pytest.raises(network.SpeakerTopicNetworkError, match="protected"):
        network._assert_safe_output_path(protected / "network")


def test_metric_registry_uses_named_actual_speaker_measures_and_cautions():
    for key in ("actual_speaker_topic_presence", "actual_speaker_topic_contribution"):
        metric = metrics.METRICS[key]
        text = " ".join(str(value) for value in metric.values()).lower()
        assert "actual" in text and "topic-free" in text and "non-additive" in text
        assert "non-causal" in text and "confidence" in text and "thin" in text
        assert metric["source"] == "speaker_topic_network/edges_v1.parquet"
    assert "actual_speaker_topic_presence" not in metrics.CHART_METRICS
    assert "actual_speaker_topic_contribution" not in metrics.CHART_METRICS


@pytest.fixture(scope="module")
def real_bundle():
    return network.build_network_bundle()


def test_real_artifact_acceptance_matches_every_pinned_count(real_bundle):
    assert real_bundle.acceptance["counts"] == network.EXPECTED_ACCEPTANCE
    assert real_bundle.acceptance["distributions"] == network.EXPECTED_DISTRIBUTIONS
    assert real_bundle.acceptance["all_pinned_acceptance_values_match"]
    for frame in (
        real_bundle.president_support, real_bundle.topic_support, real_bundle.edges
    ):
        assert frame.loc[frame["scope_type"].eq("corpus"), "story_era_id"].isna().all()
        era_rows = frame[frame["scope_type"].eq("story_era")]
        assert era_rows["story_era_id"].equals(era_rows["scope_id"])


def test_stable_governed_and_public_bytes_and_manifest_consistency(real_bundle, tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    for root in (left, right):
        network._write_candidate_governed(real_bundle, root / "governed")
        network._write_candidate_public(real_bundle, root / "site" / "data" / network.PUBLIC_DIR_NAME)
    assert network._directory_bytes(left) == network._directory_bytes(right)
    loaded = network.load_network_bundle(
        network_dir=left / "governed", validate_current_inputs=False
    )
    manifest = network.validate_publication(loaded, left / "site")
    assert manifest["counts"]["edges"] == 4_408
    assert manifest["counts"]["memberships"] == 91_119


def test_accepted_bundle_rejects_stale_promotion_provenance(real_bundle, tmp_path):
    governed = tmp_path / "governed"
    network._write_candidate_governed(real_bundle, governed)
    meta_path = governed / network.GOVERNED_FILES["meta"]
    meta = json.loads(meta_path.read_text())
    meta["input_provenance"]["active_topic_promotion"]["active_generation_id"] = "0" * 64
    meta["metadata_sha256"] = network._self_hash(meta, "metadata_sha256")
    meta_path.write_bytes(network._json_bytes(meta))
    with pytest.raises(network.SpeakerTopicNetworkError, match="active topic promotion"):
        network.load_network_bundle(network_dir=governed)


def test_atomic_publication_rolls_back_on_replacement_failure(real_bundle, tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    target = tmp_path / "target"
    network._write_candidate_public(real_bundle, candidate)
    original = network._atomic_replace
    calls = 0

    def fail_second(source: Path, destination: Path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated replacement failure")
        original(source, destination)

    monkeypatch.setattr(network, "_atomic_replace", fail_second)
    with pytest.raises(OSError, match="simulated"):
        network._publish_candidate(candidate, target, list(network.PUBLIC_FILES.values()))
    assert not list(target.glob("*"))
