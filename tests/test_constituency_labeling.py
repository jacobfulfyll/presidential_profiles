from __future__ import annotations

import builtins
import json
import socket
import sys
from pathlib import Path

import pandas as pd
import pytest

from presidential_profiles import constituency_labeling as C


def _source_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paragraphs = pd.DataFrame(
        [
            {
                "doc_name": "/speech-a",
                "para_idx": 0,
                "text": "We act for the loyal citizens of this Union.",
                "word_count": 9,
            },
            {
                "doc_name": "/speech-a",
                "para_idx": 1,
                "text": "Their rights must be protected by the Government.",
                "word_count": 8,
            },
            {
                "doc_name": "/speech-a",
                "para_idx": 2,
                "text": "The weather has improved.",
                "word_count": 4,
            },
            {
                "doc_name": "/speech-b",
                "para_idx": 0,
                "text": "Farmers will receive the benefit of this measure.",
                "word_count": 8,
            },
            {
                "doc_name": "/speech-b",
                "para_idx": 1,
                "text": "I ask Congress to pass the bill.",
                "word_count": 7,
            },
        ]
    )
    speeches = pd.DataFrame(
        [
            {
                "doc_name": "/speech-a",
                "president": "Abraham Lincoln",
                "year": 1861,
                "title": "A Union Address",
            },
            {
                "doc_name": "/speech-b",
                "president": "Ulysses S. Grant",
                "year": 1871,
                "title": "An Agricultural Address",
            },
        ]
    )
    annotations = pd.DataFrame(
        [
            {"doc_name": "/speech-a", "speech_type": "annual_message"},
            {"doc_name": "/speech-b", "speech_type": "public_remarks_or_address"},
        ]
    )
    return paragraphs, speeches, annotations


def _write_sources(
    tmp_path: Path,
    *,
    paragraphs: pd.DataFrame | None = None,
    exclusions: pd.DataFrame | None = None,
) -> tuple[Path, Path, Path, Path | None]:
    default_paragraphs, speeches, annotations = _source_frames()
    paragraphs = default_paragraphs if paragraphs is None else paragraphs
    paragraph_path = tmp_path / "canonical_paragraphs.parquet"
    speech_path = tmp_path / "canonical_speeches.parquet"
    annotation_path = tmp_path / "speech_annotations.parquet"
    paragraphs.to_parquet(paragraph_path, index=False)
    speeches.to_parquet(speech_path, index=False)
    annotations.to_parquet(annotation_path, index=False)
    exclusion_path = None
    if exclusions is not None:
        exclusion_path = tmp_path / "exclusions.parquet"
        exclusions.to_parquet(exclusion_path, index=False)
    return paragraph_path, speech_path, annotation_path, exclusion_path


def _initialize(
    tmp_path: Path,
    *,
    dataset_id: str = "test-v1",
    paragraphs: pd.DataFrame | None = None,
    exclusions: pd.DataFrame | None = None,
    scope: str = "full",
) -> tuple[Path, dict]:
    source_dir = tmp_path / f"sources-{dataset_id}"
    source_dir.mkdir()
    paragraph_path, speech_path, annotation_path, exclusion_path = _write_sources(
        source_dir, paragraphs=paragraphs, exclusions=exclusions
    )
    selection_path = None
    if scope == "pilot":
        selection_path = source_dir / "pilot_keys.parquet"
        pd.read_parquet(paragraph_path)[["doc_name", "para_idx"]].iloc[:2].to_parquet(
            selection_path, index=False
        )
    root = tmp_path / "constituencies"
    manifest = C.initialize(
        dataset_id=dataset_id,
        paragraphs_path=paragraph_path,
        speeches_path=speech_path,
        speech_annotations_path=annotation_path,
        exclusions_path=exclusion_path,
        corpus_meta_path=None,
        selection_keys_path=selection_path,
        scope=scope,
        root=root,
    )
    return root, manifest


def _request(
    root: Path,
    *,
    dataset_id: str = "test-v1",
    annotator_id: str = "codex-primary",
    model_id: str = "gpt-test",
    agent_id: str | None = "agent-1",
    pass_kind: str = "primary",
    reference_run_id: str | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    batch_size: int = 5,
    context_window: int = 1,
) -> dict:
    return C.next_batch(
        dataset_id=dataset_id,
        annotator_id=annotator_id,
        model_id=model_id,
        agent_id=agent_id,
        pass_kind=pass_kind,
        reference_run_id=reference_run_id,
        shard_index=shard_index,
        shard_count=shard_count,
        batch_size=batch_size,
        context_window=context_window,
        root=root,
    )


def _none_response(request: dict) -> dict:
    response = json.loads(json.dumps(request["response_template"]))
    for label in response["labels"]:
        label["outcome"] = "none"
        label["claims"] = []
        label["unclear_reason"] = ""
        label["rationale"] = ""
    return response


def _claim(
    label: dict,
    *,
    group_text: str,
    normalized_group: str,
    group_type: str,
    relation: str,
    evidence_span: str,
    certainty: str = "explicit",
) -> None:
    label["outcome"] = "claim"
    label["claims"] = [
        {
            "group_text": group_text,
            "normalized_group": normalized_group,
            "group_type": group_type,
            "relation": relation,
            "evidence_span": evidence_span,
            "certainty": certainty,
            "notes": "",
        }
    ]
    label["unclear_reason"] = ""
    label["rationale"] = "The paragraph explicitly identifies the group."


def test_init_fingerprints_actual_canonical_text_and_is_row_order_invariant(
    tmp_path: Path,
) -> None:
    paragraphs, _, _ = _source_frames()
    _, first = _initialize(tmp_path, dataset_id="ordered", paragraphs=paragraphs)
    _, second = _initialize(
        tmp_path,
        dataset_id="shuffled",
        paragraphs=paragraphs.sample(frac=1, random_state=4),
    )
    assert first["corpus"]["n_paragraphs"] == 5
    assert first["corpus"]["key_sha256"] == second["corpus"]["key_sha256"]
    assert (
        first["corpus"]["canonical_corpus_sha256"]
        == second["corpus"]["canonical_corpus_sha256"]
    )

    changed = paragraphs.copy()
    changed.loc[changed["para_idx"].eq(2), "text"] = "Different canonical text."
    _, third = _initialize(tmp_path, dataset_id="changed", paragraphs=changed)
    assert (
        first["corpus"]["canonical_corpus_sha256"]
        != third["corpus"]["canonical_corpus_sha256"]
    )


def test_track_a_canonical_fingerprint_is_the_run_identity(tmp_path: Path) -> None:
    source_dir = tmp_path / "track-a"
    source_dir.mkdir()
    paragraph_path, speech_path, annotation_path, _ = _write_sources(source_dir)
    track_a_identity = {
        "sha256": "canonical-track-a-sha",
        "n_speeches": 2,
        "n_paragraphs": 5,
    }
    meta_path = source_dir / "meta_v1.json"
    meta_path.write_text(
        json.dumps({"canonical_corpus_fingerprint": track_a_identity})
    )
    root = tmp_path / "constituencies"
    manifest = C.initialize(
        dataset_id="track-a",
        paragraphs_path=paragraph_path,
        speeches_path=speech_path,
        speech_annotations_path=annotation_path,
        exclusions_path=None,
        corpus_meta_path=meta_path,
        root=root,
    )
    request = _request(root, dataset_id="track-a", batch_size=1)
    assert manifest["corpus"]["canonical_corpus_fingerprint"] == track_a_identity
    assert (
        request["annotation_run"]["canonical_corpus_fingerprint"]
        == track_a_identity
    )


def test_init_refuses_track_a_metadata_for_different_artifacts(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "track-a-mismatch"
    source_dir.mkdir()
    paragraph_path, speech_path, annotation_path, _ = _write_sources(source_dir)
    meta_path = source_dir / "meta_v1.json"
    meta_path.write_text(
        json.dumps(
            {
                "canonical_corpus_fingerprint": "declared-track-a-identity",
                "canonical_counts": {"speeches": 2, "paragraphs": 5},
                "output_fingerprints": {
                    paragraph_path.name: "0" * 64,
                    speech_path.name: C._sha256_file(speech_path),
                },
            }
        )
    )
    with pytest.raises(
        ValueError,
        match="upstream output fingerprint mismatch",
    ):
        C.initialize(
            dataset_id="track-a-mismatch",
            paragraphs_path=paragraph_path,
            speeches_path=speech_path,
            speech_annotations_path=annotation_path,
            exclusions_path=None,
            corpus_meta_path=meta_path,
            root=tmp_path / "constituencies",
        )


def test_init_refuses_duplicate_keys_and_declared_exclusions_in_canonical(
    tmp_path: Path,
) -> None:
    paragraphs, _, _ = _source_frames()
    duplicated = pd.concat([paragraphs, paragraphs.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate keys"):
        _initialize(tmp_path, dataset_id="duplicates", paragraphs=duplicated)

    exclusions = pd.DataFrame(
        [
            {
                "scope": "paragraph",
                "doc_name": "/speech-a",
                "para_idx": 0,
                "reason": "declared duplicate",
            }
        ]
    )
    with pytest.raises(ValueError, match="still contain declared exclusions"):
        _initialize(tmp_path, dataset_id="excluded", exclusions=exclusions)


def test_next_is_stable_machine_json_with_bounded_same_speech_context(
    tmp_path: Path,
) -> None:
    root, _ = _initialize(tmp_path)
    request = _request(root, batch_size=1, context_window=1)
    assert json.loads(json.dumps(request))["batch_id"] == request["batch_id"]
    assert request["paragraphs"][0]["target"]["doc_name"] == "/speech-a"
    assert request["paragraphs"][0]["target"]["para_idx"] == 0
    assert request["paragraphs"][0]["context_before"] == []
    assert [row["para_idx"] for row in request["paragraphs"][0]["context_after"]] == [
        1
    ]
    assert request["paragraphs"][0]["metadata"] == {
        "year": 1861,
        "president": "Abraham Lincoln",
        "title": "A Union Address",
        "speech_type": "annual_message",
    }
    resumed = _request(root, batch_size=1, context_window=1)
    assert resumed == request


def test_stable_hash_shards_are_disjoint_and_complete(tmp_path: Path) -> None:
    root, _ = _initialize(tmp_path)
    left = _request(
        root,
        annotator_id="left",
        agent_id="left",
        shard_index=0,
        shard_count=2,
        batch_size=20,
    )
    right = _request(
        root,
        annotator_id="right",
        agent_id="right",
        shard_index=1,
        shard_count=2,
        batch_size=20,
    )
    left_keys = {C._key_tuple(row["target"]) for row in left["paragraphs"]}
    right_keys = {C._key_tuple(row["target"]) for row in right["paragraphs"]}
    expected = {
        (str(row.doc_name), int(row.para_idx))
        for row in _source_frames()[0].itertuples(index=False)
    }
    assert left_keys
    assert right_keys
    assert left_keys.isdisjoint(right_keys)
    assert left_keys | right_keys == expected


def test_ingest_stamps_every_label_with_model_prompt_schema_and_corpus(
    tmp_path: Path,
) -> None:
    root, manifest = _initialize(tmp_path)
    request = _request(root, batch_size=1)
    response = _none_response(request)
    result = C.ingest(response, dataset_id="test-v1", root=root)
    assert result["n_labels"] == 1
    response_path = next(
        (root / "test-v1" / "runs").glob("*/responses/*.json")
    )
    saved = json.loads(response_path.read_text())["labels"][0]
    assert saved["annotator_id"] == "codex-primary"
    assert saved["model_id"] == "gpt-test"
    assert saved["agent_id"] == "agent-1"
    assert saved["schema_version"] == C.SCHEMA_VERSION
    assert saved["schema_sha256"] == C.SCHEMA_SHA256
    assert saved["prompt_version"] == C.PROMPT_VERSION
    assert saved["prompt_sha256"] == C.PROMPT_SHA256
    assert (
        saved["canonical_corpus_fingerprint"]
        == manifest["corpus"]["canonical_corpus_fingerprint"]
    )
    assert saved["labeling_input_sha256"] == manifest["corpus"][
        "labeling_input_sha256"
    ]
    with pytest.raises(ValueError, match="already ingested"):
        C.ingest(response, dataset_id="test-v1", root=root)


def test_ingest_validates_exact_target_evidence_not_neighbor_context(
    tmp_path: Path,
) -> None:
    root, _ = _initialize(tmp_path)
    request = _request(root, batch_size=1, context_window=1)
    response = _none_response(request)
    _claim(
        response["labels"][0],
        group_text="Their rights",
        normalized_group="loyal citizens",
        group_type="national_public",
        relation="protected_group",
        evidence_span="Their rights must be protected by the Government.",
        certainty="context_resolved",
    )
    with pytest.raises(ValueError, match="group_text is not an exact target substring"):
        C.ingest(response, dataset_id="test-v1", root=root)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda label: label.update(outcome="bogus"),
            "unknown outcome",
        ),
        (
            lambda label: label.update(outcome="claim", claims=[]),
            "requires at least one claim",
        ),
        (
            lambda label: label.update(
                outcome="unclear", claims=[], unclear_reason=""
            ),
            "requires unclear_reason",
        ),
    ],
)
def test_ingest_refuses_outcome_schema_violations(
    tmp_path: Path, mutate, message: str
) -> None:
    root, _ = _initialize(tmp_path)
    response = _none_response(_request(root, batch_size=1))
    mutate(response["labels"][0])
    with pytest.raises(ValueError, match=message):
        C.ingest(response, dataset_id="test-v1", root=root)


def test_ingest_refuses_invalid_enums_and_incomplete_batches(
    tmp_path: Path,
) -> None:
    root, _ = _initialize(tmp_path)
    request = _request(root, batch_size=2)
    incomplete = _none_response(request)
    incomplete["labels"] = incomplete["labels"][:1]
    with pytest.raises(ValueError, match="incomplete or unassigned"):
        C.ingest(incomplete, dataset_id="test-v1", root=root)

    complete = _none_response(request)
    _claim(
        complete["labels"][0],
        group_text="loyal citizens",
        normalized_group="loyal citizens",
        group_type="invented_type",
        relation="protected_group",
        evidence_span="the loyal citizens",
    )
    with pytest.raises(ValueError, match="unknown group_type"):
        C.ingest(complete, dataset_id="test-v1", root=root)


def test_primary_claiming_prevents_cross_agent_collisions(tmp_path: Path) -> None:
    root, _ = _initialize(tmp_path)
    first = _request(
        root, annotator_id="one", agent_id="one", batch_size=1
    )
    second = _request(
        root, annotator_id="two", agent_id="two", batch_size=1
    )
    assert C._key_tuple(first["paragraphs"][0]["target"]) != C._key_tuple(
        second["paragraphs"][0]["target"]
    )


def test_independent_agreement_repeats_reference_keys_without_labels(
    tmp_path: Path,
) -> None:
    root, _ = _initialize(tmp_path)
    primary = _request(root, batch_size=2)
    C.ingest(_none_response(primary), dataset_id="test-v1", root=root)
    reference_run_id = primary["annotation_run"]["annotation_run_id"]

    agreement = _request(
        root,
        annotator_id="codex-second",
        model_id="gpt-test-2",
        agent_id="agent-2",
        pass_kind="agreement",
        reference_run_id=reference_run_id,
        batch_size=20,
    )
    primary_keys = {
        C._key_tuple(label) for label in primary["response_template"]["labels"]
    }
    agreement_keys = {
        C._key_tuple(row["target"]) for row in agreement["paragraphs"]
    }
    assert agreement_keys == primary_keys
    assert "reference_labels" not in agreement
    assert all("outcome" not in row for row in agreement["paragraphs"])


def test_corpus_prompt_and_schema_drift_refuse_before_more_work(
    tmp_path: Path,
) -> None:
    root, _ = _initialize(tmp_path)
    manifest_path = root / "test-v1" / "dataset.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["prompt"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="prompt-hash drift"):
        _request(root)

    root2, _ = _initialize(tmp_path, dataset_id="corpus-drift")
    source_path = Path(
        json.loads((root2 / "corpus-drift" / "dataset.json").read_text())[
            "source_paths"
        ]["paragraphs"]
    )
    changed = pd.read_parquet(source_path)
    changed.loc[0, "text"] = "Changed after initialization."
    changed.to_parquet(source_path, index=False)
    with pytest.raises(ValueError, match="canonical corpus fingerprint drift"):
        _request(root2, dataset_id="corpus-drift")


def test_status_audit_export_and_disagreement_workflow(tmp_path: Path) -> None:
    root, _ = _initialize(tmp_path)
    primary = _request(root, batch_size=20)
    primary_response = _none_response(primary)
    _claim(
        primary_response["labels"][0],
        group_text="loyal citizens",
        normalized_group="loyal citizens",
        group_type="national_public",
        relation="represented_constituency",
        evidence_span="the loyal citizens of this Union",
    )
    C.ingest(primary_response, dataset_id="test-v1", root=root)
    primary_run_id = primary["annotation_run"]["annotation_run_id"]

    current = C.status(dataset_id="test-v1", root=root)
    assert current["primary_completed"] == 5
    assert current["primary_missing"] == 0
    assert current["runs"][0]["by_era"]
    assert current["runs"][0]["by_genre"]
    assert current["runs"][0]["by_shard"] == [
        {"shard_id": "0-of-1", "eligible": 5, "completed": 5}
    ]
    assert C.audit(dataset_id="test-v1", root=root)["status"] == "ok"

    exported = C.export(dataset_id="test-v1", root=root)
    labels = pd.read_parquet(exported["labels"])
    summary = pd.read_parquet(exported["summary"])
    assert len(summary) == 5
    assert len(labels) == 5
    assert set(summary["model_id"]) == {"gpt-test"}

    agreement = _request(
        root,
        annotator_id="independent",
        model_id="gpt-independent",
        agent_id="agent-2",
        pass_kind="agreement",
        reference_run_id=primary_run_id,
        batch_size=20,
    )
    agreement_response = _none_response(agreement)
    C.ingest(agreement_response, dataset_id="test-v1", root=root)
    agreement_run_id = agreement["annotation_run"]["annotation_run_id"]
    result = C.disagreements(
        dataset_id="test-v1",
        left_run_id=primary_run_id,
        right_run_id=agreement_run_id,
        root=root,
    )
    assert result["overlap"] == 5
    assert result["n_disagreements"] == 1
    assert result["disagreements"][0]["reasons"] == [
        "outcome",
        "claim_semantics",
        "evidence_or_certainty",
    ]


def test_export_refuses_partial_pilot_and_unadjudicated_unclear(
    tmp_path: Path,
) -> None:
    root, _ = _initialize(tmp_path, scope="pilot")
    request = _request(root, batch_size=20)
    C.ingest(_none_response(request), dataset_id="test-v1", root=root)
    with pytest.raises(ValueError, match="full-scope"):
        C.export(dataset_id="test-v1", root=root)

    root2, _ = _initialize(tmp_path, dataset_id="unclear")
    request2 = _request(root2, dataset_id="unclear", batch_size=20)
    response2 = _none_response(request2)
    response2["labels"][0].update(
        outcome="unclear",
        claims=[],
        unclear_reason="The referent of we cannot be resolved.",
    )
    C.ingest(response2, dataset_id="unclear", root=root2)
    audit = C.audit(dataset_id="unclear", root=root2)
    assert audit["status"] == "failed"
    assert audit["unclear_labels"] == 1
    with pytest.raises(ValueError, match="unadjudicated unclear"):
        C.export(dataset_id="unclear", root=root2)


def test_module_has_no_api_or_network_client_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(C.__file__).read_text()
    assert "anthropic" not in source.casefold()
    assert "requests" not in source
    assert "socket" not in source

    original_import = builtins.__import__
    anthropic_modules_before = {
        name for name in sys.modules if name == "anthropic" or name.startswith("anthropic.")
    }

    def guarded_import(name, *args, **kwargs):
        if name == "anthropic" or name.startswith("anthropic."):
            raise AssertionError("offline workflow imported an API client")
        return original_import(name, *args, **kwargs)

    def refuse_network(*args, **kwargs):
        raise AssertionError("offline workflow attempted a network connection")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket.socket, "connect", refuse_network)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse_network)
    monkeypatch.setattr(socket, "create_connection", refuse_network)

    root, _ = _initialize(tmp_path, dataset_id="offline")
    request = _request(root, dataset_id="offline", batch_size=20)
    C.ingest(_none_response(request), dataset_id="offline", root=root)
    assert C.audit(dataset_id="offline", root=root)["status"] == "ok"
    anthropic_modules_after = {
        name for name in sys.modules if name == "anthropic" or name.startswith("anthropic.")
    }
    assert anthropic_modules_after == anthropic_modules_before
