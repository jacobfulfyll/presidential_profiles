"""Contract tests for the provenance-preserving canonical corpus layer."""

from __future__ import annotations

import copy
import hashlib
import json

import pandas as pd
import pytest

from presidential_profiles import corpus
from presidential_profiles import corpus_corrections as C


@pytest.fixture(scope="module")
def projection() -> C.CanonicalProjection:
    return C.build_from_paths()


def test_manifest_freezes_only_the_confirmed_anomalies() -> None:
    manifest = C.load_manifest()
    operations = [entry["operation"] for entry in manifest["corrections"]]
    assert operations.count("repeat_within_document") == 19
    assert operations.count("duplicate_document") == 4
    assert all(
        entry["review_status"] == "confirmed"
        for entry in manifest["corrections"]
    )


def test_real_projection_has_the_reviewed_canonical_universe(
    projection: C.CanonicalProjection,
) -> None:
    assert len(projection.speeches) == 1_053
    assert len(projection.paragraphs) == 35_394
    assert not projection.paragraphs.duplicated(C.KEYS).any()
    assert projection.meta["correction_counts"] == {
        "repeat_within_document": 19,
        "duplicate_document": 4,
        "paragraphs_excluded_internal_repeat": 751,
        "paragraphs_excluded_duplicate_document": 84,
        "paragraphs_requiring_reannotation": 13,
    }
    assert (
        projection.meta["annotation_projection_status"]
        == "blocked_requires_reannotation"
    )


@pytest.mark.parametrize(
    ("doc_name", "expected"),
    [
        (
            "/the-presidency/presidential-speeches/"
            "july-6-1852-eulogy-henry-clay",
            42,
        ),
        (
            "/the-presidency/presidential-speeches/"
            "october-16-1854-peoria-illinois",
            138,
        ),
    ],
)
def test_lincoln_duplicates_are_source_corrected_once(
    projection: C.CanonicalProjection,
    doc_name: str,
    expected: int,
) -> None:
    rows = projection.paragraphs[
        projection.paragraphs["doc_name"] == doc_name
    ]
    assert len(rows) == expected
    assert rows["para_idx"].tolist() == list(range(expected))


def test_lincoln_correction_changes_the_1850_1854_topic_rate(
    projection: C.CanonicalProjection,
) -> None:
    topic = "Slavery, Emancipation & Sectionalism"
    annotations = pd.read_parquet(
        C.DATA_DIR
        / "llm_annotations"
        / "paragraph_annotations.parquet"
    )
    reusable = C.project_frozen_paragraph_table(
        annotations,
        projection,
        require_complete=False,
    )
    canonical = (
        projection.paragraphs.merge(
            projection.speeches[["doc_name", "year"]],
            on="doc_name",
            validate="many_to_one",
        )
        .merge(
            reusable[["doc_name", "para_idx", "topics"]],
            on=C.KEYS,
            validate="one_to_one",
        )
    )
    canonical = canonical[canonical["year"].between(1850, 1854)]
    canonical_topic = canonical["topics"].map(
        lambda values: topic in set(values)
    )
    assert (int(canonical_topic.sum()), len(canonical)) == (183, 724)

    source = (
        pd.read_parquet(C.SOURCE_PARAGRAPHS_PATH)
        .merge(
            pd.read_parquet(C.SOURCE_SPEECHES_PATH)[["doc_name", "year"]],
            on="doc_name",
            validate="many_to_one",
        )
        .merge(
            annotations[["doc_name", "para_idx", "topics"]],
            on=C.KEYS,
            validate="one_to_one",
        )
    )
    source = source[source["year"].between(1850, 1854)]
    source_topic = source["topics"].map(lambda values: topic in set(values))
    assert (int(source_topic.sum()), len(source)) == (341, 904)


def test_duplicate_document_choices_follow_reviewed_provenance(
    projection: C.CanonicalProjection,
) -> None:
    names = set(projection.speeches["doc_name"])
    assert (
        "/the-presidency/presidential-speeches/"
        "august-8-1893-special-session-message"
    ) in names
    assert (
        "/the-presidency/presidential-speeches/"
        "august-8-1893-message-regarding-economic-crisis"
    ) not in names
    assert (
        "/the-presidency/presidential-speeches/"
        "january-26-1911-special-message-canadian-reciprocity"
    ) in names
    assert (
        "/the-presidency/presidential-speeches/"
        "january-26-1911-message-regarding-us-canadian-relations"
    ) not in names
    assert (
        "/the-presidency/presidential-speeches/"
        "september-2-1916-speech-acceptance"
    ) in names
    assert (
        "/the-presidency/presidential-speeches/"
        "september-3-1916-speech-accepting-democratic-nomination"
    ) not in names
    assert (
        "/the-presidency/presidential-speeches/"
        "july-24-1929-remarks-upon-proclaiming-treaty-renunciation-war"
    ) in names
    assert (
        "/the-presidency/presidential-speeches/"
        "july-24-1929-address-kellogg-briand-pact"
    ) not in names


def test_exact_boundary_reannotation_queue_is_frozen(
    projection: C.CanonicalProjection,
) -> None:
    expected = {
        (
            "/the-presidency/presidential-speeches/"
            "april-3-1968-press-conference",
            1,
        ),
        (
            "/the-presidency/presidential-speeches/"
            "april-7-1965-address-johns-hopkins-university",
            33,
        ),
        (
            "/the-presidency/presidential-speeches/"
            "august-27-1964-acceptance-speech-democratic-national",
            25,
        ),
        (
            "/the-presidency/presidential-speeches/"
            "december-17-1963-address-un-general-assembly",
            16,
        ),
        (
            "/the-presidency/presidential-speeches/"
            "february-29-1964-press-conference-state-department",
            45,
        ),
        (
            "/the-presidency/presidential-speeches/"
            "january-17-1968-state-union-address",
            57,
        ),
        (
            "/the-presidency/presidential-speeches/"
            "july-31-1991-press-conference-mikhail-gorbachev",
            56,
        ),
        (
            "/the-presidency/presidential-speeches/"
            "june-25-1965-remarks-20th-anniversary-un-charter",
            19,
        ),
        (
            "/the-presidency/presidential-speeches/"
            "june-3-1984-remarks-citizens-ballyporeen-ireland",
            10,
        ),
        (
            "/the-presidency/presidential-speeches/"
            "may-28-1984-remarks-honoring-vietnam-wars-unknown-soldier",
            11,
        ),
        (
            "/the-presidency/presidential-speeches/"
            "november-28-1963-thanksgiving-message",
            10,
        ),
        (
            "/the-presidency/presidential-speeches/"
            "october-18-1964-report-nation-events-china-and-ussr",
            23,
        ),
        (
            "/the-presidency/presidential-speeches/"
            "october-27-1983-speech-nation-lebanon-and-grenada",
            44,
        ),
    }
    actual = set(
        zip(
            projection.reannotation_required["doc_name"],
            projection.reannotation_required["para_idx"].astype(int),
            strict=True,
        )
    )
    assert actual == expected
    assert set(projection.reannotation_required["status"]) == {
        "changed_requires_reannotation"
    }
    assert (
        projection.key_mapping["status"] == "new_requires_reannotation"
    ).sum() == 0


def test_exclusions_audit_separates_both_duplicate_classes(
    projection: C.CanonicalProjection,
) -> None:
    counts = projection.exclusions.groupby(["scope", "reason"]).size()
    assert counts[("paragraph", "excluded_duplicate")] == 751
    assert counts[
        ("paragraph", "excluded_duplicate_document")
    ] == 84
    assert counts[("speech", "excluded_duplicate_document")] == 4


def test_full_frozen_annotation_projection_refuses_changed_text(
    projection: C.CanonicalProjection,
) -> None:
    annotations = pd.read_parquet(
        C.DATA_DIR
        / "llm_annotations"
        / "paragraph_annotations.parquet"
    )
    with pytest.raises(
        C.AnnotationProjectionBlocked,
        match="13 canonical keys require labels",
    ):
        C.project_frozen_paragraph_table(
            annotations,
            projection,
            require_complete=True,
        )
    reusable = C.project_frozen_paragraph_table(
        annotations,
        projection,
        require_complete=False,
    )
    assert len(reusable) == 35_381


def test_unknown_annotation_orphan_fails_loudly(
    projection: C.CanonicalProjection,
) -> None:
    table = pd.DataFrame(
        {
            "doc_name": ["/unknown/source"],
            "para_idx": [0],
            "label": ["x"],
        }
    )
    with pytest.raises(C.CorrectionError, match="unknown annotation orphans"):
        C.project_frozen_paragraph_table(
            table,
            projection,
            require_complete=False,
        )


def test_manifest_refuses_unknown_operation(tmp_path) -> None:
    manifest = copy.deepcopy(C.load_manifest())
    manifest["corrections"][0]["operation"] = "guess_and_delete"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(C.CorrectionError, match="Unknown correction operation"):
        C.load_manifest(path)


def test_real_source_fingerprint_drift_refuses_before_projection(
    tmp_path,
) -> None:
    manifest = copy.deepcopy(C.load_manifest())
    manifest["corrections"][0]["source_retained_fingerprint"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(
        C.CorrectionError,
        match="retained source paragraph fingerprint changed",
    ):
        C.build_from_paths(manifest_path=path)


def test_rebuild_from_immutable_sources_is_idempotent(
    projection: C.CanonicalProjection,
) -> None:
    second = C.build_from_paths()
    assert (
        second.meta["canonical_corpus_fingerprint"]
        == projection.meta["canonical_corpus_fingerprint"]
    )
    pd.testing.assert_frame_equal(second.speeches, projection.speeches)
    pd.testing.assert_frame_equal(second.paragraphs, projection.paragraphs)
    pd.testing.assert_frame_equal(
        second.key_mapping,
        projection.key_mapping,
    )


def test_written_meta_fingerprints_every_output(
    projection: C.CanonicalProjection,
    tmp_path,
) -> None:
    C.write_projection(projection, out_dir=tmp_path)
    meta = json.loads((tmp_path / "meta_v1.json").read_text())
    expected = {
        "canonical_speeches_v1.parquet",
        "canonical_paragraphs_v1.parquet",
        "canonical_paragraph_keys_v1.parquet",
        "paragraph_key_mapping_v1.parquet",
        "exclusions_v1.parquet",
        "reannotation_required_v1.parquet",
    }
    assert set(meta["output_fingerprints"]) == expected
    for basename, expected_sha256 in meta["output_fingerprints"].items():
        actual = hashlib.sha256((tmp_path / basename).read_bytes()).hexdigest()
        assert actual == expected_sha256


def test_corpus_loaders_can_select_the_built_canonical_layer(
    projection: C.CanonicalProjection,
    tmp_path,
    monkeypatch,
) -> None:
    speeches_path = tmp_path / "speeches.parquet"
    paragraphs_path = tmp_path / "paragraphs.parquet"
    projection.speeches.to_parquet(speeches_path, index=False)
    projection.paragraphs.to_parquet(paragraphs_path, index=False)
    monkeypatch.setattr(corpus, "CANONICAL_SPEECHES_PATH", speeches_path)
    monkeypatch.setattr(corpus, "CANONICAL_PARAGRAPHS_PATH", paragraphs_path)

    assert len(corpus.load(canonical=True)) == 1_053
    assert len(corpus.load_paragraphs(canonical=True)) == 35_394
