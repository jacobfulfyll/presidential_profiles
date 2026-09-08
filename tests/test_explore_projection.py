"""Governed publication contracts for Explore v2."""
from __future__ import annotations

import csv
import gzip
import io
import json
from collections import Counter

import pytest

from presidential_profiles import corpus, explore_assets, explore_projection as E, explorer


def _payload(bundle, filename):
    return json.loads(bundle.files[filename])


def _exact_record(bundle, query):
    filename = E.lexical_shard_path("lexical-exact-unigram", query)
    return _payload(bundle, filename)["records"][query]


def test_two_full_corpus_builds_are_byte_identical(explore_projection_pair):
    first, second = explore_projection_pair
    assert first.files == second.files


def test_projection_has_exact_deterministic_inventory(explore_projection_bundle):
    bundle = explore_projection_bundle

    assert len(bundle.files) == 263
    assert len(bundle.manifest["inventory"]) == 262
    assert bundle.index["schema_version"] == E.INDEX_SCHEMA
    assert bundle.families["schema_version"] == E.FAMILY_SCHEMA
    assert bundle.topics["schema_version"] == E.TOPIC_SCHEMA
    assert bundle.manifest["schema_version"] == E.MANIFEST_SCHEMA
    for directory in E.LEXICAL_NAMESPACES.values():
        assert {
            name for name in bundle.files if name.startswith(directory + "/")
        } == {f"{directory}/{bucket:02x}.json" for bucket in range(64)}


def test_hashing_uses_declared_fnv_vectors_and_not_first_letter_shards(
    explore_projection_bundle,
):
    assert E.fnv1a_bucket("tariff") == "17"
    assert E.fnv1a_bucket("freedom") == "11"
    assert E.fnv1a_bucket("border") == "37"
    assert explore_projection_bundle.index["sharding"]["algorithm"] == (
        "fnv1a32_utf8_low_6_bits"
    )
    assert E.lexical_shard_path("lexical-exact-unigram", "tariff") == (
        "lexical/eu/17.json"
    )


def test_lexical_axis_preserves_calendar_windows_and_unknowns(
    explore_projection_bundle,
):
    axis = explore_projection_bundle.index["lexical_axis"]

    assert len(axis) == 236
    assert sum(row["value_status"] == "observed_or_zero" for row in axis) == 226
    assert {row["x"] for row in axis}.isdisjoint({1946, 1950})
    assert all(row["period_start"] == row["x"] - 2 for row in axis)
    assert all(row["period_end"] == row["x"] + 2 for row in axis)
    assert all(
        (row["denominator_words"] > 20_000)
        == (row["value_status"] == "observed_or_zero")
        for row in axis
    )


def test_tariff_fixed_rates_match_an_independent_corpus_recount(
    explore_projection_bundle,
):
    frame = corpus.load()
    words_by_year = Counter()
    tariff_by_year = Counter()
    for year, transcript in frame[["year", "transcript"]].itertuples(
        index=False, name=None
    ):
        tokens = E.WORD_RE.findall(str(transcript).lower())
        words_by_year[int(year)] += len(tokens)
        tariff_by_year[int(year)] += tokens.count("tariff")
    decoded = E.decode_lexical_record(
        _exact_record(explore_projection_bundle, "tariff"),
        explore_projection_bundle.index["lexical_axis"],
    )
    row = next(value for value in decoded if value["x"] == 1900)
    numerator = sum(tariff_by_year[year] for year in range(1898, 1903))
    denominator = sum(words_by_year[year] for year in range(1898, 1903))

    assert row["numerator_count"] == numerator
    assert row["denominator_words"] == denominator
    assert row["display_value"] == round(
        numerator / denominator * 10_000 * E.RATE_FIXED_SCALE
    ) / E.RATE_FIXED_SCALE


def test_sparse_lexical_decode_distinguishes_zero_from_unknown(
    explore_projection_bundle,
):
    rows = E.decode_lexical_record(
        _exact_record(explore_projection_bundle, "tariff"),
        explore_projection_bundle.index["lexical_axis"],
    )

    zero = next(row for row in rows if row["value_status"] == "observed_zero")
    unknown = next(row for row in rows if row["value_status"] == "unknown_low_words")
    assert zero["display_value"] == 0
    assert zero["numerator_count"] == 0
    assert unknown["display_value"] is None
    assert unknown["numerator_count"] is None


def test_catalog_uses_governed_taxonomies_and_order(explore_projection_bundle):
    catalog = explore_projection_bundle.index["catalog"]

    assert [item["label"] for item in catalog["broad_issues"]] == [
        "Economy & jobs",
        "Taxes & budget",
        "War & military",
        "Foreign policy",
        "Immigration",
        "Civil rights & race",
        "Health care",
        "Education",
        "Crime & justice",
        "Energy & environment",
        "Trade & tariffs",
        "Agriculture",
        "Religion & values",
        "Money & banking",
        "Infrastructure",
        "Security & peace",
    ]
    assert len(catalog["detailed_topic_groups"]) == 17
    assert sum(len(group["topics"]) for group in catalog["detailed_topic_groups"]) == 50
    assert [item["query"] for item in catalog["acronyms"]] == [
        "SALT",
        "START",
        "AIDS",
    ]


def test_family_index_is_field_preserving_and_names_algorithmic_limit(
    explore_projection_bundle,
):
    payload = explore_projection_bundle.families

    assert payload["resolver"]["slaves"] == payload["resolver"]["slave"]
    node = payload["resolver"]["slaves"]
    family = payload["families"][node]
    assert family["forms"][0] == node
    assert family["form_count"] == len(family["forms"])
    assert "linguistic equivalence" in payload["algorithm"]["claim_boundary"]
    assert payload["normalizations"]["to-day"] == "today"
    assert payload["algorithm"]["source"] == "data/word_families.json"


def test_topic_projection_preserves_every_governed_band_field(
    explore_projection_bundle,
):
    topic = explore_projection_bundle.topics
    rows = topic["series"]

    assert len(rows) == 66
    assert sum(len(value) for value in rows.values()) == 1_234
    assert set(topic["field_order"]) == set(E.bands.BANDS_COLUMNS)
    assert {len(value) for key, value in rows.items() if key.startswith("corex:")} == {49}
    assert {len(value) for key, value in rows.items() if key.startswith("llm:")} == {9}
    source = E.pd.read_parquet(E.bands.BANDS_PATH)
    projected = rows["corex:Agriculture"][1]
    original = source.loc[
        source["surface"].eq("corex_issues")
        & source["series"].eq("Agriculture")
        & source["period"].eq(projected["period"])
    ].iloc[0]
    for field in E.bands.BANDS_COLUMNS:
        assert projected[field] == E._scalar(original[field])


def test_both_band_trust_gates_and_null_bounds_remain_distinct(
    explore_projection_bundle,
):
    row = explore_projection_bundle.topics["series"]["corex:Agriculture"][0]

    assert row["ci_status"] == "low_cluster_caution"
    assert row["interval_unresolvable"] is True
    assert row["point"] == 0
    assert all(row[field] is None for field in ("lo_sampling", "hi_sampling", "lo", "hi"))
    assert row["status"] == "interval_unresolvable"
    assert "not resolvable" in row["interval_label"].lower()
    assert explore_projection_bundle.topics["trust_gates"] == {
        "period_grain": "ci_status",
        "cell_grain": "interval_unresolvable",
        "null_bounds": "unknown_not_zero",
    }


def test_ai_ranges_preserve_disagreement_components(explore_projection_bundle):
    series_id = next(
        key for key in explore_projection_bundle.topics["series"] if key.startswith("llm:")
    )
    rows = explore_projection_bundle.topics["series"][series_id]

    assert {row["ci_components"] for row in rows} == {
        "sampling+annotator_disagreement"
    }
    assert all(row["disagreement_band_applied"] for row in rows)
    assert all(row["n_paired_paragraphs"] >= 50 for row in rows)


def test_csv_has_exact_order_and_formula_injection_protection(
    explore_projection_bundle,
):
    text = explore_projection_bundle.files[E.DEFAULT_VALUES_FILE].decode()
    reader = csv.DictReader(io.StringIO(text))

    assert tuple(reader.fieldnames) == E.SELECTED_CSV_FIELDS
    rows = list(reader)
    assert len(rows) == 3 * 236
    assert [(row["selection_order"], row["series_label"]) for row in rows[::236]] == [
        ("1", "tariff"),
        ("2", "freedom"),
        ("3", "border"),
    ]
    malicious = E._csv_bytes([{"value": "=HYPERLINK(1)"}], ["value"]).decode()
    assert "'=HYPERLINK(1)" in malicious


def test_manifest_receipts_and_real_size_budgets(explore_projection_bundle):
    bundle = explore_projection_bundle
    raw = sum(len(value) for value in bundle.files.values())
    compressed = sum(
        receipt["gzip_bytes"] for receipt in bundle.manifest["inventory"].values()
    ) + len(gzip.compress(bundle.files[E.MANIFEST_FILE], compresslevel=9, mtime=0))
    shard_receipts = [
        receipt
        for name, receipt in bundle.manifest["inventory"].items()
        if name.startswith("lexical/")
    ]

    assert raw <= 90_000_000
    assert compressed <= 23_000_000
    assert max(receipt["bytes"] for receipt in shard_receipts) <= 600_000
    assert max(receipt["gzip_bytes"] for receipt in shard_receipts) <= 150_000
    for name, receipt in bundle.manifest["inventory"].items():
        assert receipt["bytes"] == len(bundle.files[name])
        assert receipt["sha256"] == E._sha256_bytes(bundle.files[name])


def test_initial_page_budget_needs_no_plotly_and_at_most_seven_requests(
    explore_projection_bundle,
):
    page = explorer.render_page(explore_projection_bundle).encode()
    css = explore_assets.EXPLORE_CSS.encode()
    script = explore_assets.EXPLORE_JS.encode()
    index = explore_projection_bundle.files[E.INDEX_FILE]
    defaults = [
        explore_projection_bundle.files[
            E.lexical_shard_path("lexical-exact-unigram", query)
        ]
        for query in ("tariff", "freedom", "border")
    ]
    payloads = [page, css, script, index, *defaults]

    assert len(payloads) == 7
    assert sum(map(len, payloads)) <= 1_500_000
    assert sum(len(gzip.compress(value, compresslevel=9, mtime=0)) for value in payloads) <= 350_000
    assert b"plotly" not in page.lower()


def test_atomic_publication_replaces_only_explorer_directory(
    tmp_path, explore_projection_bundle
):
    site = tmp_path / "docs"
    target = site / "explorer"
    target.mkdir(parents=True)
    (target / "stale.json").write_text("stale")
    protected = site / "protected.txt"
    protected.write_text("unchanged")

    E.write_public_projection(explore_projection_bundle, site)

    actual = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file()
    }
    assert actual == set(explore_projection_bundle.files)
    assert protected.read_text() == "unchanged"
    assert not (target / "stale.json").exists()


def test_source_set_change_is_detected(explore_projection_bundle):
    changed = json.loads(json.dumps(explore_projection_bundle.manifest))
    changed["source_identity"]["contract_version"] = "stale"

    assert not E.source_identity_matches(changed)


def test_unknown_namespace_refuses_instead_of_guessing():
    with pytest.raises(E.ExploreProjectionError, match="unknown lexical namespace"):
        E.lexical_shard_path("invented", "tariff")
