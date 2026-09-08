"""Deterministic, field-preserving public projection for Explore v2.

The browser is a renderer for this publication.  It may decode fixed-point
values, filter catalog rows, and serialize already-published records, but it
does not calculate rates, rolling windows, joins, support states, confidence
intervals, or disagreement envelopes.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import bands, corpus, issues, topic_quality, trends, word_families


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DIR_NAME = "explorer"
PUBLIC_DIR = REPO_ROOT / "docs" / PUBLIC_DIR_NAME

CONTRACT_VERSION = "explore-interaction-and-evidence-v1"
MANIFEST_SCHEMA = "explore-publication-manifest-v2"
INDEX_SCHEMA = "explore-index-v2"
FAMILY_SCHEMA = "explore-family-index-v2"
TOPIC_SCHEMA = "explore-topic-values-v2"
LEXICAL_SCHEMA = "explore-lexical-shard-v2"

MANIFEST_FILE = "manifest_v2.json"
INDEX_FILE = "index_v2.json"
FAMILY_FILE = "families_v2.json"
TOPIC_FILE = "topic_values_v2.json"
DEFAULT_VALUES_FILE = "default_values_v2.csv"
TOPIC_VALUES_FILE = "topic_values_v2.csv"
SERIES_CATALOG_FILE = "series_catalog_v2.csv"

LEXICAL_NAMESPACES = {
    "lexical-exact-unigram": "lexical/eu",
    "lexical-exact-bigram": "lexical/eb",
    "lexical-family-unigram": "lexical/fu",
    "lexical-family-bigram": "lexical/fb",
}
SHARD_COUNT = 64
WINDOW_YEARS = 5
MIN_WINDOW_WORDS = 20_000
MIN_UNIGRAM = 30
MIN_BIGRAM = 15
SELECTION_LIMIT = 6
RATE_FIXED_SCALE = 10_000

LEXICAL_SHARD_RAW_MAX = 600_000
LEXICAL_SHARD_GZIP_MAX = 150_000
PROJECTION_RAW_MAX = 90_000_000
PROJECTION_GZIP_MAX = 23_000_000

WORD_RE = re.compile(r"[a-z']+")
_ACRONYM_RES = {
    token: re.compile(rf"\b{re.escape(token)}\b")
    for token in word_families.ACRONYMS
}

PERIOD_KEYS = (
    "founding",
    "expansion",
    "civil-war",
    "gilded-age",
    "progressives-depression",
    "war-new-deal",
    "cold-war",
    "post-cold-war",
    "present",
)

PRESETS: dict[str, dict[str, Any]] = {
    "crisis-language": {
        "label": "Crisis language",
        "question": "When does explicit crisis language rise and fall?",
        "words": ["crisis", "emergency", "threat", "danger"],
        "rationale": "Declared emergency and danger terms.",
        "ambiguities": "Crisis can describe an event without endorsing alarm.",
    },
    "superlative-politics": {
        "label": "Superlative politics",
        "question": "When do presidents use more extremal praise or condemnation?",
        "words": ["greatest", "best", "worst", "ever"],
        "rationale": "Extremal praise and condemnation.",
        "ambiguities": "Ever can be temporal rather than superlative.",
    },
    "legal-procedural": {
        "label": "Legal and procedural vocabulary",
        "question": "How does formal governing vocabulary change over time?",
        "words": ["act", "bill", "treaty", "law", "section", "appropriation"],
        "rationale": "Formal public wording about governing instruments.",
        "ambiguities": "Act and bill also have everyday meanings; this does not measure competence or policy depth.",
    },
    "national-unity": {
        "label": "National unity",
        "question": "When does explicit collective-unity language become more common?",
        "words": ["unity", "united", "together", "common"],
        "rationale": "Explicit collective-unity language.",
        "ambiguities": "United often occurs inside the country name.",
    },
    "decline-restoration": {
        "label": "Decline and restoration",
        "question": "When does rhetoric describe deterioration and return?",
        "words": ["decline", "restore", "again", "lost"],
        "rationale": "Language of deterioration and return.",
        "ambiguities": "Again and lost frequently have nonpolitical uses.",
    },
    "war-peace": {
        "label": "War and peace",
        "question": "How do direct war and peace terms move through the record?",
        "words": ["war", "peace", "military", "conflict"],
        "rationale": "Direct armed-conflict and peace vocabulary.",
        "ambiguities": "War is also used metaphorically.",
    },
    "economic-hardship": {
        "label": "Economic hardship",
        "question": "When does concrete economic-hardship language become more prominent?",
        "words": ["unemployment", "poverty", "inflation", "hardship"],
        "rationale": "Concrete hardship and price-pressure terms.",
        "ambiguities": "Words do not distinguish diagnosis from claimed improvement.",
    },
    "immigration": {
        "label": "Immigration",
        "question": "How does migration and border vocabulary change over time?",
        "words": ["immigration", "immigrant", "border", "alien"],
        "rationale": "Migration, border, and historical legal terminology.",
        "ambiguities": "Border and alien have non-immigration senses.",
    },
    "democratic-institutions": {
        "label": "Democratic institutions",
        "question": "When do electoral and constitutional terms become more common?",
        "words": ["democracy", "constitution", "election", "vote", "congress"],
        "rationale": "Electoral and constitutional institutions.",
        "ambiguities": "Democratic may refer to the political party.",
    },
}

SELECTED_CSV_FIELDS = (
    "selection_order",
    "series_id",
    "series_label",
    "series_kind",
    "instrument",
    "query",
    "grouping_active",
    "family_label",
    "family_form_count",
    "family_forms",
    "period_kind",
    "period",
    "period_order",
    "period_start",
    "period_end",
    "x",
    "display_value",
    "display_unit",
    "numerator_count",
    "denominator_count",
    "denominator_unit",
    "point",
    "lo_sampling",
    "hi_sampling",
    "lo",
    "hi",
    "n_paragraphs",
    "n_speeches",
    "n_paired_paragraphs",
    "ci_status",
    "interval_unresolvable",
    "disagreement_band_applied",
    "disagreement_half_width",
    "ci_components",
    "disagreement_status",
    "agreement_source",
    "value_status",
    "source_scope",
)


class ExploreProjectionError(RuntimeError):
    """The Explore projection failed closed."""


@dataclass(frozen=True)
class ExploreProjectionBundle:
    index: dict[str, Any]
    families: dict[str, Any]
    topics: dict[str, Any]
    manifest: dict[str, Any]
    files: dict[str, bytes]


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return _sha256_bytes(path.read_bytes())


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    unsigned = {key: item for key, item in value.items() if key != field}
    return _sha256_bytes(_canonical_json(unsigned).encode("utf-8"))


def _gzip_size(value: bytes) -> int:
    return len(gzip.compress(value, compresslevel=9, mtime=0))


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        raise ExploreProjectionError("projection contains a non-finite value")
    return value


def _file_receipt(value: bytes, schema: str, rows: int) -> dict[str, Any]:
    return {
        "schema_version": schema,
        "rows": int(rows),
        "bytes": len(value),
        "gzip_bytes": _gzip_size(value),
        "sha256": _sha256_bytes(value),
    }


def _safe_csv_cell(value: Any) -> Any:
    """Prevent spreadsheet formula execution without changing numeric cells."""
    value = _scalar(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return "" if value is None else value


def _csv_bytes(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _safe_csv_cell(row.get(field)) for field in fields})
    return stream.getvalue().encode("utf-8")


def fnv1a_bucket(value: str) -> str:
    """FNV-1a over UTF-8, addressed by the low six bits."""
    hashed = 2_166_136_261
    for byte in value.encode("utf-8"):
        hashed = ((hashed ^ byte) * 16_777_619) & 0xFFFFFFFF
    return f"{hashed & (SHARD_COUNT - 1):02x}"


def lexical_shard_path(namespace: str, key: str) -> str:
    try:
        directory = LEXICAL_NAMESPACES[namespace]
    except KeyError as exc:
        raise ExploreProjectionError(f"unknown lexical namespace: {namespace}") from exc
    return f"{directory}/{fnv1a_bucket(key)}.json"


def _display_era(label: str) -> str:
    cleaned = label.removeprefix("The ")
    return cleaned[:1].upper() + cleaned[1:]


def _periods() -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "label": _display_era(label),
            "period_start": int(start),
            "period_end": int(end),
        }
        for key, (label, start, end) in zip(PERIOD_KEYS, trends.ERAS, strict=True)
    ]


def _source_identity() -> dict[str, Any]:
    paths = {
        "speeches.parquet": corpus.PARQUET_PATH,
        "word_families.json": word_families.FAMILIES_PATH,
        "bands.parquet": bands.BANDS_PATH,
        "bands_meta.json": bands.BANDS_META_PATH,
        "issues_meta.json": issues.ISSUES_META_PATH,
        "topic_display_names.json": topic_quality.NAMES_PATH,
        "taxonomy_v1.json": corpus.DATA_DIR / "llm_annotations" / "taxonomy_v1.json",
        "explore_projection.py": Path(__file__),
        "word_families.py": Path(word_families.__file__),
    }
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    missing = sorted(name for name, digest in hashes.items() if digest is None)
    if missing:
        raise ExploreProjectionError(f"Explore inputs are missing: {missing}")
    return {
        "contract_version": CONTRACT_VERSION,
        "schemas": {
            "manifest": MANIFEST_SCHEMA,
            "index": INDEX_SCHEMA,
            "families": FAMILY_SCHEMA,
            "topics": TOPIC_SCHEMA,
            "lexical_shard": LEXICAL_SCHEMA,
        },
        "source_files": hashes,
    }


def source_identity_matches(manifest: Mapping[str, Any]) -> bool:
    """Return whether a shipped manifest still seals every current input."""
    try:
        return manifest["source_identity"] == _source_identity()
    except (KeyError, TypeError, ExploreProjectionError):
        return False


def _load_family_contract() -> tuple[dict[str, str], dict[str, Any]]:
    payload = json.loads(word_families.FAMILIES_PATH.read_text())
    if set(payload) != {"meta", "families"}:
        raise ExploreProjectionError("word-family artifact schema drifted")
    raw = payload["families"]
    if not isinstance(raw, dict) or not isinstance(payload["meta"], dict):
        raise ExploreProjectionError("word-family artifact is malformed")
    families = {
        str(form): word_families.CANONICAL_NODE_LABELS.get(str(node), str(node))
        for form, node in raw.items()
    }
    meta = payload["meta"]
    if int(meta.get("n_forms", -1)) != len(families):
        raise ExploreProjectionError("word-family form count drifted")
    return families, meta


def _tokenize_grouped(text: str, family_map: Mapping[str, str]) -> list[str]:
    return [family_map.get(token, token) for token in word_families.tokenize(text)]


def _lexical_counts(
    speeches: pd.DataFrame,
    family_map: Mapping[str, str],
) -> tuple[
    Counter,
    dict[str, Counter],
    dict[str, Counter],
    dict[str, Counter],
    dict[str, Counter],
    dict[str, Counter],
    dict[str, Counter],
    dict[str, Counter],
    dict[str, Counter],
    dict[str, Counter],
]:
    per_year_words: Counter = Counter()
    exact_uni_total: Counter = Counter()
    exact_bi_total: Counter = Counter()
    family_uni_total: Counter = Counter()
    family_bi_total: Counter = Counter()
    exact_uni_years: dict[str, Counter] = defaultdict(Counter)
    exact_bi_years: dict[str, Counter] = defaultdict(Counter)
    family_uni_years: dict[str, Counter] = defaultdict(Counter)
    family_bi_years: dict[str, Counter] = defaultdict(Counter)
    acronym_years: dict[str, Counter] = defaultdict(Counter)

    for year_value, transcript_value in speeches[["year", "transcript"]].itertuples(
        index=False, name=None
    ):
        year = int(year_value)
        raw = str(transcript_value)
        exact_tokens = WORD_RE.findall(raw.lower())
        grouped_raw = raw
        for token in word_families.ACRONYMS:
            count = len(_ACRONYM_RES[token].findall(grouped_raw))
            if count:
                acronym_years[token][year] += count
                grouped_raw = _ACRONYM_RES[token].sub(" ", grouped_raw)
        grouped_tokens = _tokenize_grouped(grouped_raw, family_map)

        exact_bigrams = [" ".join(pair) for pair in zip(exact_tokens, exact_tokens[1:])]
        grouped_bigrams = [
            " ".join(pair) for pair in zip(grouped_tokens, grouped_tokens[1:])
        ]
        per_year_words[year] += len(exact_tokens)
        exact_uni_total.update(exact_tokens)
        exact_bi_total.update(exact_bigrams)
        family_uni_total.update(grouped_tokens)
        family_bi_total.update(grouped_bigrams)
        for token in exact_tokens:
            exact_uni_years[token][year] += 1
        for phrase in exact_bigrams:
            exact_bi_years[phrase][year] += 1
        for token in grouped_tokens:
            family_uni_years[token][year] += 1
        for phrase in grouped_bigrams:
            family_bi_years[phrase][year] += 1

    return (
        per_year_words,
        exact_uni_total,
        exact_bi_total,
        family_uni_total,
        family_bi_total,
        exact_uni_years,
        exact_bi_years,
        family_uni_years,
        family_bi_years,
        acronym_years,
    )


def _lexical_axis(per_year_words: Counter) -> list[dict[str, Any]]:
    years = sorted(int(year) for year in per_year_words)
    axis: list[dict[str, Any]] = []
    for order, year in enumerate(years):
        denominator = sum(per_year_words.get(year + offset, 0) for offset in range(-2, 3))
        available = denominator > MIN_WINDOW_WORDS
        axis.append(
            {
                "period_kind": "centered_year5",
                "period": str(year),
                "period_order": order,
                "period_start": year - 2,
                "period_end": year + 2,
                "x": year,
                "denominator_words": int(denominator),
                "value_status": "observed_or_zero" if available else "unknown_low_words",
                "status_label": (
                    "Published: centered five-year window exceeds 20,000 indexed words"
                    if available
                    else "Not published: centered five-year window has 20,000 indexed words or fewer"
                ),
            }
        )
    return axis


def _delta_encode(values: list[int]) -> list[int]:
    previous = 0
    out = []
    for index, value in enumerate(values):
        out.append(value if index == 0 else value - previous)
        previous = value
    return out


def _lexical_record(
    years: Mapping[int, int],
    corpus_total: int,
    axis: Sequence[Mapping[str, Any]],
) -> list[Any]:
    offsets: list[int] = []
    numerators: list[int] = []
    fixed_rates: list[int] = []
    for index, period in enumerate(axis):
        if period["value_status"] != "observed_or_zero":
            continue
        year = int(period["x"])
        numerator = sum(int(years.get(year + offset, 0)) for offset in range(-2, 3))
        if not numerator:
            continue
        denominator = int(period["denominator_words"])
        rate = numerator / denominator * 10_000
        offsets.append(index)
        numerators.append(numerator)
        fixed_rates.append(int(round(rate * RATE_FIXED_SCALE)))
    return [int(corpus_total), _delta_encode(offsets), numerators, fixed_rates]


def decode_lexical_record(
    record: Sequence[Any],
    axis: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Decode one compact record for server fallbacks and parity tests."""
    if len(record) != 4:
        raise ExploreProjectionError("lexical record field count drifted")
    deltas = [int(value) for value in record[1]]
    offsets: list[int] = []
    current = 0
    for index, delta in enumerate(deltas):
        current = delta if index == 0 else current + delta
        offsets.append(current)
    if len(offsets) != len(record[2]) or len(offsets) != len(record[3]):
        raise ExploreProjectionError("lexical sparse arrays have different lengths")
    sparse = {
        offset: (int(record[2][index]), int(record[3][index]))
        for index, offset in enumerate(offsets)
    }
    out = []
    for index, period in enumerate(axis):
        row = dict(period)
        if period["value_status"] == "observed_or_zero":
            numerator, fixed = sparse.get(index, (0, 0))
            row.update(
                {
                    "numerator_count": numerator,
                    "display_value": fixed / RATE_FIXED_SCALE,
                    "point": fixed / RATE_FIXED_SCALE,
                    "value_status": "observed" if numerator else "observed_zero",
                }
            )
        else:
            row.update(
                {
                    "numerator_count": None,
                    "display_value": None,
                    "point": None,
                    "value_status": "unknown_low_words",
                }
            )
        out.append(row)
    return out


def _build_lexical_shards(
    axis: Sequence[Mapping[str, Any]],
    totals_and_years: Sequence[tuple[str, Counter, Mapping[str, Counter], int]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[Any]]]:
    shards: dict[str, dict[str, Any]] = {}
    lookup: dict[str, list[Any]] = {}
    for namespace, totals, yearly, threshold in totals_and_years:
        bucket_records: dict[str, dict[str, list[Any]]] = {
            f"{bucket:02x}": {} for bucket in range(SHARD_COUNT)
        }
        for key in sorted(term for term, count in totals.items() if count >= threshold):
            record = _lexical_record(yearly[key], int(totals[key]), axis)
            bucket = fnv1a_bucket(key)
            bucket_records[bucket][key] = record
            lookup[f"{namespace}:{key}"] = record
        for bucket, records in bucket_records.items():
            filename = f"{LEXICAL_NAMESPACES[namespace]}/{bucket}.json"
            payload = {
                "schema_version": LEXICAL_SCHEMA,
                "namespace": namespace,
                "bucket": bucket,
                "record_fields": [
                    "corpus_total_count",
                    "period_offsets_delta",
                    "numerator_counts",
                    "rate_per_10k_x10000",
                ],
                "records": records,
            }
            payload["projection_sha256"] = _self_hash(payload, "projection_sha256")
            shards[filename] = payload
    return shards, lookup


def _build_family_index(
    family_map: Mapping[str, str],
    family_meta: Mapping[str, Any],
) -> dict[str, Any]:
    members: dict[str, list[str]] = defaultdict(list)
    for form, node in family_map.items():
        members[node].append(form)
    family_records = {}
    for node in sorted(members, key=lambda value: (value.casefold(), value)):
        forms = sorted(set(members[node]), key=lambda value: (value.casefold(), value))
        if node in forms:
            forms.remove(node)
            forms.insert(0, node)
        variants = sorted(
            {
                variant
                for variant, canonical in word_families.NORMALIZE.items()
                if family_map.get(canonical, canonical) == node
            },
            key=lambda value: (value.casefold(), value),
        )
        family_records[node] = {
            "display_family_name": node,
            "forms": forms,
            "form_count": len(forms),
            "normalized_spellings": variants,
        }
    payload = {
        "schema_version": FAMILY_SCHEMA,
        "algorithm": {
            "source": "data/word_families.json",
            "threshold": _scalar(family_meta.get("threshold")),
            "embed_model": family_meta.get("embed_model"),
            "built_from": family_meta.get("built_from"),
            "claim_boundary": "Algorithmic word-form grouping; not a claim of linguistic equivalence.",
        },
        "resolver": dict(sorted(family_map.items())),
        "normalizations": dict(sorted(word_families.NORMALIZE.items())),
        "families": family_records,
    }
    payload["projection_sha256"] = _self_hash(payload, "projection_sha256")
    return payload


def _catalogs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    issue_meta = json.loads(issues.ISSUES_META_PATH.read_text())
    issue_names = topic_quality.display_issues(issue_meta["issues"])
    names = topic_quality.load_names()
    broad = []
    for order, source_name in enumerate(issue_names):
        label = topic_quality.display_name(source_name, names)
        anchors = list(issue_meta["topic_words"].get(source_name, []))
        broad.append(
            {
                "series_id": f"corex:{source_name}",
                "series_kind": "corex",
                "instrument": "Deterministic CorEx issue model",
                "label": label,
                "source_label": source_name,
                "definition": "Anchored issue vocabulary: " + ", ".join(anchors[:12]) + ".",
                "search_terms": [label, source_name, *anchors],
                "display_order": order,
            }
        )

    taxonomy = json.loads(
        (corpus.DATA_DIR / "llm_annotations" / "taxonomy_v1.json").read_text()
    )
    level1_order = {entry["name"]: order for order, entry in enumerate(taxonomy["level1"])}
    grouped: dict[str, list[dict[str, Any]]] = {
        entry["name"]: [] for entry in taxonomy["level1"]
    }
    for order, entry in enumerate(taxonomy["level2"]):
        parent = entry["level1"]
        if parent not in grouped:
            raise ExploreProjectionError(f"fine topic has unknown parent: {parent}")
        grouped[parent].append(
            {
                "series_id": f"llm:{entry['name']}",
                "series_kind": "llm",
                "instrument": "Exploratory AI paragraph labels",
                "label": entry["name"],
                "source_label": entry["name"],
                "parent_label": parent,
                "definition": entry["definition"],
                "search_terms": [parent, entry["name"], entry["definition"]],
                "display_order": order,
            }
        )
    detailed = [
        {
            "label": entry["name"],
            "definition": entry["definition"],
            "display_order": level1_order[entry["name"]],
            "topics": grouped[entry["name"]],
        }
        for entry in taxonomy["level1"]
    ]
    acronyms = [
        {
            "series_id": f"acronym:{token}",
            "series_kind": "acronym",
            "instrument": "Case-sensitive exact acronym count",
            "label": label,
            "source_label": token,
            "query": token,
            "definition": f"Upper-case {token} occurrences only; lower-case lexical uses remain distinct.",
            "search_terms": [token, label],
            "display_order": order,
        }
        for order, (token, label) in enumerate(word_families.ACRONYMS.items())
    ]
    if (len(broad), sum(len(group["topics"]) for group in detailed), len(acronyms)) != (
        16,
        50,
        3,
    ):
        raise ExploreProjectionError("governed Explore catalog cardinality drifted")
    return broad, detailed, acronyms


def _interval_label(row: Mapping[str, Any]) -> str:
    if bool(row.get("interval_unresolvable")):
        return (
            f"Not resolvable — {int(row['n_speeches'])} speeches agree exactly, "
            "so the bootstrap has no width to report"
        )
    lo, hi = row.get("lo"), row.get("hi")
    if lo is None or hi is None:
        return "Not published — too few speeches to bootstrap"
    return f"{float(lo) * 100:.2f}–{float(hi) * 100:.2f}%"


def _topic_status(row: Mapping[str, Any]) -> tuple[str, str]:
    ci_status = row.get("ci_status")
    unresolved = bool(row.get("interval_unresolvable"))
    if unresolved:
        return "interval_unresolvable", "Estimate published; uncertainty range not resolvable ◇"
    if ci_status == "suppressed_n_floor":
        return "range_suppressed", "Estimate published; uncertainty range not published ‡"
    if ci_status == "low_cluster_caution":
        return "low_support", "Low support: interpret the uncertainty range cautiously †"
    if ci_status == "ok":
        return "ok", "Supported"
    raise ExploreProjectionError(f"unknown ci_status: {ci_status}")


def _build_topic_values(
    band_table: pd.DataFrame,
    broad: Sequence[Mapping[str, Any]],
    detailed: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required = set(bands.BANDS_COLUMNS)
    missing = required - set(band_table.columns)
    if missing:
        raise ExploreProjectionError(f"bands.parquet lacks fields: {sorted(missing)}")
    if band_table.duplicated(["surface", "series", "period"]).any():
        raise ExploreProjectionError("band cells are not unique")
    catalog = list(broad) + [topic for group in detailed for topic in group["topics"]]
    series_rows: dict[str, list[dict[str, Any]]] = {}
    for item in catalog:
        surface = "corex_issues" if item["series_kind"] == "corex" else "llm_topics"
        source = band_table.loc[
            band_table["surface"].eq(surface)
            & band_table["series"].eq(item["source_label"])
        ].sort_values("period_order", kind="stable")
        expected = 49 if surface == "corex_issues" else 9
        if len(source) != expected:
            raise ExploreProjectionError(
                f"{item['series_id']} has {len(source)} band rows; expected {expected}"
            )
        public = []
        for raw in source.to_dict("records"):
            copied = {field: _scalar(raw[field]) for field in bands.BANDS_COLUMNS}
            status, status_label = _topic_status(copied)
            copied.update(
                {
                    "series_id": item["series_id"],
                    "series_label": item["label"],
                    "series_kind": item["series_kind"],
                    "instrument": item["instrument"],
                    "query": item["source_label"],
                    "grouping_active": None,
                    "family_label": None,
                    "family_form_count": None,
                    "family_forms": None,
                    "display_value": (
                        None if copied["point"] is None else float(copied["point"]) * 100
                    ),
                    "display_unit": "percent of eligible paragraphs",
                    "numerator_count": None,
                    "denominator_count": copied["n_paragraphs"],
                    "denominator_unit": "eligible paragraphs",
                    "value_status": "unknown" if copied["point"] is None else "observed",
                    "interval_label": _interval_label(copied),
                    "support_label": (
                        f"{copied['n_speeches']} speeches · {copied['n_paragraphs']} paragraphs"
                    ),
                    "status": status,
                    "status_label": status_label,
                    "source_scope": (
                        "Complete source-document paragraph corpus; CorEx labels"
                        if surface == "corex_issues"
                        else "Complete source-document paragraph corpus; primary AI labels"
                    ),
                }
            )
            if copied["interval_unresolvable"] and any(
                copied[field] is not None
                for field in ("lo_sampling", "hi_sampling", "lo", "hi")
            ):
                raise ExploreProjectionError("unresolvable cell publishes numeric bounds")
            public.append(copied)
        series_rows[item["series_id"]] = public
    payload = {
        "schema_version": TOPIC_SCHEMA,
        "field_order": list(bands.BANDS_COLUMNS),
        "series": series_rows,
        "trust_gates": {
            "period_grain": "ci_status",
            "cell_grain": "interval_unresolvable",
            "null_bounds": "unknown_not_zero",
        },
    }
    payload["projection_sha256"] = _self_hash(payload, "projection_sha256")
    return payload


def _lexical_public_rows(
    *,
    selection_order: int,
    series_id: str,
    label: str,
    kind: str,
    instrument: str,
    query: str,
    grouping_active: bool,
    family_label: str | None,
    family_forms: Sequence[str] | None,
    record: Sequence[Any],
    axis: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    forms_json = None if family_forms is None else _canonical_json(list(family_forms))
    rows = []
    for row in decode_lexical_record(record, axis):
        rows.append(
            {
                "selection_order": selection_order,
                "series_id": series_id,
                "series_label": label,
                "series_kind": kind,
                "instrument": instrument,
                "query": query,
                "grouping_active": grouping_active,
                "family_label": family_label,
                "family_form_count": None if family_forms is None else len(family_forms),
                "family_forms": forms_json,
                "period_kind": row["period_kind"],
                "period": row["period"],
                "period_order": row["period_order"],
                "period_start": row["period_start"],
                "period_end": row["period_end"],
                "x": row["x"],
                "display_value": row["display_value"],
                "display_unit": "uses per 10,000 indexed words",
                "numerator_count": row["numerator_count"],
                "denominator_count": (
                    row["denominator_words"] if row["display_value"] is not None else None
                ),
                "denominator_unit": "indexed source-document words",
                "point": row["point"],
                "lo_sampling": None,
                "hi_sampling": None,
                "lo": None,
                "hi": None,
                "n_paragraphs": None,
                "n_speeches": None,
                "n_paired_paragraphs": None,
                "ci_status": "not_applicable_lexical_rate",
                "interval_unresolvable": False,
                "disagreement_band_applied": False,
                "disagreement_half_width": None,
                "ci_components": "not_applicable_lexical_rate",
                "disagreement_status": "not_applicable_lexical_rate",
                "agreement_source": "not applicable",
                "value_status": row["value_status"],
                "source_scope": "Complete source-document transcripts",
            }
        )
    return rows


def _topic_csv_rows(
    topics: Mapping[str, Any],
    catalog_order: Sequence[str],
) -> list[dict[str, Any]]:
    out = []
    for selection_order, series_id in enumerate(catalog_order, start=1):
        for source in topics["series"][series_id]:
            row = {field: source.get(field) for field in SELECTED_CSV_FIELDS}
            row["selection_order"] = selection_order
            out.append(row)
    return out


def _catalog_csv_rows(
    broad: Sequence[Mapping[str, Any]],
    detailed: Sequence[Mapping[str, Any]],
    acronyms: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for item in broad:
        rows.append(
            {
                "series_id": item["series_id"],
                "series_kind": item["series_kind"],
                "instrument": item["instrument"],
                "group": "Broad issues · deterministic model",
                "parent_group": "",
                "label": item["label"],
                "source_label": item["source_label"],
                "definition": item["definition"],
                "display_order": item["display_order"],
            }
        )
    for group in detailed:
        for item in group["topics"]:
            rows.append(
                {
                    "series_id": item["series_id"],
                    "series_kind": item["series_kind"],
                    "instrument": item["instrument"],
                    "group": "Detailed topics · AI labels",
                    "parent_group": group["label"],
                    "label": item["label"],
                    "source_label": item["source_label"],
                    "definition": item["definition"],
                    "display_order": item["display_order"],
                }
            )
    for item in acronyms:
        rows.append(
            {
                "series_id": item["series_id"],
                "series_kind": item["series_kind"],
                "instrument": item["instrument"],
                "group": "Case-sensitive named acronyms",
                "parent_group": "",
                "label": item["label"],
                "source_label": item["source_label"],
                "definition": item["definition"],
                "display_order": item["display_order"],
            }
        )
    return rows


def _validate_source_seals(
    speeches: pd.DataFrame,
    family_meta: Mapping[str, Any],
    band_table: pd.DataFrame,
) -> None:
    band_meta = json.loads(bands.BANDS_META_PATH.read_text())
    fingerprint = band_meta.get("corpus_fingerprint", {})
    paragraph_count = len(pd.read_parquet(corpus.PARAGRAPHS_PATH, columns=["doc_name"]))
    document_names = sorted(str(value) for value in speeches["doc_name"].tolist())
    document_hash = hashlib.sha256("\n".join(document_names).encode("utf-8")).hexdigest()
    expected = {
        "n_speeches": len(speeches),
        "n_paragraphs": paragraph_count,
        "doc_name_sha256": document_hash,
    }
    if fingerprint != expected:
        raise ExploreProjectionError("bands corpus fingerprint is stale")
    if family_meta.get("built_from") != f"{len(speeches)} speeches":
        raise ExploreProjectionError("word-family artifact is stale for this corpus")
    if len(band_table) != 1528:
        raise ExploreProjectionError("governed band row count drifted")


def build_projection(
    *,
    speeches: pd.DataFrame | None = None,
    band_table: pd.DataFrame | None = None,
) -> ExploreProjectionBundle:
    """Build the complete Explore publication in memory without writing."""
    speeches = corpus.load() if speeches is None else speeches.copy()
    band_table = (
        pd.read_parquet(bands.BANDS_PATH) if band_table is None else band_table.copy()
    )
    family_map, family_meta = _load_family_contract()
    _validate_source_seals(speeches, family_meta, band_table)
    broad, detailed, acronyms = _catalogs()
    counts = _lexical_counts(speeches, family_map)
    (
        per_year_words,
        exact_uni_total,
        exact_bi_total,
        family_uni_total,
        family_bi_total,
        exact_uni_years,
        exact_bi_years,
        family_uni_years,
        family_bi_years,
        acronym_years,
    ) = counts
    axis = _lexical_axis(per_year_words)
    shards, lookup = _build_lexical_shards(
        axis,
        (
            ("lexical-exact-unigram", exact_uni_total, exact_uni_years, MIN_UNIGRAM),
            ("lexical-exact-bigram", exact_bi_total, exact_bi_years, MIN_BIGRAM),
            ("lexical-family-unigram", family_uni_total, family_uni_years, MIN_UNIGRAM),
            ("lexical-family-bigram", family_bi_total, family_bi_years, MIN_BIGRAM),
        ),
    )
    acronym_records = {
        token: _lexical_record(years, sum(years.values()), axis)
        for token, years in acronym_years.items()
    }
    for item in acronyms:
        item["record"] = acronym_records.get(item["query"], [0, [], [], []])

    family_payload = _build_family_index(family_map, family_meta)
    topic_payload = _build_topic_values(band_table, broad, detailed)
    index = {
        "schema_version": INDEX_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "limits": {
            "selection_max": SELECTION_LIMIT,
            "unigram_corpus_min": MIN_UNIGRAM,
            "bigram_corpus_min": MIN_BIGRAM,
            "lexical_window_years": WINDOW_YEARS,
            "lexical_window_word_floor_exclusive": MIN_WINDOW_WORDS,
        },
        "lexical_axis": axis,
        "lexical_record_fields": [
            "corpus_total_count",
            "period_offsets_delta",
            "numerator_counts",
            "rate_per_10k_x10000",
        ],
        "fixed_point_scale": RATE_FIXED_SCALE,
        "catalog": {
            "broad_issues": broad,
            "detailed_topic_groups": detailed,
            "acronyms": acronyms,
        },
        "default_series": [
            "lexical-exact:tariff",
            "lexical-exact:freedom",
            "lexical-exact:border",
        ],
        "historical_contexts": _periods(),
        "presets": PRESETS,
        "sharding": {
            "algorithm": "fnv1a32_utf8_low_6_bits",
            "bucket_count": SHARD_COUNT,
            "namespaces": LEXICAL_NAMESPACES,
            "golden_buckets": {"tariff": "17", "freedom": "11", "border": "37"},
        },
        "downloads": {
            "default_values": DEFAULT_VALUES_FILE,
            "topic_values": TOPIC_VALUES_FILE,
            "series_catalog": SERIES_CATALOG_FILE,
        },
        "metric_ids": {
            "lexical": "explore_centered_lexical_rate",
            "corex": ["paragraph_share", "confidence_interval"],
            "llm": ["paragraph_share", "uncertainty_envelope"],
        },
    }
    index["projection_sha256"] = _self_hash(index, "projection_sha256")

    default_rows: list[dict[str, Any]] = []
    for selection_order, query in enumerate(("tariff", "freedom", "border"), start=1):
        record = lookup.get(f"lexical-exact-unigram:{query}")
        if record is None:
            raise ExploreProjectionError(f"default lexical series is unavailable: {query}")
        default_rows.extend(
            _lexical_public_rows(
                selection_order=selection_order,
                series_id=f"lexical-exact:{query}",
                label=query,
                kind="lexical-exact",
                instrument="Exact lexical form",
                query=query,
                grouping_active=False,
                family_label=None,
                family_forms=None,
                record=record,
                axis=axis,
            )
        )

    catalog_order = [item["series_id"] for item in broad]
    catalog_order.extend(
        item["series_id"] for group in detailed for item in group["topics"]
    )
    topic_csv_rows = _topic_csv_rows(topic_payload, catalog_order)
    catalog_fields = (
        "series_id",
        "series_kind",
        "instrument",
        "group",
        "parent_group",
        "label",
        "source_label",
        "definition",
        "display_order",
    )
    files: dict[str, bytes] = {
        INDEX_FILE: _json_bytes(index),
        FAMILY_FILE: _json_bytes(family_payload),
        TOPIC_FILE: _json_bytes(topic_payload),
        DEFAULT_VALUES_FILE: _csv_bytes(default_rows, SELECTED_CSV_FIELDS),
        TOPIC_VALUES_FILE: _csv_bytes(topic_csv_rows, SELECTED_CSV_FIELDS),
        SERIES_CATALOG_FILE: _csv_bytes(
            _catalog_csv_rows(broad, detailed, acronyms), catalog_fields
        ),
    }
    files.update({filename: _json_bytes(payload) for filename, payload in shards.items()})
    for filename, value in files.items():
        if filename.startswith("lexical/"):
            if len(value) > LEXICAL_SHARD_RAW_MAX or _gzip_size(value) > LEXICAL_SHARD_GZIP_MAX:
                raise ExploreProjectionError(f"lexical shard exceeds budget: {filename}")

    schemas = {
        INDEX_FILE: INDEX_SCHEMA,
        FAMILY_FILE: FAMILY_SCHEMA,
        TOPIC_FILE: TOPIC_SCHEMA,
        DEFAULT_VALUES_FILE: "explore-default-values-v2",
        TOPIC_VALUES_FILE: "explore-topic-values-csv-v2",
        SERIES_CATALOG_FILE: "explore-series-catalog-v2",
        **{filename: LEXICAL_SCHEMA for filename in shards},
    }
    row_counts = {
        INDEX_FILE: len(broad) + 50 + len(acronyms),
        FAMILY_FILE: len(family_payload["families"]),
        TOPIC_FILE: sum(len(value) for value in topic_payload["series"].values()),
        DEFAULT_VALUES_FILE: len(default_rows),
        TOPIC_VALUES_FILE: len(topic_csv_rows),
        SERIES_CATALOG_FILE: len(broad) + 50 + len(acronyms),
        **{
            filename: len(payload["records"])
            for filename, payload in shards.items()
        },
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "source_identity": _source_identity(),
        "inventory": {
            filename: _file_receipt(files[filename], schemas[filename], row_counts[filename])
            for filename in sorted(files)
        },
        "budgets": {
            "lexical_shard_raw_max": LEXICAL_SHARD_RAW_MAX,
            "lexical_shard_gzip_max": LEXICAL_SHARD_GZIP_MAX,
            "projection_raw_max": PROJECTION_RAW_MAX,
            "projection_gzip_max": PROJECTION_GZIP_MAX,
        },
    }
    manifest["manifest_sha256"] = _self_hash(manifest, "manifest_sha256")
    files[MANIFEST_FILE] = _json_bytes(manifest)
    bundle = ExploreProjectionBundle(index, family_payload, topic_payload, manifest, files)
    validate_projection(bundle)
    return bundle


def validate_projection(projection: ExploreProjectionBundle) -> None:
    expected = {
        MANIFEST_FILE,
        INDEX_FILE,
        FAMILY_FILE,
        TOPIC_FILE,
        DEFAULT_VALUES_FILE,
        TOPIC_VALUES_FILE,
        SERIES_CATALOG_FILE,
        *(
            f"{directory}/{bucket:02x}.json"
            for directory in LEXICAL_NAMESPACES.values()
            for bucket in range(SHARD_COUNT)
        ),
    }
    if len(expected) != 263 or set(projection.files) != expected:
        raise ExploreProjectionError("Explore publication must contain exactly 263 files")
    if projection.index.get("schema_version") != INDEX_SCHEMA:
        raise ExploreProjectionError("invalid Explore index schema")
    if projection.families.get("schema_version") != FAMILY_SCHEMA:
        raise ExploreProjectionError("invalid family-index schema")
    if projection.topics.get("schema_version") != TOPIC_SCHEMA:
        raise ExploreProjectionError("invalid topic-values schema")
    if projection.manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ExploreProjectionError("invalid Explore manifest schema")
    inventory = projection.manifest.get("inventory", {})
    if len(inventory) != 262:
        raise ExploreProjectionError("Explore manifest must inventory the other 262 files")
    for filename, receipt in inventory.items():
        value = projection.files.get(filename)
        if value is None:
            raise ExploreProjectionError(f"manifest inventories missing file: {filename}")
        if receipt["bytes"] != len(value) or receipt["sha256"] != _sha256_bytes(value):
            raise ExploreProjectionError(f"manifest receipt drift for {filename}")
    raw_total = sum(len(value) for value in projection.files.values())
    # Each deterministic-gzip size was computed while constructing the receipt.
    # The SHA check above seals the corresponding bytes, so recompressing all
    # 78 MB on every page render/publication validation adds no independent
    # evidence.  Only the uninventoried manifest itself is compressed here.
    gzip_total = sum(
        int(receipt["gzip_bytes"]) for receipt in inventory.values()
    ) + _gzip_size(projection.files[MANIFEST_FILE])
    if raw_total > PROJECTION_RAW_MAX or gzip_total > PROJECTION_GZIP_MAX:
        raise ExploreProjectionError(
            f"Explore projection exceeds aggregate budget: {raw_total}/{gzip_total}"
        )
    if projection.index["sharding"]["golden_buckets"] != {
        "tariff": "17",
        "freedom": "11",
        "border": "37",
    }:
        raise ExploreProjectionError("lexical hash golden vectors drifted")
    for namespace, directory in LEXICAL_NAMESPACES.items():
        for bucket in range(SHARD_COUNT):
            filename = f"{directory}/{bucket:02x}.json"
            payload = json.loads(projection.files[filename])
            if payload["namespace"] != namespace or payload["bucket"] != f"{bucket:02x}":
                raise ExploreProjectionError(f"lexical shard address drift for {filename}")
            for key, record in payload["records"].items():
                if fnv1a_bucket(key) != f"{bucket:02x}":
                    raise ExploreProjectionError(f"lexical key is in wrong shard: {key}")
                decode_lexical_record(record, projection.index["lexical_axis"])
    if not source_identity_matches(projection.manifest):
        raise ExploreProjectionError("Explore source-set seal changed during build")


def write_public_projection(
    projection: ExploreProjectionBundle,
    site_dir: Path,
) -> Path:
    """Atomically replace ``docs/explorer`` after complete validation."""
    validate_projection(projection)
    if not source_identity_matches(projection.manifest):
        raise ExploreProjectionError("Explore inputs changed before publication")
    target = site_dir / PUBLIC_DIR_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix="explore-projection-", dir=target.parent)
    )
    temporary = temporary_root / PUBLIC_DIR_NAME
    temporary.mkdir()
    try:
        for filename, value in projection.files.items():
            destination = temporary / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(value)
        for filename, expected in projection.files.items():
            if (temporary / filename).read_bytes() != expected:
                raise ExploreProjectionError(f"temporary projection drift for {filename}")
        backup = target.with_name(target.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(temporary, target)
        except Exception:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    validate_publication(projection, site_dir)
    return target


def validate_publication(projection: ExploreProjectionBundle, site_dir: Path) -> None:
    validate_projection(projection)
    target = site_dir / PUBLIC_DIR_NAME
    actual = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file()
    } if target.exists() else set()
    if actual != set(projection.files):
        raise ExploreProjectionError("published Explore file inventory drifted")
    for filename, expected in projection.files.items():
        if (target / filename).read_bytes() != expected:
            raise ExploreProjectionError(f"published Explore file drift for {filename}")
