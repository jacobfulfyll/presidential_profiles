"""Render the reader-facing article explaining the AI annotation pipeline.

The page has three reading depths: a short trust verdict, a worked paragraph
and visual data flow, and an evidence-rich technical layer. Every quantitative
figure has an HTML table or prose equivalent, so the method remains legible
without JavaScript or an external chart runtime.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import pandas as pd

from . import attention, corpus, grammar, quality_audit, trust_appendices
from .figures import REPO_ROOT
from .site_style import PAGE_CSS


AGREEMENT_PATH = REPO_ROOT / "data" / "llm_annotations" / "agreement_v1.parquet"
FOUNDATION_META_PATH = REPO_ROOT / "data" / "speaker_views" / "meta_v1.json"
FOUNDATION_VIEW_PATH = (
    REPO_ROOT / "data" / "speaker_views" / "paragraph_view_v1.parquet"
)
AGREEMENT_SAMPLE_PATH = (
    REPO_ROOT / "data" / "llm_annotations" / "agreement_sample_v1.json"
)
QUALITY_AGREEMENT_PATH = (
    REPO_ROOT / "docs" / "data" / "quality" / "agreement_by_field_and_era.csv"
)
QUALITY_AGREEMENT_COLUMNS = (
    "field",
    "metric",
    "era_key",
    "era_label",
    "era_order",
    "value",
    "ci_low",
    "ci_high",
    "n_units",
    "support_unit",
    "n_paragraphs",
    "n_speeches",
    "prevalence_primary",
    "prevalence_second",
    "interval_status",
)
WORKED_EXAMPLE_KEY = (
    "/the-presidency/presidential-speeches/december-6-1923-first-annual-message",
    36,
)

FIELD_LABELS = {
    "topics": "Topic sets",
    "party_attack": "Partisan attack",
    "enemy_naming": "Named adversary",
    "zero_sum": "Zero-sum framing",
    "proposal_values": "Proposal / values",
    "speech_type": "Speech type",
    "audience": "Audience",
    "medium": "Medium",
    "entities": "Entity extraction",
}

METRIC_LABELS = {
    "jaccard": "mean Jaccard",
    "jaccard_normalized": "mean normalized-topic Jaccard",
    "cohen_kappa": "Cohen’s κ",
    "exact_match": "exact agreement",
    "name_match_rate": "normalized-name match",
    "stance_agreement": "matched-name stance agreement",
}

METRIC_ANCHORS = {
    "jaccard": "jaccard",
    "jaccard_normalized": "jaccard",
    "cohen_kappa": "kappa",
    "exact_match": "exact_agreement",
    "name_match_rate": "jaccard",
    "stance_agreement": "exact_agreement",
}

CORE_ERA_METRICS = (
    ("topics", "jaccard"),
    ("party_attack", "cohen_kappa"),
    ("enemy_naming", "cohen_kappa"),
    ("zero_sum", "cohen_kappa"),
    ("proposal_values", "exact_match"),
)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _agreement(ai_data: dict, key: str) -> tuple[float | None, int | None]:
    rec = ai_data["agreement"].get(key)
    return (rec["value"], rec["n"]) if rec else (None, None)


def dashboard_summary(ai_data: dict) -> str:
    """Compact methodology bridge retained for the Story generator."""
    s = ai_data["summary"]
    jaccard, n = _agreement(ai_data, "topics:jaccard")
    agreement = (
        f"{_pct(jaccard)} mean topic overlap on {n:,} paired paragraphs"
        if jaccard is not None
        else "a separately measured cross-model reproducibility pass"
    )
    return f"""<div class="ai-bridge">
  <div class="ai-bridge-stat"><strong>{s['n_paragraphs']:,}</strong><span>paragraph judgments completed</span></div>
  <div class="ai-bridge-stat"><strong>{s['n_topics']}</strong><span>corpus-derived topics</span></div>
  <div class="ai-bridge-stat"><strong>{s['n_domains']}</strong><span>broad domains</span></div>
  <div class="ai-bridge-stat"><strong>{s['mean_topics']:.3f}</strong><span>topics per paragraph</span></div>
  <p>Names and definitions were derived from era-isolated samples of this corpus, then
  frozen before the full pass. The primary model saw consecutive paragraph chunks and decade;
  president, party, and title were hidden. Cross-model reproducibility is reported,
  not mistaken for accuracy: {agreement}. <a href="methodology.html">Inspect the
  auditable labeling method →</a></p>
</div>"""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _manifest_metadata_hash(manifest: dict[str, Any]) -> str:
    semantic = {
        key: value for key, value in manifest.items() if key != "metadata_sha256"
    }
    payload = json.dumps(
        semantic,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _load_governed_quality_projection(path: Path) -> pd.DataFrame:
    """Read one public quality table only after its manifest proves its identity."""
    manifest_path = path.parent / quality_audit.MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(
            f"quality projection manifest is missing for {path.name}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("quality projection manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise ValueError("quality projection manifest must be a JSON object")
    if manifest.get("schema_version") != quality_audit.MANIFEST_SCHEMA:
        raise ValueError("quality projection manifest schema is not data-quality-v2")
    if manifest.get("contract_version") != quality_audit.CONTRACT_VERSION:
        raise ValueError("quality projection contract version is not data-quality-v2")
    if manifest.get("metadata_sha256") != _manifest_metadata_hash(manifest):
        raise ValueError("quality projection manifest metadata hash mismatch")
    if manifest.get("generation_status") != "deterministic_local_no_api_calls":
        raise ValueError("quality projection generation status is not governed")
    validations = manifest.get("validations")
    if not isinstance(validations, dict) or not validations or set(validations.values()) != {"passed"}:
        raise ValueError("quality projection manifest validations are incomplete")
    if manifest.get("units", {}).get("agreement") != "proportion_0_to_1":
        raise ValueError("quality agreement projection has the wrong unit")

    artifacts = manifest.get("artifacts")
    receipt = artifacts.get(path.name) if isinstance(artifacts, dict) else None
    if not isinstance(receipt, dict):
        raise ValueError(f"quality manifest has no artifact receipt for {path.name}")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"quality projection {path.name} is unreadable") from exc
    if receipt.get("sha256") != _sha256_bytes(payload):
        raise ValueError(f"quality projection hash mismatch for {path.name}")
    if receipt.get("bytes") != len(payload):
        raise ValueError(f"quality projection byte count mismatch for {path.name}")

    schemas = manifest.get("table_schemas")
    schema = schemas.get(path.name) if isinstance(schemas, dict) else None
    if not isinstance(schema, dict):
        raise ValueError(f"quality manifest has no schema for {path.name}")
    if schema.get("columns") != list(QUALITY_AGREEMENT_COLUMNS):
        raise ValueError("quality agreement manifest does not declare the v2 schema")
    if receipt.get("schema_version") != quality_audit.CONTRACT_VERSION:
        raise ValueError("quality agreement artifact schema version mismatch")
    try:
        table = pd.read_csv(io.BytesIO(payload))
    except Exception as exc:  # pandas may raise several parser-specific errors
        raise ValueError(f"quality projection {path.name} is not valid CSV") from exc
    if list(table.columns) != list(QUALITY_AGREEMENT_COLUMNS):
        raise ValueError(f"quality projection schema mismatch for {path.name}")
    if receipt.get("rows") != len(table):
        raise ValueError(f"quality projection row-count mismatch for {path.name}")
    if schema.get("semantic_key") != quality_audit.PUBLIC_KEYS[
        "agreement_by_field_and_era"
    ]:
        raise ValueError("quality agreement projection has the wrong semantic key")
    return table


def _normalize_agreement_table(table: pd.DataFrame) -> pd.DataFrame:
    required = {"field", "metric", "value"}
    missing = sorted(required - set(table))
    if missing:
        raise ValueError(f"agreement evidence is missing columns {missing}")
    out = table.copy()
    if "n" not in out:
        for support_column in ("n_units", "support"):
            if support_column in out:
                out["n"] = out[support_column]
                break
    if "n" not in out:
        out["n"] = pd.NA
    if "era_label" not in out:
        out["era_label"] = "overall"
    is_story_era = "era_key" in out or "era_order" in out
    out["_era_scheme"] = "story" if is_story_era else "sampling_bin"
    if "era_bin" not in out:
        if "era_order" in out:
            out["era_bin"] = out["era_order"]
        else:
            out["era_bin"] = pd.NA
        overall = out["era_label"].astype(str).str.casefold().eq("overall")
        if "era_key" in out:
            overall |= out["era_key"].astype(str).str.casefold().eq("overall")
        out.loc[overall, "era_bin"] = -1
    aliases = {
        "ci_lower": "ci_low",
        "ci_upper": "ci_high",
        "lower": "ci_low",
        "upper": "ci_high",
    }
    for old, new in aliases.items():
        if new not in out and old in out:
            out[new] = out[old]
    for column in ("ci_low", "ci_high"):
        if column not in out:
            out[column] = math.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if "interval_status" not in out:
        out["interval_status"] = "interval_unavailable"
    have_interval = out["ci_low"].notna() & out["ci_high"].notna()
    out.loc[have_interval & out["interval_status"].eq("interval_unavailable"),
            "interval_status"] = "speech_cluster_bootstrap"
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    duplicate = out.duplicated(["field", "metric", "era_label"], keep=False)
    if duplicate.any():
        example = out.loc[duplicate, ["field", "metric", "era_label"]].head()
        raise ValueError(
            "agreement evidence has duplicate semantic keys: "
            + repr(example.to_dict("records"))
        )
    return out


def load_agreement_evidence(
    path: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load the richest available agreement table without recomputing paid labels.

    The Data Quality v2 projection may add speech-clustered intervals. Until it
    exists, the frozen v1 table remains valid point-estimate evidence and the
    renderer visibly marks every interval as unavailable.
    """
    candidate = Path(path) if path is not None else QUALITY_AGREEMENT_PATH
    if candidate.exists():
        table = _load_governed_quality_projection(candidate)
        try:
            display_path = str(candidate.relative_to(REPO_ROOT / "docs"))
        except ValueError:
            try:
                display_path = str(candidate.relative_to(REPO_ROOT))
            except ValueError:
                display_path = str(candidate)
        return _normalize_agreement_table(table), display_path
    if not AGREEMENT_PATH.exists():
        return _normalize_agreement_table(
            pd.DataFrame(columns=["field", "metric", "value"])
        ), "agreement evidence unavailable"
    return _normalize_agreement_table(pd.read_parquet(AGREEMENT_PATH)), str(
        AGREEMENT_PATH.relative_to(REPO_ROOT)
    )


def _overall_row(
    agreement: pd.DataFrame, field: str, metric: str
) -> pd.Series | None:
    metric_names = [metric, "jaccard_normalized"] if metric == "jaccard" else [metric]
    rows = agreement[
        agreement["field"].eq(field)
        & agreement["metric"].isin(metric_names)
        & (
            agreement["era_label"].astype(str).str.casefold().eq("overall")
            | agreement["era_bin"].eq(-1)
        )
    ]
    return rows.iloc[0] if len(rows) == 1 else None


def _metric_value(metric: str, value: float) -> str:
    if not math.isfinite(float(value)):
        return "not estimable"
    if metric == "cohen_kappa":
        return f"κ {value:.2f}"
    return _pct(float(value))


def _metric_interval(row: pd.Series) -> str:
    status = str(row.get("interval_status", "interval_unavailable"))
    if pd.notna(row.get("ci_low")) and pd.notna(row.get("ci_high")):
        metric = str(row["metric"])
        low = float(row["ci_low"])
        high = float(row["ci_high"])
        if metric == "cohen_kappa":
            return f"95% speech-clustered interval {low:.2f}–{high:.2f}"
        return f"95% speech-clustered interval {_pct(low)}–{_pct(high)}"
    labels = {
        "unavailable_degenerate_point": "Interval unavailable: point estimate is degenerate",
        "unavailable_degenerate_resamples": "Interval unavailable: bootstrap resamples are degenerate",
        "interval_unavailable": "Interval unavailable in the current frozen evidence",
    }
    return labels.get(status, "Interval unavailable in the current frozen evidence")


def _metric_link(metric: str) -> str:
    anchor = METRIC_ANCHORS.get(metric)
    label = html.escape(METRIC_LABELS.get(metric, metric.replace("_", " ")))
    return f'<a href="metrics.html#{anchor}">{label}</a>' if anchor else label


def _prevalence_note(row: pd.Series) -> str:
    primary = row.get("prevalence_primary")
    second = row.get("prevalence_second")
    if pd.notna(primary) and pd.notna(second):
        return (
            f"Primary positive {_pct(float(primary))}; "
            f"second-model positive {_pct(float(second))}"
        )
    return ""


def _agreement_panels(agreement: pd.DataFrame, evidence: str) -> str:
    groups = (
        (
            "Set overlap",
            "Which topics did both models select for the same paragraph?",
            (("topics", "jaccard"),),
        ),
        (
            "Binary judgments",
            "Do the two models reproduce the same yes/no rubric decisions after chance correction?",
            (
                ("party_attack", "cohen_kappa"),
                ("enemy_naming", "cohen_kappa"),
                ("zero_sum", "cohen_kappa"),
            ),
        ),
        (
            "Closed categories",
            "Do they choose the identical category when exactly one is required?",
            (
                ("proposal_values", "exact_match"),
                ("speech_type", "exact_match"),
                ("audience", "exact_match"),
                ("medium", "exact_match"),
            ),
        ),
    )
    blocks: list[str] = []
    exact_rows: list[str] = []
    for title, question, keys in groups:
        cards: list[str] = []
        for field, metric in keys:
            row = _overall_row(agreement, field, metric)
            if row is None:
                continue
            value = float(row["value"])
            n = int(row["n"]) if pd.notna(row["n"]) else None
            cards.append(
                '<div class="evaluation-card">'
                f'<p class="metric-name">{html.escape(FIELD_LABELS[field])}</p>'
                f'<strong>{html.escape(_metric_value(metric, value))}</strong>'
                f'<span>{_metric_link(metric)}</span>'
                f'<small>{html.escape(_metric_interval(row))}'
                + (f" · n={n:,}" if n is not None else "")
                + "</small>"
                + (
                    f'<small>{html.escape(_prevalence_note(row))}</small>'
                    if _prevalence_note(row) else ""
                )
                + "</div>"
            )
            support_cell = f"<td>{n:,}</td>" if n is not None else "<td>not published</td>"
            exact_rows.append(
                "<tr>"
                f"<th scope=\"row\">{html.escape(FIELD_LABELS[field])}</th>"
                f"<td>{_metric_link(metric)}</td>"
                f"<td>{html.escape(_metric_value(metric, value))}</td>"
                + support_cell
                + f"<td>{html.escape(_metric_interval(row))}</td></tr>"
            )
        if cards:
            blocks.append(
                f'<section class="metric-family"><h3>{html.escape(title)}</h3>'
                f'<p>{html.escape(question)}</p><div class="evaluation-grid">'
                + "".join(cards)
                + "</div></section>"
            )
    return f"""<figure class="evidence-figure" data-substantive-chart
 data-metric="intermodel_agreement_overall" data-evidence="{html.escape(evidence)}"
 aria-labelledby="agreement-overall-title">
<figcaption id="agreement-overall-title"><strong>One audit, several non-interchangeable measures</strong>
<span>Metric families are separated so bar height cannot masquerade as a league table.
<a href="metrics.html#intermodel_agreement_overall">Define overall cross-model reproducibility.</a></span></figcaption>
{''.join(blocks)}
<details class="evidence-table"><summary>Exact values and interval status</summary>
<div class="table-scroll" role="region" aria-label="Exact cross-model agreement values" tabindex="0"><table><thead><tr><th>Field</th><th>Measure</th><th>Estimate</th>
<th>Support</th><th>Uncertainty</th></tr></thead><tbody>{''.join(exact_rows)}</tbody></table></div>
</details></figure>"""


def _era_agreement_table(agreement: pd.DataFrame, evidence: str) -> str:
    era_rows = agreement[
        ~agreement["era_label"].astype(str).str.casefold().eq("overall")
        & agreement["era_bin"].ne(-1)
    ].copy()
    era_rows["_order"] = pd.to_numeric(era_rows["era_bin"], errors="coerce")
    is_story = bool((era_rows.get("_era_scheme") == "story").any()) if len(era_rows) else False
    eras = (
        era_rows[["era_label", "_order"]]
        .drop_duplicates()
        .sort_values(["_order", "era_label"], na_position="last")
    )
    n_eras = len(eras)
    scheme_title = (
        "Reproducibility across governed Story eras"
        if is_story else "Reproducibility across historical sampling bins"
    )
    scheme_note = (
        f"Rows use all {n_eras} chronological era definitions available in the Story-era evidence."
        if is_story
        else f"The v1 fallback uses {n_eras} fixed 30-year audit bins—not the reviewed Story eras."
    )
    era_column_label = "Story era" if is_story else "Sampling bin"
    headers = "".join(
        f"<th scope=\"col\">{html.escape(FIELD_LABELS[field])}<br>"
        f"<small>{_metric_link(metric)}</small></th>"
        for field, metric in CORE_ERA_METRICS
    )
    rows: list[str] = []
    for era in eras.itertuples(index=False):
        cells: list[str] = []
        for field, metric in CORE_ERA_METRICS:
            metric_names = (
                [metric, "jaccard_normalized"]
                if metric == "jaccard" else [metric]
            )
            match = era_rows[
                era_rows["era_label"].eq(era.era_label)
                & era_rows["field"].eq(field)
                & era_rows["metric"].isin(metric_names)
            ]
            if len(match) != 1:
                cells.append("<td><span class=\"na\">not available</span></td>")
                continue
            row = match.iloc[0]
            support = int(row["n"]) if pd.notna(row["n"]) else None
            support_text = f"n={support:,}" if support is not None else "support unavailable"
            prevalence_text = _prevalence_note(row)
            heat = max(0.0, min(100.0, float(row["value"]) * 100.0))
            cells.append(
                f'<td class="agreement-cell" style="--agreement:{heat:.3f}%">'
                f"<strong>{html.escape(_metric_value(metric, float(row['value'])))}</strong>"
                f"<small>{html.escape(support_text)}</small>"
                + (
                    f"<small>{html.escape(prevalence_text)}</small>"
                    if prevalence_text else ""
                )
                + f"<small>{html.escape(_metric_interval(row))}</small></td>"
            )
        rows.append(
            f'<tr><th scope="row">{html.escape(str(era.era_label))}</th>'
            + "".join(cells)
            + "</tr>"
        )
    if not rows:
        rows.append(
            '<tr><td colspan="6">Era-sliced agreement evidence is unavailable.</td></tr>'
        )
    container = "figure" if is_story else "section"
    semantic_attributes = (
        ' data-substantive-chart data-metric="intermodel_agreement_by_story_era"'
        f' data-evidence="{html.escape(evidence)}"'
        if is_story else ""
    )
    caption = (
        f'<figcaption id="agreement-era-title"><strong>{scheme_title} · era × field heatmap</strong>'
        f'<span>{scheme_note} <a href="metrics.html#intermodel_agreement_by_story_era">'
        'Define era-sliced reproducibility.</a></span></figcaption>'
        if is_story
        else f'<h3 id="agreement-era-title">{scheme_title}</h3><p>{scheme_note}</p>'
    )
    return f"""<{container} class="evidence-figure"{semantic_attributes}
 aria-labelledby="agreement-era-title">
{caption}
<div class="table-scroll" role="region" aria-label="Cross-model agreement by era and field" tabindex="0"><table class="era-matrix"><thead><tr><th scope="col">{era_column_label}</th>
{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
</{container}>"""


def _population_data(ai_data: dict, agreement: pd.DataFrame) -> dict[str, Any]:
    summary = ai_data["summary"]
    foundation = _read_json(FOUNDATION_META_PATH)
    sample = _read_json(AGREEMENT_SAMPLE_PATH)
    paired = _overall_row(agreement, "topics", "jaccard")
    paired_n = int(paired["n"]) if paired is not None and pd.notna(paired["n"]) else None
    model_comparison = trust_appendices.load_model_comparison()
    comparison_support = model_comparison["overall"]["n"]
    if comparison_support.nunique() != 1:
        raise ValueError("Model Comparison does not have one common paragraph population")
    model_comparison_n = int(comparison_support.iloc[0])
    comparison_population = model_comparison["population"]
    expected = None
    sample_docs = set(sample.get("doc_names", []))
    if sample_docs and corpus.PARAGRAPHS_PATH.exists():
        paragraph_keys = pd.read_parquet(
            corpus.PARAGRAPHS_PATH, columns=["doc_name", "para_idx"]
        )
        expected = int(
            paragraph_keys[paragraph_keys["doc_name"].isin(sample_docs)]
            .drop_duplicates(["doc_name", "para_idx"])
            .shape[0]
        )
    retained_documents = foundation.get("retained_documents")
    if retained_documents is None and FOUNDATION_VIEW_PATH.exists():
        retained_documents = int(
            pd.read_parquet(FOUNDATION_VIEW_PATH, columns=["doc_name"])["doc_name"]
            .nunique()
        )
    population = {
        "source_speeches": int(summary["n_speeches"]),
        "source_paragraphs": int(summary["n_paragraphs"]),
        "retained_documents": retained_documents,
        "canonical_paragraphs": foundation.get("paragraphs"),
        "eligible_paragraphs": foundation.get("eligible_presidential_paragraphs"),
        "excluded_paragraphs": foundation.get("excluded_paragraphs"),
        "cross_owner_paragraphs": foundation.get("cross_owner_presidential_paragraphs"),
        "appearances": foundation.get("appearances"),
        "sample_speeches": sample.get("n_sampled"),
        "sample_expected": expected,
        "sample_paired": paired_n,
        "model_comparison_paragraphs": model_comparison_n,
        "model_comparison_source_paragraphs": comparison_population["source_paragraphs"],
        "model_comparison_excluded_no_topic": comparison_population[
            "no_topic_paragraphs_excluded"
        ],
        "sample_missing": (
            expected - paired_n
            if expected is not None and paired_n is not None
            else None
        ),
    }
    _validate_population_data(population)
    return population


def _validate_population_data(population: dict[str, Any]) -> None:
    """Fail closed when ledger counts imply an impossible population."""
    canonical = population.get("canonical_paragraphs")
    eligible = population.get("eligible_paragraphs")
    excluded = population.get("excluded_paragraphs")
    cross_owner = population.get("cross_owner_paragraphs")
    expected = population.get("sample_expected")
    paired = population.get("sample_paired")
    comparison = population.get("model_comparison_paragraphs")
    comparison_source = population.get("model_comparison_source_paragraphs")
    comparison_excluded = population.get("model_comparison_excluded_no_topic")
    if all(value is not None for value in (canonical, eligible, excluded)):
        if int(eligible) + int(excluded) != int(canonical):
            raise ValueError("speaker population counts do not reconcile")
    if cross_owner is not None and eligible is not None and not (
        0 <= int(cross_owner) <= int(eligible)
    ):
        raise ValueError("cross-owner population falls outside eligible paragraphs")
    if expected is not None and paired is not None and int(paired) > int(expected):
        raise ValueError("paired evaluation exceeds its expected population")
    if comparison is not None and int(comparison) <= 0:
        raise ValueError("Model Comparison paragraph population must be positive")
    if all(
        value is not None
        for value in (comparison, comparison_source, comparison_excluded)
    ) and int(comparison) + int(comparison_excluded) != int(comparison_source):
        raise ValueError("Model Comparison exclusion population does not reconcile")


def _count(value: Any) -> str:
    return f"{int(value):,}" if value is not None and pd.notna(value) else "not available"


def _population_passport(pop: dict[str, Any]) -> str:
    sample_count = (
        f"{_count(pop['sample_paired'])} paired of {_count(pop['sample_expected'])} expected; "
        f"{_count(pop['sample_missing'])} unavailable"
    )
    rows = (
        (
            "Frozen label layer",
            "Source-document speech and keyed paragraph",
            f"{_count(pop['source_speeches'])} speeches; {_count(pop['source_paragraphs'])} paragraphs",
            "Taxonomy labels, reproducibility audit, and legacy document-owner summaries",
        ),
        (
            "Corrected speaker foundation",
            "Retained document, actual-speaker appearance, and attributed paragraph",
            f"{_count(pop['retained_documents'])} retained documents; {_count(pop['canonical_paragraphs'])} paragraphs; {_count(pop['appearances'])} appearances",
            "The governed population for new president-attributed work",
        ),
        (
            "Eligible actual-speaker view",
            "Paragraph attributed to exactly one controlled president",
            f"{_count(pop['eligible_paragraphs'])} eligible; {_count(pop['excluded_paragraphs'])} excluded; {_count(pop['cross_owner_paragraphs'])} cross-owner",
            "Speaker-audited Story and connection analyses; each consumer declares treatment",
        ),
        (
            "Second-model audit",
            "Same paragraph key read with the same prompt by another model",
            f"{_count(pop['sample_speeches'])} sampled speeches; {sample_count}",
            "Cross-model reproducibility only; prohibited from headline population rankings",
        ),
    )
    body = "".join(
        "<tr>"
        f"<th scope=\"row\">{html.escape(layer)}</th>"
        f"<td>{html.escape(unit)}</td><td>{html.escape(count)}</td>"
        f"<td>{html.escape(use)}</td></tr>"
        for layer, unit, count, use in rows
    )
    consumers = (
        (
            "Story",
            "index.html",
            "Mixed and chart-declared: speaker-audited era agenda/network evidence; retained legacy document-owner summaries where their receipts say so.",
        ),
        (
            "America in Summary",
            "summary.html",
            "Source-document ownership for president summaries; corpus-wide trends retain their own speech or paragraph denominator.",
        ),
        (
            "President Profiles",
            "presidents/index.html",
            "Core v3 measures use source-document owners; Connections uses eligible actual-speaker paragraphs and exposes cross-owner receipts.",
        ),
        (
            "Compare",
            "compare.html",
            "Profile v3 source-document-owner records, with thin records excluded from ranking.",
        ),
        (
            "Issue Atlas",
            "issues/index.html",
            "Corpus paragraphs keyed through source documents; it does not make a corrected president-attribution claim.",
        ),
        (
            "Explore",
            "explorer.html",
            "Source speeches and paragraphs with metric-specific denominators; it does not substitute the speaker foundation for legacy series.",
        ),
        (
            "Era Choices",
            "era-boundaries.html",
            "Source-speech and cycle aggregates; boundary sensitivity is not a president-level ranking.",
        ),
        (
            "Data Quality",
            "data-quality.html",
            "Both ownership contracts plus exclusions, cross-owner records, and the paired second-model audit.",
        ),
        (
            "Model Comparison",
            "label-models.html",
            f"{_count(pop['model_comparison_paragraphs'])} primary-LLM topic-bearing paragraphs from {_count(pop['model_comparison_source_paragraphs'])} source paragraphs; {_count(pop['model_comparison_excluded_no_topic'])} no-topic paragraphs excluded by the preregistered estimand. Separate from the {_count(pop['sample_paired'])}-paragraph second-model reproducibility audit.",
        ),
        (
            "Methods",
            "methodology.html",
            "Both population contracts, the paired audit, and governed feature-engineering evidence.",
        ),
    )
    consumer_body = "".join(
        "<tr>"
        f'<th scope="row"><a href="{html.escape(href, quote=True)}">{html.escape(label)}</a></th>'
        f"<td>{html.escape(contract)}</td></tr>"
        for label, href, contract in consumers
    )
    return f"""<figure class="evidence-figure" data-substantive-chart
 data-metric="annotation_population_lineage"
 data-evidence="data/speaker_views/meta_v1.json + data/llm_annotations/agreement_sample_v1.json"
 aria-labelledby="population-title"><figcaption id="population-title">
<strong>The population passport</strong><span>The denominator changes only when the analytical question changes.</span>
<span><a href="metrics.html#annotation_population_lineage">Define population lineage.</a></span>
</figcaption><div class="table-scroll" role="region" aria-label="Population passport layers" tabindex="0"><table><thead><tr><th>Layer</th><th>Unit</th><th>Population</th>
<th>Permitted use</th></tr></thead><tbody>{body}</tbody></table></div>
<h3>Population contract by public consumer</h3>
<p>Every analytical page is named here; mixed pages must resolve the contract at the figure receipt.</p>
<div class="table-scroll" role="region" aria-label="Population contracts by public consumer" tabindex="0"><table><thead><tr><th>Public consumer</th><th>Declared population treatment</th>
</tr></thead><tbody>{consumer_body}</tbody></table></div></figure>"""


def _entity_funnel(agreement: pd.DataFrame, evidence: str) -> str:
    names = _overall_row(agreement, "entities", "name_match_rate")
    stance = _overall_row(agreement, "entities", "stance_agreement")
    if names is None or stance is None:
        return """<figure class="evidence-figure" data-substantive-chart
 data-metric="entity_name_and_stance_reproducibility" data-evidence="agreement evidence unavailable"
 aria-labelledby="entity-funnel-title"><figcaption id="entity-funnel-title"><strong>Entity audit unavailable</strong>
<span>The page will not infer a result without matched-name evidence.
<a href="metrics.html#entity_name_and_stance_reproducibility">Define the two-stage measure.</a></span></figcaption></figure>"""
    union_n = int(names["n"])
    matched_n = int(stance["n"])
    stance_n = int(round(float(stance["value"]) * matched_n))
    return f"""<figure class="evidence-figure entity-funnel" data-substantive-chart
 data-metric="entity_name_and_stance_reproducibility" data-evidence="{html.escape(evidence)}"
 aria-labelledby="entity-funnel-title"><figcaption id="entity-funnel-title">
<strong>Entity reliability has two separate gates</strong>
<span>Finding the same normalized name is harder than judging stance after a name matches.
<a href="metrics.html#entity_name_and_stance_reproducibility">Define the two-stage measure.</a></span></figcaption>
<ol><li><span>Candidate union</span><strong>{union_n:,}</strong><small>normalized name instances produced by either model</small></li>
<li><span>Same name found</span><strong>{matched_n:,} · {_pct(float(names['value']))}</strong><small>strict name match; wording differences remain disagreements</small></li>
<li><span>Same stance</span><strong>{stance_n:,} · {_pct(float(stance['value']))}</strong><small>conditional on the {matched_n:,} matched names</small></li></ol>
<p class="figure-note">The final percentage is conditional. It must not be read as {_pct(float(stance['value']))}
agreement across all extracted names.</p></figure>"""


def _worked_example(ai_data: dict) -> dict[str, Any]:
    annotations = ai_data.get("paragraphs", pd.DataFrame()).copy()
    if annotations.empty:
        return {}
    wanted = annotations[
        annotations["doc_name"].eq(WORKED_EXAMPLE_KEY[0])
        & annotations["para_idx"].eq(WORKED_EXAMPLE_KEY[1])
    ]
    row = (wanted.iloc[0] if len(wanted) == 1 else
           annotations.sort_values(["doc_name", "para_idx"]).iloc[0])
    key = (str(row["doc_name"]), int(row["para_idx"]))
    text = "Source text unavailable in this fixture."
    request_indices = [key[1]]
    request_context: list[dict[str, Any]] = []
    request_kind = "fixture target only"
    if corpus.PARAGRAPHS_PATH.exists():
        paragraphs = pd.read_parquet(
            corpus.PARAGRAPHS_PATH, columns=["doc_name", "para_idx", "text"]
        )
        speech_paragraphs = paragraphs[paragraphs["doc_name"].eq(key[0])].sort_values(
            "para_idx"
        ).reset_index(drop=True)
        source = speech_paragraphs[speech_paragraphs["para_idx"].eq(key[1])]
        if len(source) == 1:
            text = str(source.iloc[0]["text"])
    title = "Speech metadata joined after inference"
    president = str(row.get("president", "Withheld identity"))
    year = int(row.get("year", 0))
    if corpus.PARQUET_PATH.exists():
        speeches = pd.read_parquet(
            corpus.PARQUET_PATH, columns=["doc_name", "president", "year", "title"]
        )
        source = speeches[speeches["doc_name"].eq(key[0])]
        if len(source) == 1:
            title = str(source.iloc[0]["title"])
            president = str(source.iloc[0]["president"])
            year = int(source.iloc[0]["year"])
    raw_topics = [str(value) for value in row.get("topics", [])]
    normalized_topics = attention.normalize_topics(
        raw_topics, attention.canonical_label_map(ai_data["taxonomy"])
    )
    run_id = str(row.get("run_id", "not recorded in fixture"))
    run_manifest = _read_json(
        REPO_ROOT / "data" / "llm_annotations" / "manifests" / f"{run_id}.json"
    )
    if corpus.PARAGRAPHS_PATH.exists() and not speech_paragraphs.empty:
        chunk_match = re.search(r"judgment-chunk(\d+)", run_id)
        if chunk_match:
            chunk_size = int(chunk_match.group(1))
            target_position = int(
                speech_paragraphs.index[
                    speech_paragraphs["para_idx"].eq(key[1])
                ][0]
            )
            chunk_start = (target_position // chunk_size) * chunk_size
            request_frame = speech_paragraphs.iloc[
                chunk_start:chunk_start + chunk_size
            ]
            request_kind = f"chunk-escalated request (maximum {chunk_size} paragraphs)"
        else:
            request_frame = speech_paragraphs
            request_kind = "one unchunked request for this speech"
        request_indices = request_frame["para_idx"].astype(int).tolist()
        target_position = int(
            request_frame.index[request_frame["para_idx"].eq(key[1])][0]
        )
        visible_positions = [
            position for position in (target_position - 1, target_position, target_position + 1)
            if position in request_frame.index
        ]
        for position in visible_positions:
            context_row = request_frame.loc[position]
            context_idx = int(context_row["para_idx"])
            request_context.append({
                "para_idx": context_idx,
                "role": "target" if context_idx == key[1] else "adjacent context",
                "text": str(context_row["text"]),
            })
    return {
        "doc_name": key[0],
        "para_idx": key[1],
        "text": text,
        "decade": (year // 10) * 10,
        "year": year,
        "president": president,
        "title": title,
        "topics_raw": raw_topics,
        "topics_normalized": normalized_topics,
        "party_attack": bool(row.get("party_attack", False)),
        "enemy_naming": bool(row.get("enemy_naming", False)),
        "zero_sum": bool(row.get("zero_sum", False)),
        "proposal_values": str(row.get("proposal_values", "not available")),
        "run_id": run_id,
        "model": str(run_manifest.get("model", "not available in fixture")),
        "prompt_hash": str(run_manifest.get("prompt_hash", "not available in fixture")),
        "request_indices": request_indices,
        "request_kind": request_kind,
        "request_context": request_context,
    }


def _format_indices(indices: list[int]) -> str:
    if not indices:
        return "none"
    if indices == list(range(indices[0], indices[-1] + 1)):
        return str(indices[0]) if len(indices) == 1 else f"{indices[0]}–{indices[-1]}"
    return ", ".join(map(str, indices))


def _excerpt(value: str, limit: int = 240) -> str:
    text = " ".join(value.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _worked_example_html(example: dict[str, Any]) -> str:
    if not example:
        return "<p>The worked example is unavailable because no paragraph table was supplied.</p>"
    raw_topics = ", ".join(example["topics_raw"]) or "no topic"
    normalized_topics = ", ".join(example["topics_normalized"]) or "no topic"
    context_rows = "".join(
        "<li><strong>"
        f"para_idx {int(row['para_idx'])} · {html.escape(str(row['role']))}</strong>"
        f"<span>{html.escape(_excerpt(str(row['text'])))}</span></li>"
        for row in example["request_context"]
    )
    return f"""<figure class="worked-example" data-substantive-chart
 data-metric="annotation_evidence_visibility paragraph_share"
 data-evidence="data/paragraphs.parquet + data/llm_annotations/paragraph_annotations.parquet"
 aria-labelledby="worked-title"><figcaption id="worked-title"><strong>One paragraph, end to end</strong>
<span>The identity reveal happens after the model judgment, not before it.
<a href="metrics.html#annotation_evidence_visibility">Define evidence visibility.</a></span></figcaption>
<div class="worked-flow"><div class="worked-step"><span class="step-label">1 · Source evidence</span>
<blockquote>{html.escape(example['text'])}</blockquote></div>
<div class="worked-step"><span class="step-label">2 · Masked request</span>
<p><strong>Actual request unit:</strong> {html.escape(example['request_kind'])};
{len(example['request_indices'])} consecutive analytical paragraphs with para_idx
<code>{_format_indices(example['request_indices'])}</code>, plus {example['decade']}s context.</p>
<p><strong>Withheld:</strong> president, party, title, URL, and existing labels.</p>
<p>Adjacent-context excerpts (the complete indexed text was submitted):</p>
<ol class="request-context">{context_rows}</ol></div>
<div class="worked-step"><span class="step-label">3 · Structured output</span>
<dl><div><dt>Topics</dt><dd>{html.escape(raw_topics)}</dd></div>
<div><dt>Proposal / values</dt><dd>{html.escape(example['proposal_values'])}</dd></div>
<div><dt>Partisan attack</dt><dd>{str(example['party_attack']).lower()}</dd></div>
<div><dt>Named adversary</dt><dd>{str(example['enemy_naming']).lower()}</dd></div>
<div><dt>Zero-sum framing</dt><dd>{str(example['zero_sum']).lower()}</dd></div></dl></div>
<div class="worked-step"><span class="step-label">4 · Normalization</span>
<p>Topic names are case-normalized against the frozen taxonomy and deduplicated within the paragraph.</p>
<p><strong>Canonical set:</strong> {html.escape(normalized_topics)}</p></div>
<div class="worked-step"><span class="step-label">5 · Semantic key + receipt</span>
<p><code>(doc_name, para_idx)</code><br><code>{html.escape(example['doc_name'])}</code>,
<code>{example['para_idx']}</code></p><p>{html.escape(example['president'])},
“{html.escape(example['title'])}” ({example['year']}).</p>
<p><strong>Run:</strong> <code>{html.escape(example['run_id'])}</code><br>
<strong>Model:</strong> {html.escape(example['model'])}<br>
<strong>Prompt hash:</strong> <code>{html.escape(example['prompt_hash'])}</code></p></div>
<div class="worked-step"><span class="step-label">6 · Downstream chart</span>
<p>This keyed paragraph contributes one denominator record and one numerator record for each
canonical topic on every consuming topic-share chart.</p><p><a href="metrics.html#paragraph_share">Define paragraph share</a> ·
<a href="index.html">See the Story</a></p></div></div></figure>"""


def _pipeline_html(s: dict[str, Any]) -> str:
    stages = (
        ("Corpus", f"{s['n_speeches']:,} source speeches"),
        ("Segmentation", f"{s['n_paragraphs']:,} keyed paragraphs"),
        ("Discovery", "Embeddings + seeded paragraph clusters"),
        ("Sampling", "Historically isolated proposal bins"),
        ("Freeze", f"{s['n_domains']} domains + {s['n_topics']} topics"),
        ("Judgment", "Consecutive paragraph chunk + decade; identity masked"),
        ("Normalize", "Canonical names + duplicate removal"),
        ("Audit", "Persisted second-model sample"),
    )
    items = "".join(
        f"<li><span>{index}</span><strong>{html.escape(label)}</strong>"
        f"<small>{html.escape(detail)}</small></li>"
        for index, (label, detail) in enumerate(stages, start=1)
    )
    return f"""<figure class="pipeline-figure" data-substantive-chart
 data-metric="annotation_pipeline_lineage"
 data-evidence="data/llm_annotations/manifests/*.json"
 aria-labelledby="pipeline-title"><figcaption id="pipeline-title"><strong>The labeling pipeline</strong>
<span>A responsive HTML flow keeps every stage readable on a phone and without JavaScript.
<a href="metrics.html#annotation_pipeline_lineage">Define pipeline lineage.</a></span></figcaption>
<ol class="pipeline">{items}</ol></figure>"""


def _masking_html() -> str:
    return """<figure class="masking-figure" data-substantive-chart
 data-metric="annotation_evidence_visibility"
 data-evidence="data/llm_annotations/manifests/*.json + prompt hashes"
 aria-labelledby="masking-title"><figcaption id="masking-title"><strong>Two passes, two evidence policies</strong>
<span>Evidence is removed only when it could bias the requested judgment.
<a href="metrics.html#annotation_evidence_visibility">Define evidence visibility.</a></span></figcaption>
<div class="masking-grid"><section><h3>Paragraph judgment · masked</h3>
<ul><li class="visible">Several consecutive paragraphs from one speech</li><li class="visible">Speech decade</li>
<li class="hidden">President hidden</li><li class="hidden">Party hidden</li>
<li class="hidden">Speech title hidden</li></ul><p>Used for topics, combat flags,
proposal/values, and entities.</p></section><section><h3>Speech facts · unmasked</h3>
<ul><li class="visible">Speech title</li><li class="visible">Year</li>
<li class="visible">First five paragraphs</li></ul><p>Used for speech type, audience,
and medium. The title is direct evidence for this factual task.</p></section></div></figure>"""


def _taxonomy_atlas(ai_data: dict) -> str:
    taxonomy = ai_data["taxonomy"]
    by_domain: dict[str, list[dict]] = {}
    for topic in taxonomy["level2"]:
        by_domain.setdefault(topic["level1"], []).append(topic)
    kinds = {domain["name"]: domain["kind"] for domain in taxonomy["level1"]}
    blocks = []
    for domain in taxonomy["level1"]:
        name = domain["name"]
        topics = by_domain.get(name, [])
        rows = "".join(
            f"""<li><strong>{html.escape(topic['name'])}</strong>
  <span>{html.escape(topic['definition'])}</span></li>"""
            for topic in topics
        )
        noun = "topic" if len(topics) == 1 else "topics"
        blocks.append(f"""<details class="tax-domain">
<summary><span>{html.escape(name)}</span><span class="tax-kind">{html.escape(kinds[name])} · {len(topics)} {noun}</span></summary>
<p>{html.escape(domain['definition'])}</p><ul>{rows}</ul></details>""")
    return "\n".join(blocks)


def _rubric_cards() -> str:
    cards = (
        ("topics", "Zero or more names from the frozen topic list. No label is forced; several may apply."),
        ("party_attack", "True only when an identifiable partisan actor is assigned fault, failure, or bad motive."),
        ("enemy_naming", "True only when a specific person, group, nation, or institution is framed as an adversary or threat."),
        ("zero_sum", "True only when the paragraph says one side’s gain requires another side’s loss—not merely when conflict appears."),
        ("proposal", "A concrete policy, program, law, appropriation, or course of action is urged."),
        ("values", "Principles, identity, faith, gratitude, or civic virtue appear without a concrete policy ask."),
        ("mixed", "A substantial concrete ask and a substantial values appeal both appear."),
        ("neither", "Narrative, procedure, ceremony, factual reporting, or rebuttal that fits neither side."),
        ("entities", "Specific actors are extracted with a type and adversarial, favorable, or neutral stance."),
    )
    return "".join(
        f'<div class="rubric"><code>{html.escape(name)}</code><p>{html.escape(text)}</p></div>'
        for name, text in cards
    )


def _grammar_appendix() -> str:
    try:
        _, manifest = grammar.validate_cached_artifacts()
        era = pd.read_csv(grammar.ERA_POS_PATH)
    except (grammar.GrammarArtifactError, OSError, ValueError) as exc:
        return (
            '<div class="callout warning"><p><strong>POS appendix withheld.</strong> '
            + html.escape(str(exc))
            + " The site does not publish a stale model artifact.</p></div>"
        )
    shown = ("NOUN", "PROPN", "VERB", "PRON", "ADJ")
    labels = {
        "NOUN": "Common nouns", "PROPN": "Proper nouns", "VERB": "Verbs",
        "PRON": "Pronouns", "ADJ": "Adjectives",
    }
    headers = "".join(f"<th>{labels[tag]}</th>" for tag in shown)
    rows: list[str] = []
    for era_name in dict.fromkeys(era["era"]):
        subset = era[era["era"].eq(era_name)].set_index("part_of_speech")
        values = "".join(
            f"<td>{float(subset.loc[tag, 'share_percent']):.2f}%</td>"
            if tag in subset.index else "<td>not available</td>"
            for tag in shown
        )
        rows.append(f'<tr><th scope="row">{html.escape(str(era_name))}</th>{values}</tr>')
    model = manifest["model"]
    evidence = "data/grammar/era_pos.csv + data/grammar/manifest.json"
    return f"""<figure class="evidence-figure" data-substantive-chart
 data-metric="coarse_pos_share" data-evidence="{evidence}"
 aria-labelledby="grammar-title"><figcaption id="grammar-title"><strong>Feature-engineering appendix · coarse parts of speech</strong>
<span>A descriptive tagger check, not an evaluation of syntactic complexity.
<a href="metrics.html#coarse_pos_share">Define coarse POS share.</a></span></figcaption>
<div class="table-scroll" role="region" aria-label="Coarse parts of speech by reporting era" tabindex="0"><table><thead><tr><th>Reporting era</th>{headers}</tr></thead>
<tbody>{''.join(rows)}</tbody></table></div><p class="figure-note">Tagged locally with
<code>{html.escape(model['package'])}</code> {html.escape(model['package_version'])} under
spaCy {html.escape(model['library_version'])}. Historical spelling, OCR, quotation, and genre
can change tag distributions. The manifest pins source and artifact hashes; stale caches fail closed.</p></figure>"""


def publish_grammar_artifacts(site_dir: Path) -> dict[str, Path]:
    """Publish the validated small POS receipt files without exposing the parquet.

    The canonical public location belongs to Methods. The former Data Quality
    CSV remains as a byte-identical compatibility alias so existing links do not
    break during the ownership move.
    """
    grammar.validate_cached_artifacts()
    targets = {
        "era_csv": Path(site_dir) / "data" / "grammar" / "era_pos.csv",
        "manifest": Path(site_dir) / "data" / "grammar" / "manifest.json",
        "compatibility_csv": (
            Path(site_dir) / "data" / "quality" / "era_part_of_speech.csv"
        ),
    }
    sources = {
        "era_csv": grammar.ERA_POS_PATH,
        "manifest": grammar.MANIFEST_PATH,
        "compatibility_csv": grammar.ERA_POS_PATH,
    }
    staged: list[tuple[Path, Path]] = []
    try:
        for name, target in targets.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(sources[name].read_bytes())
                handle.flush()
                os.fsync(handle.fileno())
            staged.append((temporary, target))
        for temporary, target in staged:
            os.replace(temporary, target)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    if targets["era_csv"].read_bytes() != targets["compatibility_csv"].read_bytes():
        raise RuntimeError("grammar compatibility CSV is not byte-identical")
    return targets


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _artifact_identities(paths: list[str]) -> str:
    rows = []
    for relative in paths:
        path = REPO_ROOT / relative
        identity = _sha256_path(path) if path.is_file() else "not present in isolated render"
        rows.append(
            f"<code>{html.escape(relative)}</code><br><code>{html.escape(identity)}</code>"
        )
    return "<br>".join(rows)


def _code_paths(paths: list[str]) -> str:
    return "<br>".join(f"<code>{html.escape(path)}</code>" for path in paths)


def _provenance_cards(ai_data: dict) -> str:
    # Keep the argument for renderer compatibility; identities below come from
    # the stage manifests and artifact bytes rather than a lossy page summary.
    _ = ai_data
    run_dir = REPO_ROOT / "data" / "llm_annotations" / "manifests"
    taxonomy_run = _read_json(run_dir / "taxonomy-v1-20260719.json")
    primary_run = _read_json(run_dir / "judgment-full-20260720.json")
    factual_run = _read_json(run_dir / "factual-full-20260720.json")
    second_run = _read_json(run_dir / "opus-judgment-20260721.json")
    quality_manifest = _read_json(
        REPO_ROOT / "docs" / "data" / "quality" / quality_audit.MANIFEST_NAME
    )
    grammar_manifest = _read_json(grammar.MANIFEST_PATH)

    def run_identity(run: dict[str, Any], outputs: list[str]) -> str:
        prompt_version = str(run.get("prompt_version", "schema unavailable"))
        prompt_hash = str(run.get("prompt_hash", "prompt hash unavailable"))
        return (
            f"<strong>{html.escape(prompt_version)}</strong><br>"
            f"<code>{html.escape(prompt_hash)}</code><br>"
            f"{_artifact_identities(outputs)}"
        )

    def run_detail(run: dict[str, Any], qualifier: str = "") -> str:
        detail = (
            f"Historical run {run.get('run_id', 'unavailable')}: "
            f"{int(run.get('n_requests', 0)):,} requests · "
            f"${float(run.get('cost_usd', 0)):.4f} · "
            f"{run.get('model', 'model recorded in manifest')}."
        )
        return f"{detail} {qualifier}".strip()

    quality_identity = (
        f"<strong>{html.escape(str(quality_manifest.get('schema_version', quality_audit.MANIFEST_SCHEMA)))}</strong>"
        f"<br><code>{html.escape(str(quality_manifest.get('metadata_sha256', 'emitted during canonical build')))}</code>"
    )
    grammar_identity = (
        f"<strong>{html.escape(str(grammar_manifest.get('schema_version', grammar.SCHEMA_VERSION)))}</strong>"
        f"<br><code>{html.escape(str(grammar_manifest.get('artifacts', {}).get('era_pos.csv', 'artifact hash unavailable')))}</code>"
    )
    stages = (
        {
            "label": "Taxonomy discovery",
            "status": "Frozen paid artifact · read only",
            "inputs": ["data/paragraphs.parquet", "era-isolated proposal samples"],
            "outputs": ["data/llm_annotations/taxonomy_v1.json", "data/llm_annotations/crosswalk_v1.json"],
            "identity": run_identity(
                taxonomy_run,
                ["data/llm_annotations/taxonomy_v1.json", "data/llm_annotations/crosswalk_v1.json"],
            ),
            "command": "arch -x86_64 .venv/bin/python -m pytest tests/test_ai_labels_site.py -q",
            "tests": ["tests/test_ai_labels_site.py"],
            "detail": run_detail(taxonomy_run),
        },
        {
            "label": "Primary paragraph judgments",
            "status": "Frozen paid artifact · read only",
            "inputs": ["data/paragraphs.parquet", "data/llm_annotations/taxonomy_v1.json"],
            "outputs": ["data/llm_annotations/paragraph_annotations.parquet", "data/llm_annotations/paragraph_entities.parquet"],
            "identity": run_identity(
                primary_run,
                ["data/llm_annotations/paragraph_annotations.parquet", "data/llm_annotations/paragraph_entities.parquet"],
            ),
            "command": "arch -x86_64 .venv/bin/python -m pytest tests/test_ai_labels_site.py tests/test_quality_audit.py -q",
            "tests": ["tests/test_ai_labels_site.py", "tests/test_quality_audit.py"],
            "detail": run_detail(
                primary_run,
                "Final frozen tables also contain validated retry/chunk convergence rows under the same prompt contract.",
            ),
        },
        {
            "label": "Factual speech labels",
            "status": "Frozen paid artifact · read only",
            "inputs": ["data/speeches.parquet", "first five source paragraphs"],
            "outputs": ["data/llm_annotations/speech_annotations.parquet"],
            "identity": run_identity(
                factual_run, ["data/llm_annotations/speech_annotations.parquet"]
            ),
            "command": "arch -x86_64 .venv/bin/python -m pytest tests/test_ai_labels_site.py tests/test_validation_protocols.py -q",
            "tests": ["tests/test_ai_labels_site.py", "tests/test_validation_protocols.py"],
            "detail": run_detail(
                factual_run,
                "Final frozen speech labels also contain validated retry rows under the same prompt contract.",
            ),
        },
        {
            "label": "Second-model reproducibility",
            "status": "Frozen labels + deterministic derived agreement",
            "inputs": ["data/llm_annotations/agreement_sample_v1.json", "primary and second annotation parquets"],
            "outputs": ["data/llm_annotations/agreement_v1.parquet", "docs/data/quality/agreement_by_field_and_era.csv"],
            "identity": run_identity(
                second_run,
                ["data/llm_annotations/agreement_v1.parquet", "data/llm_annotations/paragraph_annotations__opus4-8.parquet"],
            ),
            "command": "arch -x86_64 .venv/bin/python -m pytest tests/test_agreement_metrics.py tests/test_quality_audit.py -q",
            "tests": ["tests/test_agreement_metrics.py", "tests/test_quality_audit.py"],
            "detail": run_detail(
                second_run,
                "The persisted paired audit retains later validated retry rows and missingness.",
            ),
        },
        {
            "label": "Coarse POS appendix",
            "status": "Deterministic local derived artifact",
            "inputs": ["data/speeches.parquet", "en_core_web_sm"],
            "outputs": ["data/grammar/speech_pos.parquet", "data/grammar/era_pos.csv", "data/grammar/manifest.json"],
            "identity": grammar_identity,
            "command": "arch -x86_64 .venv/bin/python -m presidential_profiles.grammar",
            "tests": ["tests/test_grammar.py", "tests/test_ai_labels_site.py"],
            "detail": None,
        },
        {
            "label": "Data Quality and static projections",
            "status": "Deterministic local derived output",
            "inputs": ["governed corpus, frozen labels, speaker foundation, era specifications"],
            "outputs": ["docs/*.html", "docs/data/quality/*", "docs/data/validation-protocols-v1/*"],
            "identity": quality_identity,
            "command": "arch -x86_64 .venv/bin/python -m presidential_profiles.site",
            "tests": ["tests/test_quality_audit.py", "tests/test_site_validation.py", "tests/test_validation_protocols.py"],
            "detail": None,
        },
    )
    cards: list[str] = []
    for stage in stages:
        detail = stage["detail"] or "No paid request is made by this command."
        if stage["detail"] is not None:
            detail += " The verification command makes no paid request."
        cards.append(f"""<article class="receipt"><span>{html.escape(stage['status'])}</span>
<h3>{html.escape(stage['label'])}</h3><dl class="receipt-contract">
<div><dt>Inputs</dt><dd>{_code_paths(stage['inputs'])}</dd></div>
<div><dt>Outputs</dt><dd>{_code_paths(stage['outputs'])}</dd></div>
<div><dt>Schema / hash identity</dt><dd>{stage['identity']}</dd></div>
<div><dt>Command</dt><dd><code>{html.escape(stage['command'])}</code></dd></div>
<div><dt>Relevant tests</dt><dd>{_code_paths(stage['tests'])}</dd></div></dl>
<small>{html.escape(detail)}</small></article>""")
    return "".join(cards)


METHOD_CSS = r"""
  header, main, footer { max-width:1080px; }
  header { padding-top:38px; }
  .skip-link { position:fixed; left:12px; top:10px; z-index:1000; transform:translateY(-180%);
    background:#172a38; color:#fff; padding:10px 14px; border-radius:8px; }
  .skip-link:focus { transform:none; }
  .crumbs { margin-bottom:20px; font-size:.88rem; } .crumbs a { color:var(--ink2); }
  .kicker { color:#275d8c; font-weight:780; letter-spacing:.08em; text-transform:uppercase; font-size:.76rem; }
  header h1 { font-size:clamp(2.25rem,6vw,4.6rem); line-height:1.01; max-width:900px; margin-top:12px; }
  header .sub { font-size:1.1rem; max-width:800px; }
  .stat-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(165px,1fr)); gap:10px; margin-top:26px; }
  .stat { background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:16px; }
  .stat strong { display:block; font-size:1.55rem; } .stat span { color:var(--muted); font-size:.8rem; }
  .methods-local-nav { position:sticky; top:var(--global-nav-height,0); z-index:30; margin:22px 0 0;
    background:rgba(249,249,247,.96); border-block:1px solid var(--border); overflow-x:auto; }
  .methods-local-nav ul { display:flex; list-style:none; padding:6px 0; margin:0 auto; max-width:1080px; }
  .methods-scroll-cue { display:none; align-items:center; white-space:nowrap; padding:6px 10px;
    color:#52616c; font-size:.7rem; font-weight:750; }
  .methods-local-nav a { display:grid; place-items:center; min-height:40px; padding:6px 12px;
    white-space:nowrap; color:var(--ink2); text-decoration:none; font-size:.78rem; font-weight:680; }
  .methods-local-nav a:hover { background:#ece7df; border-radius:8px; }
  main { padding-top:18px; } .method-section { scroll-margin-top:190px; margin-top:62px; }
  .method-section>h2 { font:700 clamp(1.65rem,3vw,2.15rem)/1.15 Georgia,serif; letter-spacing:-.02em; }
  .method-section>p, .method-section>ul, .method-section>ol { max-width:780px; color:var(--ink2); margin-top:12px; }
  .verdict-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin-top:20px; }
  .verdict { background:#fff; border:1px solid var(--border); border-radius:15px; padding:18px; }
  .verdict span { color:#275d8c; text-transform:uppercase; letter-spacing:.07em; font-size:.7rem; font-weight:800; }
  .verdict h3 { font-size:1rem; margin-top:7px; } .verdict p { color:var(--ink2); margin-top:7px; font-size:.9rem; }
  figure { margin:26px 0; } figcaption { margin-bottom:14px; }
  figcaption strong, figcaption span { display:block; } figcaption strong { font-size:1.08rem; }
  figcaption span { color:var(--muted); font-size:.86rem; margin-top:3px; }
  .pipeline-figure,.masking-figure,.worked-example,.evidence-figure { background:var(--surface);
    border:1px solid var(--border); border-radius:18px; padding:20px; }
  .pipeline { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); list-style:none; padding:0; gap:10px; }
  .pipeline li { position:relative; min-height:126px; padding:15px; border:1px solid #cbd3d9;
    border-radius:13px; background:#fff; }
  .pipeline li>span { display:grid; width:27px; height:27px; place-items:center; border-radius:50%;
    background:#172a38; color:#fff; font-weight:800; font-size:.72rem; }
  .pipeline strong,.pipeline small { display:block; } .pipeline strong { margin-top:12px; }
  .pipeline small { color:var(--muted); margin-top:5px; line-height:1.4; }
  .masking-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }
  .masking-grid section { margin:0; border:1px solid var(--border); border-radius:14px; padding:18px; background:#fff; }
  .masking-grid h3 { font-size:1rem; } .masking-grid ul { display:flex; flex-wrap:wrap; list-style:none; padding:0; gap:7px; margin-top:13px; }
  .masking-grid li { border-radius:999px; padding:7px 11px; font-size:.8rem; }
  .masking-grid .visible { background:#e5f3e9; border:1px solid #70aa85; }
  .masking-grid .hidden { background:#f6eeee; border:1px dashed #a66b6b; }
  .masking-grid p { color:var(--muted); margin-top:14px; font-size:.86rem; }
  .worked-flow { display:grid; grid-template-columns:1.3fr 1fr 1fr; gap:10px; }
  .worked-step { background:#fff; border:1px solid var(--border); border-radius:13px; padding:15px; min-width:0; }
  .step-label { color:#275d8c; text-transform:uppercase; letter-spacing:.07em; font-size:.68rem; font-weight:800; }
  blockquote { border-left:3px solid #6da7ec; padding-left:12px; margin-top:12px; color:var(--ink2); font-family:Georgia,serif; }
  .worked-step p { color:var(--ink2); font-size:.84rem; margin-top:12px; }
  .request-context { padding-left:18px; } .request-context li { margin:8px 0; }
  .request-context strong,.request-context span { display:block; font-size:.76rem; }
  .request-context span { color:var(--ink2); margin-top:3px; }
  .worked-step code { overflow-wrap:anywhere; } .worked-step dl { margin-top:10px; }
  .worked-step dl div { display:grid; grid-template-columns:minmax(92px,.8fr) 1.2fr; gap:8px; padding:6px 0; border-bottom:1px solid var(--grid); }
  .worked-step dt { color:var(--muted); font-size:.75rem; } .worked-step dd { font-size:.8rem; font-weight:650; }
  .callout { border-left:4px solid #6da7ec; background:#f1f6fc; border-radius:0 12px 12px 0; padding:16px 18px; margin:20px 0; max-width:800px; }
  .callout.warning { border-color:#9b6227; background:#fbf3e9; } .callout p { margin:0; }
  .rubrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; margin-top:18px; }
  .rubric { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:14px; }
  .rubric code { font-weight:750; color:#275d8c; } .rubric p { font-size:.86rem; color:var(--ink2); margin-top:7px; }
  .metric-family { margin-top:22px; } .metric-family h3 { font-size:1rem; }
  .metric-family>p { color:var(--muted); margin-top:4px; font-size:.84rem; }
  .evaluation-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:9px; margin-top:11px; }
  .evaluation-card { background:#fff; border:1px solid var(--border); border-radius:12px; padding:14px; }
  .evaluation-card>* { display:block; } .evaluation-card .metric-name { color:var(--muted); font-size:.76rem; }
  .evaluation-card strong { font-size:1.35rem; margin:4px 0; } .evaluation-card span { font-size:.78rem; }
  .evaluation-card small { color:var(--muted); font-size:.7rem; margin-top:8px; }
  .table-scroll { width:100%; overflow-x:auto; } .table-scroll:focus-visible { outline:3px solid #824600; outline-offset:2px; }
  table { width:100%; border-collapse:collapse; background:#fff; }
  th,td { border:1px solid var(--border); padding:10px; text-align:left; vertical-align:top; font-size:.8rem; }
  thead th { color:var(--ink2); background:#f1f3f4; } td small { display:block; color:var(--muted); margin-top:3px; font-size:.67rem; }
  .era-matrix { min-width:850px; } .evidence-table { margin-top:18px; }
  .agreement-cell { background:linear-gradient(90deg,#d8e9f4 0 var(--agreement),#fff var(--agreement) 100%); }
  .scroll-cue { display:none; color:var(--muted); font-size:.75rem; margin-bottom:8px; }
  details summary { cursor:pointer; font-weight:700; }
  .entity-funnel ol { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); list-style:none; padding:0; gap:10px; }
  .entity-funnel li { background:#fff; border:1px solid var(--border); border-radius:13px; padding:15px; }
  .entity-funnel li span,.entity-funnel li strong,.entity-funnel li small { display:block; }
  .entity-funnel li span { color:var(--muted); font-size:.76rem; } .entity-funnel li strong { font-size:1.25rem; margin:4px 0; }
  .entity-funnel li small,.figure-note { color:var(--muted); font-size:.75rem; }
  .figure-note { margin-top:13px; } .passport-note { color:var(--muted); font-size:.82rem; max-width:800px; }
  .tax-atlas { margin-top:18px; } .tax-domain { background:var(--surface); border:1px solid var(--border); border-radius:12px; margin:8px 0; padding:0 16px; }
  .tax-domain summary { display:flex; justify-content:space-between; gap:20px; padding:15px 0; }
  .tax-kind { color:var(--muted); font-size:.76rem; font-weight:500; } .tax-domain>p { margin:0 0 12px; font-size:.88rem; color:var(--ink2); }
  .tax-domain ul { margin:0 0 18px; padding-left:22px; } .tax-domain li { margin:10px 0; color:var(--ink2); }
  .tax-domain li span { display:block; font-size:.84rem; }
  .receipt-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; margin-top:18px; }
  .receipt { background:#fff; border:1px solid var(--border); border-radius:13px; padding:16px; min-width:0; }
  .receipt>span { color:#275d8c; text-transform:uppercase; letter-spacing:.06em; font-size:.68rem; font-weight:800; }
  .receipt h3 { font-size:.98rem; margin-top:6px; } .receipt p { color:var(--ink2); font-size:.8rem; margin-top:8px; }
  .receipt code { overflow-wrap:anywhere; } .receipt small { display:block; color:var(--muted); margin-top:8px; }
  .receipt-contract { margin:12px 0 0; } .receipt-contract>div { padding:8px 0; border-top:1px solid var(--grid); }
  .receipt-contract dt { color:var(--muted); font-size:.68rem; font-weight:800; text-transform:uppercase; letter-spacing:.05em; }
  .receipt-contract dd { margin:4px 0 0; color:var(--ink2); font-size:.76rem; overflow-wrap:anywhere; }
  .trust-receipt { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1px;
    background:var(--border); border:1px solid var(--border); border-radius:14px;
    overflow:hidden; margin:20px 0; }
  .trust-receipt div { background:#fff; padding:13px; min-width:0; }
  .trust-receipt dt { color:#52616c; font-size:.68rem; font-weight:800;
    letter-spacing:.06em; text-transform:uppercase; }
  .trust-receipt dd { margin:.35rem 0 0; font-size:.8rem; overflow-wrap:anywhere; }
  .deep-section { margin-top:22px; padding:0 18px; border:1px solid var(--border); border-radius:14px; background:#fff; }
  .deep-section>summary { padding:17px 0; } .deep-section>div { padding-bottom:18px; }
  .limitations { columns:2; column-gap:36px; padding-left:20px; } .limitations li { break-inside:avoid; margin-bottom:12px; color:var(--ink2); }
  .download-list { display:flex; flex-wrap:wrap; gap:9px; list-style:none; padding:0; margin-top:15px; }
  .download-list a { display:block; border:1px solid var(--border); border-radius:9px; padding:9px 12px; background:#fff; }
  @media(max-width:760px) {
    .verdict-grid,.masking-grid,.worked-flow,.receipt-grid,.trust-receipt { grid-template-columns:1fr; }
    .pipeline { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .entity-funnel ol { grid-template-columns:1fr; }
    .limitations { columns:1; }
    .scroll-cue { display:block; }
  }
  @media(max-width:780px) {
    .methods-scroll-cue { display:flex; }
  }
  @media(max-width:850px) {
    .table-scroll::before { content:"Scroll table horizontally →"; display:block; position:sticky;
      left:0; width:max-content; margin:0 0 7px; padding:4px 7px; border-radius:6px;
      background:#eef2f3; color:#52616c; font-size:.7rem; font-weight:750; }
  }
  @media(max-width:430px) {
    header,main,footer { padding-inline:15px; }
    .pipeline { grid-template-columns:1fr; } .pipeline li { min-height:0; }
    .pipeline-figure,.masking-figure,.worked-example,.evidence-figure { padding:15px; }
    .methods-local-nav a { padding-inline:10px; font-size:.72rem; }
  }
  @media(prefers-reduced-motion:reduce) { html { scroll-behavior:auto; } }
  @media(forced-colors:active) { .visible,.hidden { border:1px solid CanvasText!important; } }
"""


def render_methodology(
    ai_data: dict,
    *,
    agreement_evidence: pd.DataFrame | None = None,
    agreement_source: str | None = None,
) -> str:
    s = ai_data["summary"]
    tax = ai_data["manifests"]["taxonomy"]
    if agreement_evidence is None:
        agreement, evidence = load_agreement_evidence()
    else:
        agreement = _normalize_agreement_table(agreement_evidence)
        evidence = agreement_source or "caller-supplied agreement evidence"
    pop = _population_data(ai_data, agreement)
    zero = _overall_row(agreement, "zero_sum", "cohen_kappa")
    zero_label = _metric_value("cohen_kappa", float(zero["value"])) if zero is not None else "not available"
    entity_names = _overall_row(agreement, "entities", "name_match_rate")
    entity_label = _pct(float(entity_names["value"])) if entity_names is not None else "not available"
    evaluation_receipt = trust_appendices.trust_receipt(
        claim="Cross-model agreement measures reproducibility under the same rubric, not correctness.",
        population=f"{_count(pop['sample_paired'])} paired paragraphs of {_count(pop['sample_expected'])} expected",
        method="Field-specific normalized agreement with whole-speech clustered intervals",
        uncertainty="95% deterministic speech-cluster bootstrap where estimable",
        artifact=evidence,
        status="Measured reproducibility; human validity not measured",
        limitation="The model passes share prompt design and may share model-family biases",
        downstream_use="Qualify model-sensitive findings and prioritize blinded human review",
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>From speech to auditable paragraph judgment - Presidential Profiles</title>
<meta name="description" content="An auditable account of the data, evidence policy, semantic keys, model reproducibility, uncertainty, and limitations behind Presidential Profiles.">
<style>{PAGE_CSS}{METHOD_CSS}</style></head><body>
<a class="skip-link" href="#main-content">Skip to the method</a>
<header><p class="crumbs"><a href="index.html">← Story</a> &nbsp;·&nbsp;
<a href="data-quality.html">Data Quality</a> &nbsp;·&nbsp; <a href="era-boundaries.html">Era Choices</a></p>
<div class="kicker">Methods · labeling and evaluation</div>
<h1>From speech to auditable paragraph judgment</h1>
<p class="sub">Every label can be traced to a semantic key, a frozen prompt, a model run,
and a declared population. A second model tests reproducibility—not truth—and human validity
remains an explicit open question.</p><div class="stat-grid">
<div class="stat"><strong>{s['n_paragraphs']:,}</strong><span>primary paragraph judgments completed</span></div>
<div class="stat"><strong>{_count(pop['sample_paired'])}</strong><span>paired judgments available for evaluation</span></div>
<div class="stat"><strong>{s['n_topics']}</strong><span>frozen fine-grained topics</span></div>
<div class="stat"><strong>{s['n_assignments']:,}</strong><span>normalized, distinct topic assignments</span></div>
<div class="stat"><strong>{entity_label}</strong><span>strict entity-name reproducibility</span></div>
</div></header>
{trust_appendices.methods_local_nav('overview')}
<main id="main-content">
<section class="method-section" id="overview"><h2>The 30-second trust verdict</h2>
<div class="verdict-grid"><div class="verdict"><span>What is strong</span><h3>Failures are designed to be visible</h3>
<p>Exact semantic keys, one-to-one join validation, frozen paid artifacts, prompt hashes,
corpus fingerprints, complete denominators, and speech-level uncertainty make silent drift harder.</p></div>
<div class="verdict"><span>What the audit supports</span><h3>Reproducibility varies by field</h3>
<p>The second model tests sensitivity to model choice under the same prompt. Topic overlap is
set-based; binary judgments use κ; closed categories use exact agreement.</p></div>
<div class="verdict"><span>What it does not support</span><h3>No human accuracy claim</h3>
<p>Related models can share biases. Agreement cannot establish correctness, historical validity,
or a causal interpretation, and disagreement is never erased from the record.</p></div>
<div class="verdict"><span>Where caution is highest</span><h3>Zero-sum and entity detection</h3>
<p>Zero-sum reproducibility is {html.escape(zero_label)}. Entity matching is strict and should be
separated from stance agreement after a name has matched.</p></div></div>
{_pipeline_html(s)}</section>

<section class="method-section" id="labeling"><h2>Follow one paragraph through the system</h2>
<p>The easiest way to understand the pipeline is to watch evidence disappear, a constrained
judgment appear, and provenance return only after inference.</p>{_worked_example_html(_worked_example(ai_data))}
{_masking_html()}
<h3>Why the paragraph is the unit</h3><p>A speech can change subjects, move from policy to ceremony,
or name an adversary once. Paragraph units keep those transitions visible. Every paragraph join uses
<code>(doc_name, para_idx)</code>; row position is never identity.</p>
<details class="deep-section"><summary>How the {s['n_topics']}-topic taxonomy was discovered and frozen</summary><div>
<ol><li><strong>Find blind spots.</strong> Paragraphs missed by the anchored issue layer were
deliberately oversampled; the frozen run receipt records the source population.</li>
<li><strong>Cluster before naming.</strong> Normalized embeddings and a seeded clustering pass
supplied evidence; clusters were not treated as final topics.</li>
<li><strong>Use isolated historical samples.</strong> Fixed historical proposal bins prevented an
early-period proposal from borrowing modern topic vocabulary.</li>
<li><strong>Merge once, then freeze.</strong> The proposals became {s['n_domains']} domains and
{s['n_topics']} topics plus a declared crosswalk to the legacy issues.</li>
<li><strong>Test labelability on held-out text.</strong> A separate held-out set exceeded its
pre-registered coverage gate. That establishes rubric coverage, not semantic correctness; the exact
sample size, split, and result remain in the frozen taxonomy receipt.</li></ol>
<p>The taxonomy run used {html.escape(', '.join(tax['models']))}, {tax['requests']:,} recorded API
requests, and ${tax['cost_usd']:.4f} in usage-derived cost.</p></div></details>
<details class="deep-section"><summary>Inspect the closed judgment rubric</summary><div>
<p>The structured response accepted only declared fields and enum values. Unknown topic strings
could not be invented.</p><div class="rubrics">{_rubric_cards()}</div></div></details>
<details class="deep-section"><summary>Inspect the complete frozen taxonomy</summary><div>
<p>The {s['n_domains']} domains include policy and non-policy language. Open a domain to see every available
topic and its frozen definition.</p><div class="tax-atlas">{_taxonomy_atlas(ai_data)}</div></div></details>
{_grammar_appendix()}</section>

<section class="method-section" id="evaluation"><h2>Evaluation asks five different questions</h2>
<ol><li><strong>Annotation processing completeness:</strong> did the primary pipeline return one valid row for
every expected key?</li><li><strong>Taxonomy labelability:</strong> could held-out text be expressed
with the frozen topic vocabulary?</li><li><strong>Paired-evaluation availability:</strong> how much of
the persisted second-model sample has both judgments?</li><li><strong>Cross-model reproducibility:</strong>
do two models make the same rubric decisions under byte-identical prompts?</li>
<li><strong>Human validity:</strong> do blinded people applying the rubric agree with one another and
with the models? This has not yet been measured.</li></ol>
<div class="callout"><p><strong>Important:</strong> the second model does not reveal where the first
model is “uncertain.” It measures labeler sensitivity to model choice within a related model family.
Shared training data and shared prompt interpretation can make both models wrong in the same way.</p></div>
{_agreement_panels(agreement, evidence)}
{_era_agreement_table(agreement, evidence)}
{_entity_funnel(agreement, evidence)}
{evaluation_receipt}
<h3>How uncertainty reaches public charts</h3><p>Trend bands resample whole speeches because
paragraphs within one speech are correlated. AI-topic bands can separately include sampling
variation and measured cross-model disagreement. CorEx intervals and LLM intervals use different
time grains and estimands, so their average widths are not compared as if one were intrinsically
more certain.</p></section>

<section class="method-section" id="populations"><h2>Know which record a number describes</h2>
<p>A source document owner and the person speaking a paragraph are often the same—but not always.
The population passport prevents those units from being silently substituted.</p>
{_population_passport(pop)}
<p class="passport-note">The consumer register above makes document ownership, actual-speaker
attribution, corpus aggregation, and the Data Quality-only paired sample explicit. The paired audit
is evidence about measurement reproducibility, not a population for president rankings.</p>
<h3>Era calibration is an unresolved measurement question</h3><p>The judgment prompt includes the
speech decade so labels can be interpreted against their historical setting. This reduces obvious
anachronism, but it also means the label may represent an era-calibrated threshold rather than one
unchanging threshold across the full historical span. A preregistered decade-hidden sensitivity study is specified,
but no paid run or result exists.</p></section>

<section class="method-section" id="limitations"><h2>Limitations that remain visible</h2>
<ul class="limitations"><li>The Miller Center collection is curated formal speech, not every rally,
interview, social post, or private conversation.</li><li>A frozen taxonomy improves comparability but
retains known gaps, including no standalone education topic and no modern monetary-policy topic.</li>
<li>Era context reduces anachronism; it does not prove measurement invariance or eliminate implausible
assignments.</li><li>Cross-model agreement is not a human reference standard. Zero-sum framing
({html.escape(zero_label)}) warrants more caution than high-reproducibility factual fields.</li>
<li>Entity normalization is deliberately conservative, so span or wording differences lower strict
name agreement.</li><li>President-level percentages are descriptive, not causal. Thin records remain
marked rather than promoted as precise estimates.</li><li>Coarse POS shares can reflect genre, spelling,
OCR, quotation, or tagger behavior; they are not syntactic-complexity scores.</li><li>The two model
passes share a prompt and model family, so their errors are not independent.</li></ul>
<div class="callout"><p><strong>Planned, not completed:</strong> the repository now specifies a blinded
human-validity study and a paid decade-hidden measurement-invariance experiment. Both remain gated;
the site makes no result claim until evidence and approvals exist. Inspect the public-safe
<a href="data/validation-protocols-v1/manifest_v1.json">protocol manifest</a>; selected keys,
coder assignments, and the dry-run request plan are not exposed in the public projection. This is
an operational gate, not cryptographic secrecy: coders must not access the deterministic builder or
its reconstruction inputs before their independent labels are locked.</p></div></section>

<section class="method-section" id="provenance"><h2>Reproducibility receipts</h2>
<p>The paid model outputs are immutable inputs. Everything shown here is either read from their
manifests or derived locally with validated keys.</p><div class="receipt-grid">{_provenance_cards(ai_data)}</div>
<details class="deep-section"><summary>Reproduce the deterministic projections</summary><div>
<p><code>arch -x86_64 .venv/bin/python -m presidential_profiles.site</code></p>
<p>This command consumes existing frozen artifacts; it does not submit a model request. Targeted
tests should precede the full suite, followed by JavaScript syntax checks, site validation,
frozen-artifact hash comparison, and Reggie Doctor.</p></div></details></section>

<section class="method-section" id="downloads"><h2>Continue into the evidence</h2>
<ul class="download-list"><li><a href="data-quality.html">Data Quality audit</a></li>
<li><a href="label-models.html">Model Comparison</a></li><li><a href="metrics.html">Metric Dictionary</a></li>
<li><a href="data/quality/second_model_by_era.csv" download>Second-model completion by Story era · CSV</a></li>
<li><a href="data/quality/second_model_by_speech.csv" download>Sample completion by speech · CSV</a></li>
<li><a href="data/quality/agreement_by_field_and_era.csv" download>Agreement, intervals, and prevalence · CSV</a></li>
<li><a href="data/quality/population_ledger.csv" download>Population ledger · CSV</a></li>
<li><a href="data/quality/entity_name_agreement.csv" download>Entity-name audit · CSV</a></li>
<li><a href="data/quality/public_chart_treatments.csv" download>Public-chart treatment map · CSV</a></li>
<li><a href="data/quality/manifest_v2.json" download>Data Quality v2 manifest · JSON</a></li></ul>
<ul class="download-list"><li><a href="data/grammar/era_pos.csv" download>Coarse POS era shares · CSV</a></li>
<li><a href="data/grammar/manifest.json" download>Coarse POS provenance · JSON</a></li></ul>
<ul class="download-list"><li><a href="data/validation-protocols-v1/manifest_v1.json" download>Planned validation manifest · JSON</a></li>
<li><a href="data/validation-protocols-v1/paragraph_selection_cells_v1.csv" download>Paragraph allocation cells · CSV</a></li>
<li><a href="data/validation-protocols-v1/speech_selection_cells_v1.csv" download>Factual-label allocation cells · CSV</a></li>
<li><a href="data/validation-protocols-v1/current_context_prompt_v1.json" download>Current-context prompt arm · JSON</a></li>
<li><a href="data/validation-protocols-v1/decade_hidden_prompt_v1.json" download>Decade-hidden prompt arm · JSON</a></li></ul>
<p>The full no-execution specification is also stored in
<a href="https://github.com/jacobfulfyll/presidential_profiles/blob/main/notes/data-trust-validation-protocols-v1.md">the public research notes</a>.</p></section>
</main><footer><p>Data: <a href="https://data.millercenter.org">Miller Center of Public Affairs,
University of Virginia</a>. Source code and research notes:
<a href="https://github.com/jacobfulfyll/presidential_profiles">Presidential Profiles</a>.</p></footer>
</body></html>"""


def write_methodology(ai_data: dict) -> None:
    path = REPO_ROOT / "docs" / "methodology.html"
    publish_grammar_artifacts(path.parent)
    path.write_text(render_methodology(ai_data))
    print("  wrote docs/methodology.html")
