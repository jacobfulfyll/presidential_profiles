"""Deterministic, non-executable planning for the Phase 2 annotation pilot.

This module reads the immutable canonical corpus and active Phase 1 generation,
constructs the approved stratified pilot selections, and publishes only
content-addressed selection and plan receipts.  It cannot initialize a campaign,
create an assignment, ingest a response, or invoke a model.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from . import annotation_backfill as backfill
from . import annotation_ledger as ledger
from . import annotation_workflow as workflow


PILOT_PLAN_VERSION = "annotation-refresh-pilot-plan-v1"
PILOT_SELECTION_VERSION = "annotation-refresh-pilot-selection-v1"
PILOT_ALGORITHM_VERSION = "annotation-refresh-pilot-sampling-v1-amended-482"
PILOT_SEED = "annotation-refresh-campaign-v1|pilot|20260724"
SELECTIONS_ROOT = ledger.LEDGER_ROOT / "selections"
CAMPAIGN_PLANS_ROOT = ledger.LEDGER_ROOT / "campaign_plans"

COMBINED_BUNDLE_ID = "paragraph_judgment_v2_candidate"
DIAGNOSTIC_BUNDLE_ID = "constituency_only_diagnostic"
COMBINED_BUNDLE_VERSION = "candidate-1"
DIAGNOSTIC_BUNDLE_VERSION = "candidate-1"

CORE_TARGET_COUNT = 482
ENRICHMENT_TARGET_COUNT = 96
DIAGNOSTIC_TARGET_COUNT = 144
COMBINED_TARGET_COUNT = 722
REFERENCE_TARGET_COUNT = 240
BLOCK_SIZE = 4

ERAS = (
    ("The founding", 1789, 1815),
    ("Expansion", 1816, 1849),
    ("Civil War & Reconstruction", 1850, 1877),
    ("The Gilded Age", 1878, 1900),
    ("Progressives & Depression", 1901, 1932),
    ("War & New Deal", 1933, 1945),
    ("The Cold War", 1946, 1988),
    ("Post-Cold War", 1989, 2016),
    ("The present era", 2017, 2026),
)

CUE_FAMILIES = {
    "national_public": ("people", "citizen", "american"),
    "geographic": ("state", "territory", "north", "south", "region"),
    "racial_status": (
        "black",
        "negro",
        "indian",
        "tribe",
        "immigrant",
        "freed",
        "slave",
    ),
    "economic_class": (
        "working class",
        "middle class",
        "poor",
        "wealthy",
        "capitalist",
    ),
    "occupation_industry": (
        "farmer",
        "laborer",
        "worker",
        "miner",
        "manufacturer",
        "business",
    ),
    "military_veteran": (
        "soldier",
        "sailor",
        "veteran",
        "armed forces",
        "military",
    ),
    "party_movement": (
        "democrat",
        "republican",
        "party",
        "union",
        "movement",
    ),
    "religious": ("church", "christian", "jew", "muslim", "religious"),
    "age_gender_family": (
        "women",
        "woman",
        "men",
        "children",
        "family",
        "families",
    ),
    "foreign_population_nation": (
        "nation",
        "peoples",
        "allies",
        "refugee",
    ),
    "institution_organization": (
        "congress",
        "court",
        "bank",
        "school",
        "organization",
    ),
    "relation_language": (
        "represent",
        "protect",
        "rights",
        "welfare",
        "benefit",
        "behalf",
        "authority",
        "duty",
    ),
}
CUE_PATTERNS = {
    family: tuple(
        re.compile(r"(?<!\w)" + re.escape(cue) + r"(?!\w)")
        for cue in cues
    )
    for family, cues in CUE_FAMILIES.items()
}

SHARED_CONTEXT_RENDER_POLICY = {
    "policy_version": "paragraph-shared-context-v1",
    "targets_per_assignment": 4,
    "target_partition": "persisted_nonoverlapping_position_blocks",
    "target_text_transmission": "once_per_target",
    "outer_context": "at_most_one_predecessor_and_one_successor_per_block",
    "speech_mixing": "prohibited",
    "trusted_keys_in_model_view": False,
}

EVALUATION_POLICY = {
    "policy_id": "ARCV1-evaluation-v1-amended-482",
    "contract_sections": ["13", "14"],
    "core_inference_frame": "complete_four_target_blocks_plus_sparse_census_cells",
    "whole_corpus_prevalence_claims": "prohibited",
    "legacy_current_agreement": "diagnostic_not_ground_truth",
    "comparison_changes_current_labels": False,
}

PROMOTION_POLICY = {
    "policy_id": "ARCV1-shadow-no-implicit-promotion-v1",
    "campaign_completion_promotes": False,
    "sealing_promotes": False,
    "comparison_promotes": False,
    "explicit_expected_state_transition_required": True,
    "grouped_entities_and_constituencies_atomic": True,
}


def _era_for_year(year: int) -> str:
    matches = [label for label, low, high in ERAS if low <= year <= high]
    if len(matches) != 1:
        raise ValueError(f"year {year} does not map to exactly one approved era")
    return matches[0]


def _hash_order(*parts: Any) -> str:
    return ledger.sha256_text(
        "\x1f".join(str(part) for part in (PILOT_SEED, *parts))
    )


def _key(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "doc_name": str(row["doc_name"]),
        "para_idx": int(row["para_idx"]),
    }


def _context_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {**_key(row), "text": str(row["text"])}


def _corpus_snapshot() -> tuple[pd.DataFrame, dict[str, Any]]:
    current_pointer = ledger.CURRENT_PATH.read_text(encoding="utf-8").strip()
    generation_dir = ledger.MATERIALIZED_ROOT / "generations" / current_pointer
    if not generation_dir.is_dir():
        raise ValueError("active materialized generation is missing")
    generation_path = generation_dir / "generation.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    correction_meta = json.loads(
        backfill.CORRECTION_META_PATH.read_text(encoding="utf-8")
    )
    paragraphs = pd.read_parquet(backfill.CANONICAL_PARAGRAPHS_PATH)
    speeches = pd.read_parquet(backfill.CANONICAL_SPEECHES_PATH)
    current = pd.read_parquet(generation_dir / "current_labels.parquet")
    factual = current[
        current["promotion_channel"].eq("primary")
        & current["label_type"].eq("speech_type")
        & current["event_role"].eq("value")
    ].copy()
    if len(factual) != len(speeches) or factual["canonical_doc_name"].duplicated().any():
        raise ValueError("active primary speech_type projection is not one-to-one")
    factual["speech_type"] = factual["raw_value_json"].map(json.loads)
    speech_meta = speeches[["doc_name", "year"]].merge(
        factual[["canonical_doc_name", "speech_type"]],
        left_on="doc_name",
        right_on="canonical_doc_name",
        how="left",
        validate="one_to_one",
    )
    speech_meta = speech_meta.drop(columns=["canonical_doc_name"])
    frame = paragraphs.merge(
        speech_meta, on="doc_name", how="left", validate="many_to_one"
    )
    if (
        len(frame) != 35_394
        or frame[["year", "speech_type"]].isna().any().any()
        or frame.duplicated(["doc_name", "para_idx"]).any()
    ):
        raise ValueError("canonical paragraph universe failed keyed validation")
    frame["year"] = frame["year"].astype(int)
    frame["era"] = frame["year"].map(_era_for_year)
    frame["genre"] = frame["speech_type"].astype(str)
    frame["decade"] = (frame["year"] // 10 * 10).astype(str) + "s"
    frame = frame.sort_values(["doc_name", "para_idx"]).reset_index(drop=True)
    expected_outputs = correction_meta["output_fingerprints"]
    observed_paragraph_hash = ledger.sha256_file(
        backfill.CANONICAL_PARAGRAPHS_PATH
    )
    observed_speech_hash = ledger.sha256_file(backfill.CANONICAL_SPEECHES_PATH)
    if observed_paragraph_hash != "sha256:" + expected_outputs[
        "canonical_paragraphs_v1.parquet"
    ]:
        raise ValueError("canonical paragraph artifact hash drift")
    if observed_speech_hash != "sha256:" + expected_outputs[
        "canonical_speeches_v1.parquet"
    ]:
        raise ValueError("canonical speech artifact hash drift")
    snapshot = {
        "fingerprint": "sha256:"
        + correction_meta["canonical_corpus_fingerprint"],
        "canonical_paragraphs": {
            "path": str(backfill.CANONICAL_PARAGRAPHS_PATH),
            "sha256": observed_paragraph_hash,
            "count": int(len(paragraphs)),
        },
        "canonical_speeches": {
            "path": str(backfill.CANONICAL_SPEECHES_PATH),
            "sha256": observed_speech_hash,
            "count": int(len(speeches)),
        },
        "active_generation_id": current_pointer,
        "active_generation_manifest_sha256": ledger.sha256_file(generation_path),
        "active_generation_tables": dict(generation["tables"]),
    }
    return frame, snapshot


def _subject_input(row: Mapping[str, Any], speech: Sequence[Mapping[str, Any]], position: int) -> dict[str, Any]:
    before = [_context_row(speech[position - 1])] if position > 0 else []
    after = [_context_row(speech[position + 1])] if position + 1 < len(speech) else []
    semantic = {
        "text": str(row["text"]),
        "context_before": [{"text": item["text"]} for item in before],
        "context_after": [{"text": item["text"]} for item in after],
        "decade": str(row["decade"]),
    }
    return {
        **dict(row),
        "context_before_json": ledger.canonical_json(before),
        "context_after_json": ledger.canonical_json(after),
        "bundle_input_sha256": ledger.sha256_text(ledger.canonical_json(semantic)),
    }


def _make_block(
    rows: Sequence[Mapping[str, Any]],
    *,
    speech: Sequence[Mapping[str, Any]],
    start_position: int,
    frame: str,
) -> dict[str, Any]:
    if not rows or len({str(row["doc_name"]) for row in rows}) != 1:
        raise ValueError("pilot block must be non-empty and speech-local")
    keys = [_key(row) for row in rows]
    first = rows[0]
    order_hash = _hash_order(
        first["doc_name"],
        int(first["para_idx"]),
        ledger.canonical_json(keys),
    )
    batch_id = "batch_" + order_hash.removeprefix("sha256:")
    subjects = [
        _subject_input(row, speech, start_position + offset)
        for offset, row in enumerate(rows)
    ]
    return {
        "batch_id": batch_id,
        "order_sha256": order_hash,
        "frame": frame,
        "era": str(first["era"]),
        "genre": str(first["genre"]),
        "doc_name": str(first["doc_name"]),
        "first_para_idx": int(first["para_idx"]),
        "subjects": subjects,
    }


def _blocks(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    complete: list[dict[str, Any]] = []
    tails: list[dict[str, Any]] = []
    for _, speech_frame in frame.groupby("doc_name", sort=True):
        speech = speech_frame.sort_values("para_idx").to_dict("records")
        for start in range(0, len(speech), BLOCK_SIZE):
            chunk = speech[start : start + BLOCK_SIZE]
            block = _make_block(
                chunk,
                speech=speech,
                start_position=start,
                frame="complete_block" if len(chunk) == BLOCK_SIZE else "speech_tail",
            )
            (complete if len(chunk) == BLOCK_SIZE else tails).append(block)
    return complete, tails


def _largest_remainder(
    capacities: Mapping[tuple[str, str], int], total: int
) -> dict[tuple[str, str], int]:
    allocation = {cell: 0 for cell in capacities}
    remaining = total
    original_weights = {
        cell: math.sqrt(capacity) for cell, capacity in capacities.items()
    }
    era_order = {label: index for index, (label, _, _) in enumerate(ERAS)}
    while remaining:
        available = [
            cell
            for cell, capacity in capacities.items()
            if allocation[cell] < capacity
        ]
        if not available:
            raise ValueError("core block allocation exhausted capacity")
        weight_total = sum(original_weights[cell] for cell in available)
        quotas = {
            cell: remaining * original_weights[cell] / weight_total
            for cell in available
        }
        floor_additions = {
            cell: min(
                capacities[cell] - allocation[cell], int(math.floor(quotas[cell]))
            )
            for cell in available
        }
        added = sum(floor_additions.values())
        if added:
            for cell, count in floor_additions.items():
                allocation[cell] += count
            remaining -= added
            continue
        ranked = sorted(
            available,
            key=lambda cell: (
                -(quotas[cell] - math.floor(quotas[cell])),
                era_order[cell[0]],
                cell[1],
            ),
        )
        for cell in ranked[:remaining]:
            allocation[cell] += 1
        remaining -= min(remaining, len(ranked))
    return allocation


def _select_core(
    frame: pd.DataFrame, complete: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cell_counts = (
        frame.groupby(["era", "genre"], sort=True)
        .size()
        .astype(int)
        .to_dict()
    )
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for block in complete:
        by_cell[(block["era"], block["genre"])].append(block)
    for blocks in by_cell.values():
        blocks.sort(key=lambda row: row["order_sha256"])
    sparse_cells = {cell: count for cell, count in cell_counts.items() if count < 4}
    sparse_blocks = []
    for cell, count in sorted(sparse_cells.items()):
        rows = frame[
            frame["era"].eq(cell[0]) & frame["genre"].eq(cell[1])
        ].sort_values(["doc_name", "para_idx"])
        if rows["doc_name"].nunique() != 1 or len(rows) != count:
            raise ValueError("sparse census cell cannot form one speech-local batch")
        speech_rows = frame[
            frame["doc_name"].eq(rows.iloc[0]["doc_name"])
        ].sort_values("para_idx")
        positions = {
            int(row.para_idx): index
            for index, row in enumerate(speech_rows.itertuples())
        }
        start = positions[int(rows.iloc[0]["para_idx"])]
        block = _make_block(
            rows.to_dict("records"),
            speech=speech_rows.to_dict("records"),
            start_position=start,
            frame="sparse_census",
        )
        if len(block["subjects"]) != count:
            raise AssertionError("sparse cell count drift")
        sparse_blocks.append(block)
    supported_cells = sorted(
        cell for cell, count in cell_counts.items() if count >= 4
    )
    if any(not by_cell[cell] for cell in supported_cells):
        raise ValueError("supported core cell has no complete block")
    selected = [by_cell[cell][0] for cell in supported_cells]
    capacities = {cell: len(by_cell[cell]) - 1 for cell in supported_cells}
    target_extra_blocks = (
        CORE_TARGET_COUNT
        - sum(len(block["subjects"]) for block in sparse_blocks)
        - BLOCK_SIZE * len(selected)
    ) // BLOCK_SIZE
    if (
        CORE_TARGET_COUNT
        - sum(len(block["subjects"]) for block in sparse_blocks)
        - BLOCK_SIZE * len(selected)
    ) % BLOCK_SIZE:
        raise ValueError("amended core target is not constructible")
    allocation = _largest_remainder(capacities, target_extra_blocks)
    for cell in supported_cells:
        selected.extend(by_cell[cell][1 : 1 + allocation[cell]])
    selected.extend(sparse_blocks)
    if sum(len(block["subjects"]) for block in selected) != CORE_TARGET_COUNT:
        raise AssertionError("core target count drift")
    selected_counts = defaultdict(int)
    for block in selected:
        if block["frame"] != "sparse_census":
            selected_counts[(block["era"], block["genre"])] += 1
    for block in selected:
        cell = (block["era"], block["genre"])
        if block["frame"] == "sparse_census":
            probability = 1.0
            population_blocks = 1
            selected_blocks = 1
        else:
            population_blocks = len(by_cell[cell])
            selected_blocks = selected_counts[cell]
            probability = selected_blocks / population_blocks
        block["inclusion_probability"] = probability
        block["cell_paragraph_count"] = cell_counts[cell]
        block["cell_complete_block_count"] = population_blocks
        block["cell_selected_block_count"] = selected_blocks
    return sorted(selected, key=lambda row: row["order_sha256"]), {
        "populated_cells": len(cell_counts),
        "sparse_cells": [
            {"era": cell[0], "genre": cell[1], "paragraphs": count}
            for cell, count in sorted(sparse_cells.items())
        ],
        "supported_cells": len(supported_cells),
        "additional_complete_blocks": target_extra_blocks,
        "allocation": [
            {
                "era": cell[0],
                "genre": cell[1],
                "additional_blocks": allocation[cell],
            }
            for cell in supported_cells
        ],
    }


def _cue_matches(text: str) -> set[str]:
    folded = text.casefold()
    return {
        family
        for family, patterns in CUE_PATTERNS.items()
        if any(pattern.search(folded) for pattern in patterns)
    }


def _select_enrichment(
    complete: Sequence[dict[str, Any]], excluded: set[str]
) -> list[dict[str, Any]]:
    candidates = [
        block for block in complete if block["batch_id"] not in excluded
    ]
    families_by_batch = {
        block["batch_id"]: _cue_matches(
            " ".join(str(row["text"]) for row in block["subjects"])
        )
        for block in candidates
    }
    coverage: set[tuple[str, str]] = set()
    selected = []
    for _ in range(ENRICHMENT_TARGET_COUNT // BLOCK_SIZE):
        if not candidates:
            raise ValueError("enrichment block capacity exhausted")
        scored = []
        for block in candidates:
            families = families_by_batch[block["batch_id"]]
            pairs = {(block["era"], family) for family in families}
            scored.append((len(pairs - coverage), block["order_sha256"], block, pairs))
        _, _, chosen, pairs = min(scored, key=lambda row: (-row[0], row[1]))
        chosen["frame"] = "rare_enrichment"
        chosen["cue_families"] = sorted(family for _, family in pairs)
        chosen["inclusion_probability"] = 1.0
        selected.append(chosen)
        coverage.update(pairs)
        candidates = [
            block for block in candidates if block["batch_id"] != chosen["batch_id"]
        ]
    return sorted(selected, key=lambda row: row["order_sha256"])


def _select_diagnostic(
    complete: Sequence[dict[str, Any]], excluded: set[str]
) -> list[dict[str, Any]]:
    candidates = [
        block for block in complete if block["batch_id"] not in excluded
    ]
    selected: list[dict[str, Any]] = []
    for era, _, _ in ERAS:
        era_blocks = sorted(
            [block for block in candidates if block["era"] == era],
            key=lambda row: row["order_sha256"],
        )
        by_genre: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for block in era_blocks:
            by_genre[block["genre"]].append(block)
        diverse = sorted(
            (blocks[0] for blocks in by_genre.values()),
            key=lambda row: row["order_sha256"],
        )[:4]
        chosen = list(diverse)
        if len(chosen) < 4:
            chosen_ids = {block["batch_id"] for block in chosen}
            chosen.extend(
                block
                for block in era_blocks
                if block["batch_id"] not in chosen_ids
            )
            chosen = chosen[:4]
        if len(chosen) != 4:
            raise ValueError(f"{era}: diagnostic block capacity is below four")
        for block in chosen:
            block["frame"] = "composition_diagnostic"
            block["inclusion_probability"] = 1.0
        selected.extend(chosen)
    if len(selected) * BLOCK_SIZE != DIAGNOSTIC_TARGET_COUNT:
        raise AssertionError("diagnostic target count drift")
    return sorted(selected, key=lambda row: row["order_sha256"])


def _identity_rows(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for block in blocks:
        for subject in block["subjects"]:
            rows.append(
                {
                    "subject_key": _key(subject),
                    "bundle_input_sha256": subject["bundle_input_sha256"],
                    "stratum": (
                        f"{block['frame']}|{block['era']}|{block['genre']}"
                    ),
                    "inclusion_probability": float(
                        block["inclusion_probability"]
                    ),
                }
            )
    return rows


def _rich_rows(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for block in blocks:
        for order, subject in enumerate(block["subjects"], start=1):
            rows.append(
                {
                    "doc_name": str(subject["doc_name"]),
                    "para_idx": int(subject["para_idx"]),
                    "text": str(subject["text"]),
                    "decade": str(subject["decade"]),
                    "context_before_json": subject["context_before_json"],
                    "context_after_json": subject["context_after_json"],
                    "bundle_input_sha256": subject["bundle_input_sha256"],
                    "stratum": (
                        f"{block['frame']}|{block['era']}|{block['genre']}"
                    ),
                    "inclusion_probability": float(
                        block["inclusion_probability"]
                    ),
                    "poststratification_weight": (
                        1.0 / float(block["inclusion_probability"])
                        if block["frame"] in {"complete_block", "sparse_census"}
                        else None
                    ),
                    "planned_batch_id": block["batch_id"],
                    "planned_batch_order": order,
                    "selection_frame": block["frame"],
                    "era": block["era"],
                    "genre": block["genre"],
                }
            )
    return sorted(rows, key=lambda row: (row["doc_name"], row["para_idx"]))


def _batch_receipts(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "planned_batch_id": block["batch_id"],
            "frame": block["frame"],
            "era": block["era"],
            "genre": block["genre"],
            "order_sha256": block["order_sha256"],
            "target_keys": [_key(row) for row in block["subjects"]],
            "outer_context_keys": {
                "before": [
                    _key(row)
                    for row in json.loads(
                        block["subjects"][0]["context_before_json"]
                    )
                ],
                "after": [
                    _key(row)
                    for row in json.loads(
                        block["subjects"][-1]["context_after_json"]
                    )
                ],
            },
        }
        for block in sorted(blocks, key=lambda row: row["order_sha256"])
    ]


def _selection_artifact(
    *,
    role: str,
    blocks: Sequence[Mapping[str, Any]],
    corpus_snapshot: Mapping[str, Any],
    bundle_receipts: Sequence[Mapping[str, Any]],
    parent_selection_id: str | None,
    sampling_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    from .annotation_refresh import subject_selection_id

    identity_rows = _identity_rows(blocks)
    selection_id = subject_selection_id(identity_rows)
    semantic = {
        "selection_manifest_version": PILOT_SELECTION_VERSION,
        "selection_role": role,
        "scope": "pilot",
        "seed": PILOT_SEED,
        "algorithm_version": PILOT_ALGORITHM_VERSION,
        "corpus_snapshot": dict(corpus_snapshot),
        "bundle_receipts": [dict(row) for row in bundle_receipts],
        "parent_selection_id": parent_selection_id,
        "n_subjects": len(identity_rows),
        "n_assignments": len(blocks),
        "identity_rows": sorted(
            identity_rows,
            key=lambda row: ledger.canonical_json(row["subject_key"]),
        ),
        "rows": _rich_rows(blocks),
        "batches": _batch_receipts(blocks),
        "sampling_receipt": dict(sampling_receipt),
        "evaluation_boundary": (
            "design_balanced_evaluation_frame_only_no_whole_corpus_prevalence"
        ),
    }
    selection_sha256 = ledger.sha256_text(ledger.canonical_json(semantic))
    return {
        **semantic,
        "selection_id": selection_id,
        "selection_sha256": selection_sha256,
    }


def _bundle_receipt(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "bundle_id": bundle["bundle_id"],
        "bundle_version": bundle["bundle_version"],
        "bundle_sha256": bundle["bundle_sha256"],
        "prompt_sha256": ledger.sha256_text(str(bundle["prompt_text"])),
        "response_schema_sha256": ledger.sha256_text(
            ledger.canonical_json(bundle["response_schema"])
        ),
        "context_policy_sha256": ledger.sha256_text(
            ledger.canonical_json(bundle["context_policy"])
        ),
    }


def _rendered_assignment(
    bundle: Mapping[str, Any], block: Mapping[str, Any]
) -> dict[str, Any]:
    targets = []
    for local_id, row in enumerate(block["subjects"], start=1):
        targets.append(
            {
                **dict(row),
                "subject_key": _key(row),
                "local_id": local_id,
                "context_before": json.loads(row["context_before_json"]),
                "context_after": json.loads(row["context_after_json"]),
            }
        )
    template_item = dict(bundle["response_template_item"])
    assignment = {
        "protocol_version": workflow.PROTOCOL_VERSION,
        "response_mode": "label_bundle_v1",
        "subject_type": bundle["subject_type"],
        "bundle_id": bundle["bundle_id"],
        "bundle_version": bundle["bundle_version"],
        "prompt_version": bundle["prompt_version"],
        "prompt": bundle["prompt_text"],
        "targets": targets,
        "response_schema": bundle["response_schema"],
        "response_template": {
            "annotations": [
                {**template_item, "local_id": target["local_id"]}
                for target in targets
            ]
        },
    }
    return workflow.render_assignment(assignment)


def _proxy_tokens(value: str) -> int:
    return math.ceil(len(value.encode("utf-8")) / 4)


def _token_receipt(
    *,
    combined_bundle: Mapping[str, Any],
    diagnostic_bundle: Mapping[str, Any],
    combined_blocks: Sequence[Mapping[str, Any]],
    diagnostic_blocks: Sequence[Mapping[str, Any]],
    all_blocks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def input_tokens(
        bundle: Mapping[str, Any], blocks: Sequence[Mapping[str, Any]]
    ) -> int:
        return sum(
            _proxy_tokens(
                json.dumps(
                    _rendered_assignment(bundle, block),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
            )
            for block in blocks
        )

    def transmitted_text_tokens(blocks: Sequence[Mapping[str, Any]]) -> int:
        total = 0
        for block in blocks:
            texts = [str(row["text"]) for row in block["subjects"]]
            texts.extend(
                str(row["text"])
                for row in json.loads(
                    block["subjects"][0]["context_before_json"]
                )
            )
            texts.extend(
                str(row["text"])
                for row in json.loads(
                    block["subjects"][-1]["context_after_json"]
                )
            )
            total += sum(_proxy_tokens(text) for text in texts)
        return total

    combined_one = input_tokens(combined_bundle, combined_blocks)
    diagnostic = input_tokens(diagnostic_bundle, diagnostic_blocks)
    complete_input = combined_one * 2 + diagnostic
    full_input = input_tokens(combined_bundle, all_blocks)
    full_text = transmitted_text_tokens(all_blocks)
    separate_baseline = len(all_blocks) * 7_700 + full_text * 7
    return {
        "estimator": "deterministic_ceil_utf8_bytes_divided_by_4",
        "runtime_tokenizer_available": False,
        "reasoning_and_internal_tokens_included": False,
        "render_serialization": "indent_2_sorted_keys_ensure_ascii_false",
        "pilot": {
            "combined_assignments_per_pass": len(combined_blocks),
            "diagnostic_assignments": len(diagnostic_blocks),
            "combined_input_tokens_per_pass": combined_one,
            "combined_output_tokens_per_pass": COMBINED_TARGET_COUNT * 160,
            "diagnostic_input_tokens": diagnostic,
            "diagnostic_output_tokens": DIAGNOSTIC_TARGET_COUNT * 45,
            "complete_input_tokens": complete_input,
            "complete_output_tokens": (
                COMBINED_TARGET_COUNT * 160 * 2
                + DIAGNOSTIC_TARGET_COUNT * 45
            ),
            "input_token_ceiling_10_percent": math.ceil(complete_input * 1.10),
            "preplan_input_estimate": 2_690_000,
            "preplan_change_fraction": (complete_input - 2_690_000)
            / 2_690_000,
            "human_rereview_if_absolute_change_exceeds": 0.10,
        },
        "full_paragraph_shadow": {
            "subjects": 35_394,
            "assignments": len(all_blocks),
            "combined_input_tokens": full_input,
            "combined_output_tokens": 35_394 * 160,
            "seven_separate_prompt_input_baseline": separate_baseline,
            "combined_input_savings": separate_baseline - full_input,
            "combined_input_savings_fraction": (
                (separate_baseline - full_input) / separate_baseline
            ),
            "shared_transmitted_text_tokens": full_text,
            "separate_static_prompt_tokens_per_assignment": 7_700,
        },
    }


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise FileExistsError(f"content-addressed artifact drift: {path}")
        return
    ledger.atomic_write_json(path, value)


def create_pilot_plan(
    *,
    bundles: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Publish exact selections and a non-executable plan; create no run state."""
    combined_bundle = bundles[(COMBINED_BUNDLE_ID, COMBINED_BUNDLE_VERSION)]
    diagnostic_bundle = bundles[
        (DIAGNOSTIC_BUNDLE_ID, DIAGNOSTIC_BUNDLE_VERSION)
    ]
    frame, corpus_snapshot = _corpus_snapshot()
    complete, tails = _blocks(frame)
    core, core_receipt = _select_core(frame, complete)
    core_ids = {block["batch_id"] for block in core}
    enrichment = _select_enrichment(complete, core_ids)
    excluded = core_ids | {block["batch_id"] for block in enrichment}
    diagnostic = _select_diagnostic(complete, excluded)
    combined = [*core, *enrichment, *diagnostic]
    combined_keys = [
        ledger.canonical_json(_key(row))
        for block in combined
        for row in block["subjects"]
    ]
    if (
        len(combined_keys) != COMBINED_TARGET_COUNT
        or len(set(combined_keys)) != COMBINED_TARGET_COUNT
    ):
        raise AssertionError("combined pilot is not exactly 722 distinct subjects")
    reference = [*enrichment, *diagnostic]
    receipts = [_bundle_receipt(combined_bundle), _bundle_receipt(diagnostic_bundle)]
    sampling_receipt = {
        "core": core_receipt,
        "counts": {
            "core_subjects": sum(len(row["subjects"]) for row in core),
            "enrichment_subjects": sum(
                len(row["subjects"]) for row in enrichment
            ),
            "diagnostic_subjects": sum(
                len(row["subjects"]) for row in diagnostic
            ),
            "combined_subjects": len(combined_keys),
            "reference_subjects": sum(
                len(row["subjects"]) for row in reference
            ),
            "full_block_frame_subjects": len(complete) * BLOCK_SIZE,
            "speech_tail_subjects": sum(len(row["subjects"]) for row in tails),
        },
        "cue_matching": (
            "unicode_casefolded_literal_phrase_with_nonword_boundaries"
        ),
        "enrichment_greedy_score": (
            "newly_covered_era_by_cue_family_pairs_then_sha256_order"
        ),
        "diagnostic_rule": (
            "four_blocks_per_era_maximize_genre_diversity_then_sha256_order"
        ),
    }
    combined_selection = _selection_artifact(
        role="combined_pilot_722",
        blocks=combined,
        corpus_snapshot=corpus_snapshot,
        bundle_receipts=[receipts[0]],
        parent_selection_id=None,
        sampling_receipt=sampling_receipt,
    )
    diagnostic_selection = _selection_artifact(
        role="composition_diagnostic_144",
        blocks=diagnostic,
        corpus_snapshot=corpus_snapshot,
        bundle_receipts=[receipts[1]],
        parent_selection_id=combined_selection["selection_id"],
        sampling_receipt=sampling_receipt,
    )
    reference_selection = _selection_artifact(
        role="adjudicated_reference_240",
        blocks=reference,
        corpus_snapshot=corpus_snapshot,
        bundle_receipts=receipts,
        parent_selection_id=combined_selection["selection_id"],
        sampling_receipt=sampling_receipt,
    )
    all_blocks = []
    for block in [*complete, *tails]:
        block["frame"] = "full_shadow"
        block["inclusion_probability"] = 1.0
        all_blocks.append(block)
    token_receipt = _token_receipt(
        combined_bundle=combined_bundle,
        diagnostic_bundle=diagnostic_bundle,
        combined_blocks=combined,
        diagnostic_blocks=diagnostic,
        all_blocks=all_blocks,
    )
    render_policy_sha256 = ledger.sha256_text(
        ledger.canonical_json(SHARED_CONTEXT_RENDER_POLICY)
    )
    evaluation_policy_sha256 = ledger.sha256_text(
        ledger.canonical_json(EVALUATION_POLICY)
    )
    promotion_policy_sha256 = ledger.sha256_text(
        ledger.canonical_json(PROMOTION_POLICY)
    )
    plan_semantic = {
        "pilot_plan_version": PILOT_PLAN_VERSION,
        "status": "non_executable_awaiting_ARCV1_G001_and_ARCV1_G002",
        "scope": "pilot",
        "corpus_snapshot": corpus_snapshot,
        "bundle_receipts": receipts,
        "selection_receipts": {
            "combined": {
                "selection_id": combined_selection["selection_id"],
                "selection_sha256": combined_selection["selection_sha256"],
                "n_subjects": combined_selection["n_subjects"],
                "n_assignments": combined_selection["n_assignments"],
            },
            "diagnostic": {
                "selection_id": diagnostic_selection["selection_id"],
                "selection_sha256": diagnostic_selection["selection_sha256"],
                "parent_selection_id": combined_selection["selection_id"],
                "n_subjects": diagnostic_selection["n_subjects"],
                "n_assignments": diagnostic_selection["n_assignments"],
            },
            "reference": {
                "selection_id": reference_selection["selection_id"],
                "selection_sha256": reference_selection["selection_sha256"],
                "parent_selection_id": combined_selection["selection_id"],
                "n_subjects": reference_selection["n_subjects"],
            },
        },
        "passes": [
            {
                "bundle_id": COMBINED_BUNDLE_ID,
                "pass_role": "blind_a",
                "selection_role": "combined",
            },
            {
                "bundle_id": COMBINED_BUNDLE_ID,
                "pass_role": "blind_b",
                "selection_role": "combined",
            },
            {
                "bundle_id": DIAGNOSTIC_BUNDLE_ID,
                "pass_role": "composition_diagnostic",
                "selection_role": "diagnostic",
            },
        ],
        "required_execution_profile": {
            "model_id": "pending_exact_runtime_receipt_ARCV1_G001",
            "model_identity_specificity": "exact_runtime_identifier",
            "reasoning_effort": "high",
            "same_model_and_effort_for_all_three_passes": True,
            "assignments_blocked_until_receipt": True,
        },
        "render_policy": SHARED_CONTEXT_RENDER_POLICY,
        "render_policy_sha256": render_policy_sha256,
        "evaluation_policy": EVALUATION_POLICY,
        "evaluation_policy_sha256": evaluation_policy_sha256,
        "promotion_policy": PROMOTION_POLICY,
        "promotion_policy_sha256": promotion_policy_sha256,
        "blindness": {
            "separate_immutable_runs": True,
            "identical_combined_selection_and_batch_plan": True,
            "sibling_responses_unavailable_to_labeling_sessions": True,
            "fresh_restricted_labeling_sessions_required": True,
        },
        "token_receipt": token_receipt,
        "approval_ids": [
            "ARCV1-RES001",
            "ARCV1-RES003",
            "ARCV1-RES004",
        ],
        "gates": {
            "ARCV1-G001": "pending_exact_future_label_runtime_receipt",
            "ARCV1-G002": "pending_human_plan_approval",
        },
        "prohibited_effects": [
            "campaign_initialization",
            "child_run_initialization",
            "assignment_creation",
            "model_execution",
            "response_ingestion",
            "sealing",
            "adjudication",
            "promotion",
            "materialization",
        ],
    }
    plan_sha256 = ledger.sha256_text(ledger.canonical_json(plan_semantic))
    plan = {
        **plan_semantic,
        "plan_id": "plan_" + plan_sha256.removeprefix("sha256:"),
        "plan_sha256": plan_sha256,
    }
    for selection in [
        combined_selection,
        diagnostic_selection,
        reference_selection,
    ]:
        _publish(
            SELECTIONS_ROOT / f"{selection['selection_id']}.json", selection
        )
    _publish(CAMPAIGN_PLANS_ROOT / f"{plan['plan_id']}.json", plan)
    return {
        "status": "planned_not_initialized",
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "plan_path": str(
            CAMPAIGN_PLANS_ROOT / f"{plan['plan_id']}.json"
        ),
        "selection_ids": {
            role: receipt["selection_id"]
            for role, receipt in plan["selection_receipts"].items()
        },
        "token_receipt": token_receipt,
    }
