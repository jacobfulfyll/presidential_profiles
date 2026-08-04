"""Resumable provider-neutral execution for the locked ARCV1 provisional plan.

The planner remains non-executable.  This module is an additive execution
boundary requiring the append-only ARCV1-G007 owner resolution, an exact
runtime receipt, and byte/content revalidation of every locked input before it
creates state.

Model invocation is deliberately outside this module.  Workers receive one
rendered claimed assignment and return an attempt payload plus runtime receipt.
SQLite supplies atomic claims, leases, token reservations, and unique
acceptance.  Conventional ledger JSONL files are materialized once, in
deterministic assignment order, immediately before audit and sealing.
"""

from __future__ import annotations

import hashlib
import argparse
import gzip
import json
import math
import os
import shutil
import sqlite3
import sys
import threading
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import annotation_ledger as ledger
from . import annotation_refresh as refresh
from . import annotation_refresh_provisional as provisional
from . import annotation_workflow as workflow


EXECUTION_VERSION = "annotation-refresh-provisional-execution-v1"
CONTROL_SCHEMA_VERSION = "annotation-refresh-provisional-control-v1"
RETRY_POLICY_VERSION = "annotation-refresh-provisional-retry-v1"
COMPOSITE_VERSION = "annotation-refresh-provisional-composite-v1"

PLAN_ID = (
    "pcplan_541b52ab42e978227017f4d2ca5f67d7e624e5a4d76b8b256b8fb371577427bf"
)
REUSE_ID = (
    "reuse_d5f9059121d080eab7ae31f0b4cdcf2d1be9c5bf8d3b1a6d411a4598f3ad7ec1"
)
FRESH_ID = (
    "fresh_1ab0bf0f22825aa61e83e54744d90bf0b23793e748e3d71d7801f8e8b9533cd3"
)
AUTHORIZATION_RESOLUTION_ID = "ARCV1-RES014"
RECOVERY_AMENDMENT_RESOLUTION_ID = "ARCV1-RES018"
VALIDATOR_REPAIR_RESOLUTION_ID = "ARCV1-RES020"
VALIDATOR_RETRY_POLICY_RESOLUTION_ID = "ARCV1-RES021"
ENHANCED_VALIDATOR_RETRY_RESOLUTION_ID = "ARCV1-RES023"
TARGETED_LITERAL_RETRY_RESOLUTION_ID = "ARCV1-RES025"
COMPLETION_AMENDMENT_RESOLUTION_ID = "ARCV1-RES026"
POSTCOMPLETION_DOCS_OVERLAY_RESOLUTION_ID = "ARCV1-RES027"
VALIDATOR_REPAIR_FEEDBACK = (
    "Validator-feedback supplement for this one authorized repair attempt: "
    "re-annotate the locked targets from scratch and do not infer or reproduce "
    "any prior response. For every constituency claim, group_text and "
    "evidence_span must each be exact substrings of the target paragraph, and "
    "group_text must occur literally inside evidence_span. Use certainty "
    "\"explicit\" only when group_text itself names the intended group and then "
    "resolved_referent must be empty. When group_text is a pronoun, demonstrative, "
    "or otherwise requires bounded context to identify the group, use certainty "
    "\"context_resolved\" and put the resolved group in resolved_referent. All "
    "other locked instructions, labels, targets, and schema rules are unchanged."
)
ENHANCED_VALIDATOR_REPAIR_FEEDBACK = (
    VALIDATOR_REPAIR_FEEDBACK
    + " Treat the target paragraph as an immutable character sequence. Copy "
    "group_text and evidence_span verbatim from it without correcting grammar, "
    "expanding or inserting contractions, adding or removing punctuation, "
    "changing capitalization, normalizing Unicode characters, or collapsing "
    "whitespace. Awkward or ungrammatical transcript wording must remain "
    "awkward or ungrammatical. For example, if the target literally says "
    "\"we protecting America at home\", use those exact characters and never "
    "change them to \"we're protecting America at home\"."
)
AUTHORIZATION_QUOTE = (
    "I explicitly authorize initialization and execution of the exact "
    "provisional-corpus plan:\n\n"
    + PLAN_ID
)
REQUIRED_MODEL_ID = "gpt-5.6-sol"
REQUIRED_REASONING_EFFORT = "high"

EXPECTED_ASSIGNMENTS = 9_186
EXPECTED_FRESH_SUBJECTS = 35_154
EXPECTED_FRESH_FIELDS = 246_078
EXPECTED_REUSED_SUBJECTS = 240
EXPECTED_REUSED_FIELDS = 1_680
EXPECTED_TOTAL_SUBJECTS = 35_394
EXPECTED_TOTAL_FIELDS = 247_758
EXPECTED_INPUT_TOKENS = 81_663_911
INPUT_TOKEN_CEILING = 89_830_303
COMPLETION_INPUT_TOKEN_CEILING = 93_000_000
COMPLETION_RETRY_ASSIGNMENT_INDICES = (
    844,
    2265,
    2266,
    3907,
    4282,
    6496,
    8895,
)
COMPLETION_AUTHORITY_QUOTE = (
    "I explicitly authorize Codex to continue ARCV1 campaign "
    "pcamp_ce0ff257543c3a342d96147b9e5e46953ca7016aab9bf29f1c6f1b573100939b "
    "and increase its input-proxy token ceiling from 89,830,303 to "
    "**93,000,000**.\n"
    "This authorization covers completion of the 109 currently nonterminal "
    "assignments under the existing locked gpt-5.6-sol/high payload, schema, "
    "and retry policies. It also authorizes up to three additional attempts "
    "for terminal assignments 844, 2265, 2266, 3907, 4282, 6496, and 8895. "
    "Infrastructure-failed assignments must use the unchanged locked payload. "
    "Assignments 844 and 3907 may receive an appended validator-feedback "
    "supplement, but the model must not normalize, correct, repunctuate, or "
    "otherwise alter transcript wording. No response may be locally repaired.\n"
    "I authorize transmission of these remaining and retry payloads—including "
    "their locked prompts, schemas, context artifacts, and Miller Center "
    "paragraph text—to OpenAI’s gpt-5.6-sol model at high reasoning, and "
    "authorize storing all attempts, responses, receipts, and validation "
    "evidence locally as provisional data.\n"
    "I also authorize recording the current docs/ tree—348 files, "
    "sha256:a7215af43f9c35378a45aae7fbd9dd251a415140f2e805569bcb43a7938a8af4—"
    "as the protected-state overlay without modifying or regenerating docs/.\n"
    "Continue automatically through the remaining assignments and authorized "
    "retries, then audit and seal the provisional run and create the "
    "non-promoted composite evidence artifact if every integrity requirement "
    "passes. This does not authorize promotion, production materialization, "
    "site generation, deployment, or modification of frozen paid artifacts "
    "in data/llm_annotations/. Stop only at a new genuine integrity blocker."
)
POSTCOMPLETION_DOCS_OVERLAY_AUTHORITY_QUOTE = (
    "I explicitly authorize Codex to append a protected-state overlay "
    "replacing the RES026-approved docs/ digest with the current observed "
    "tree—348 files, "
    "sha256:e96417db8b1b8e9b8bddb05f3755df498ba8842ab3fc5c097e38591152ba4705—"
    "without modifying or regenerating docs/. Continue automatically through "
    "audit, sealing, and creation and verification of the non-promoted "
    "composite evidence artifact. This does not authorize promotion, "
    "production materialization, site generation, deployment, or modification "
    "of frozen paid artifacts."
)

CAMPAIGNS_DIR = "provisional_corpus_campaigns"
COMPOSITES_DIR = "provisional_corpus_composites"
CONTROL_FILENAME = "provisional-control.sqlite3"
EXECUTION_MANIFEST_FILENAME = "provisional-execution-manifest.json"
RETRY_POLICY_FILENAME = "retry-policy.json"
BASELINE_FILENAME = "protected-state-baseline.json"
SUBJECT_FIELDS_FILENAME = "subject_label_values.jsonl"
AMENDMENTS_DIRNAME = "execution-amendments"
LITERAL_RETRY_GRANTS_DIRNAME = "literal-retry-grants"
_LITERAL_GRANT_THREAD_LOCK = threading.Lock()


class ProvisionalExecutionError(RuntimeError):
    """Base error for the provisional executor."""


class IntegrityDriftError(ProvisionalExecutionError):
    """A locked identity or protected side effect changed."""


class RetryExhaustedError(ProvisionalExecutionError):
    """An assignment exhausted the predeclared mechanical retry policy."""


class TokenCeilingError(ProvisionalExecutionError):
    """A new attempt would exceed the locked input-token ceiling."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return ledger.sha256_text(ledger.canonical_json(dict(value)))


def _artifact(
    semantic: Mapping[str, Any],
    *,
    prefix: str,
    id_field: str,
    sha_field: str,
) -> dict[str, Any]:
    sha = _canonical_sha(semantic)
    return {
        **dict(semantic),
        id_field: f"{prefix}_{sha.removeprefix('sha256:')}",
        sha_field: sha,
    }


def _publish_exact_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    compressed: bool = False,
) -> str:
    path = Path(path)
    if path.exists():
        existing = _load_json(path)
        if existing != dict(value):
            raise IntegrityDriftError(f"existing artifact drift: {path}")
        return "already_published"
    if not compressed:
        ledger.atomic_write_json(path, value)
        return "published"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with ledger.open_deterministic_gzip_text(temporary) as handle:
            json.dump(
                dict(value),
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            handle.write("\n")
        os.replace(temporary, path)
        ledger.fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "published"


def _load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    if compressed:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IntegrityDriftError(f"expected JSON object: {path}")
    return value


def _tree_digest(path: Path) -> dict[str, Any]:
    path = Path(path)
    digest = hashlib.sha256()
    count = 0
    if path.exists():
        for member in sorted(item for item in path.rglob("*") if item.is_file()):
            relative = member.relative_to(path).as_posix().encode("utf-8")
            payload_sha = hashlib.sha256(member.read_bytes()).hexdigest().encode(
                "ascii"
            )
            digest.update(relative)
            digest.update(b"\0")
            digest.update(payload_sha)
            digest.update(b"\n")
            count += 1
    return {"files": count, "sha256": "sha256:" + digest.hexdigest()}


def _protected_state(root: Path) -> dict[str, Any]:
    root = Path(root)
    current = root / "materialized" / "current"
    current_id = (
        current.read_text(encoding="utf-8").strip() if current.exists() else None
    )
    active_dir = root / "materialized" / str(current_id) if current_id else None
    repo_root = ledger.LEDGER_ROOT.parent.parent
    return {
        "snapshot_version": "arcv1-protected-state-v1",
        "active_materialization_pointer": {
            "value": current_id,
            "sha256": (
                ledger.sha256_bytes(current.read_bytes())
                if current.exists()
                else None
            ),
        },
        "active_generation_tree": (
            _tree_digest(active_dir)
            if active_dir is not None
            else {"files": 0, "sha256": None}
        ),
        "adjudications_tree": _tree_digest(root / "decisions" / "adjudications"),
        "promotions_tree": _tree_digest(root / "decisions" / "promotions"),
        "materialized_generation_directories": len(
            [
                path
                for path in (root / "materialized").iterdir()
                if path.is_dir()
            ]
        )
        if (root / "materialized").exists()
        else 0,
        "docs_tree": _tree_digest(repo_root / "docs"),
        "frozen_paid_tree": _tree_digest(repo_root / "data" / "llm_annotations"),
        "bundle_registry_sha256": ledger.sha256_bytes(
            refresh.BUNDLE_REGISTRY_PATH.read_bytes()
        ),
        "label_registry_sha256": ledger.sha256_bytes(
            ledger.REGISTRY_V2_PATH.read_bytes()
        ),
        "canonical_paragraphs_sha256": ledger.sha256_bytes(
            (
                repo_root
                / "data/corpus_corrections/canonical_paragraphs_v1.parquet"
            ).read_bytes()
        ),
        "canonical_speeches_sha256": ledger.sha256_bytes(
            (
                repo_root
                / "data/corpus_corrections/canonical_speeches_v1.parquet"
            ).read_bytes()
        ),
    }


def _retry_policy() -> dict[str, Any]:
    semantic = {
        "retry_policy_version": RETRY_POLICY_VERSION,
        "frozen_before_first_attempt": True,
        "maximum_model_invocations_per_assignment": 3,
        "initial_attempt_plus_retries": "1+2",
        "same_rendered_input_required": True,
        "repair_prompt": "prohibited",
        "repartition": "prohibited",
        "semantic_disagreement_retry": "prohibited",
        "retryable_mechanical_codes": [
            "incomplete_duplicate_or_unknown_local_id",
            "invocation_transient_failure",
            "json_parse_failure",
            "registered_special_validator_failure",
            "response_schema_failure",
        ],
        "non_retryable_stop_codes": [
            "accepted_response_exists",
            "identity_or_integrity_drift",
            "runtime_profile_drift",
            "token_ceiling_breach",
        ],
        "started_attempt_token_accounting": "conservative_reservation",
        "input_token_ceiling": INPUT_TOKEN_CEILING,
    }
    return _artifact(
        semantic,
        prefix="retry",
        id_field="retry_policy_id",
        sha_field="retry_policy_sha256",
    )


def _attempt_limit(
    connection: sqlite3.Connection, assignment_id: str
) -> int:
    maximum = int(_retry_policy()["maximum_model_invocations_per_assignment"])
    row = connection.execute(
        "SELECT value_json FROM meta WHERE key = 'attempt_limit_exceptions'"
    ).fetchone()
    if row is None:
        return maximum
    exceptions = json.loads(row["value_json"])
    value = exceptions.get(assignment_id)
    if value is None:
        return maximum
    if value not in {maximum + 1, maximum + 2, maximum + 3}:
        raise IntegrityDriftError("attempt-limit exception drift")
    return int(value)


def _validated_execution_amendments(run_dir: Path) -> list[dict[str, Any]]:
    directory = Path(run_dir) / AMENDMENTS_DIRNAME
    if not directory.exists():
        return []
    amendments = []
    for path in sorted(directory.glob("*.json")):
        amendment = provisional._validate_artifact(
            _load_json(path),
            prefix="pexam",
            id_field="amendment_id",
            sha_field="amendment_sha256",
        )
        if path.stem != amendment["amendment_id"]:
            raise IntegrityDriftError("execution amendment filename drift")
        amendments.append(amendment)
    return amendments


def _validated_literal_retry_grants(run_dir: Path) -> list[dict[str, Any]]:
    directory = Path(run_dir) / LITERAL_RETRY_GRANTS_DIRNAME
    if not directory.exists():
        return []
    grants = []
    assignment_ids: set[str] = set()
    for path in sorted(directory.glob("*.json")):
        grant = provisional._validate_artifact(
            _load_json(path),
            prefix="plitgrant",
            id_field="grant_id",
            sha_field="grant_sha256",
        )
        if path.stem != grant["grant_id"]:
            raise IntegrityDriftError("literal retry grant filename drift")
        assignment_id = str(grant.get("assignment_id", ""))
        if not assignment_id or assignment_id in assignment_ids:
            raise IntegrityDriftError("literal retry grant assignment drift")
        assignment_ids.add(assignment_id)
        grants.append(grant)
    return grants


def _declared_attempt_limit_exceptions(run_dir: Path) -> dict[str, int]:
    """Derive the exact retry exceptions from content-addressed amendments."""
    exceptions: dict[str, int] = {}
    amendments = _validated_execution_amendments(run_dir)
    for amendment in amendments:
        resolution_id = amendment.get("resolution_id")
        retry = amendment.get("retry_exception")
        if retry is not None:
            maximum = retry.get("amended_maximum_attempts")
            expected = {
                RECOVERY_AMENDMENT_RESOLUTION_ID: (4, 12),
                VALIDATOR_REPAIR_RESOLUTION_ID: (5, 12),
                VALIDATOR_RETRY_POLICY_RESOLUTION_ID: (4, 41),
                ENHANCED_VALIDATOR_RETRY_RESOLUTION_ID: (4, 8912),
            }.get(resolution_id)
            if (
                expected is None
                or maximum != expected[0]
                or retry.get("assignment_index") != expected[1]
                or not str(retry.get("assignment_id", "")).strip()
            ):
                raise IntegrityDriftError("attempt-limit amendment drift")
            assignment_id = retry["assignment_id"]
            exceptions[assignment_id] = max(
                exceptions.get(assignment_id, 0), int(maximum)
            )
        completion_retries = amendment.get("retry_exceptions")
        if completion_retries is not None:
            if resolution_id != COMPLETION_AMENDMENT_RESOLUTION_ID:
                raise IntegrityDriftError(
                    "completion retry exceptions lack RES026"
                )
            if not isinstance(completion_retries, list):
                raise IntegrityDriftError("completion retry exceptions drift")
            observed_indices: list[int] = []
            for completion_retry in completion_retries:
                if (
                    completion_retry.get("base_maximum_attempts") != 3
                    or completion_retry.get("amended_maximum_attempts") != 6
                    or completion_retry.get("additional_attempts") != 3
                    or not str(
                        completion_retry.get("assignment_id", "")
                    ).strip()
                ):
                    raise IntegrityDriftError(
                        "completion retry exception drift"
                    )
                observed_indices.append(
                    int(completion_retry["assignment_index"])
                )
                assignment_id = completion_retry["assignment_id"]
                exceptions[assignment_id] = max(
                    exceptions.get(assignment_id, 0), 6
                )
            if tuple(observed_indices) != COMPLETION_RETRY_ASSIGNMENT_INDICES:
                raise IntegrityDriftError(
                    "completion retry assignment indices drift"
                )
    grants = _validated_literal_retry_grants(run_dir)
    if grants and not any(
        amendment.get("resolution_id") == TARGETED_LITERAL_RETRY_RESOLUTION_ID
        for amendment in amendments
    ):
        raise IntegrityDriftError("literal retry grant lacks policy amendment")
    for grant in grants:
        if (
            grant.get("resolution_id") != TARGETED_LITERAL_RETRY_RESOLUTION_ID
            or grant.get("base_maximum_attempts") != 3
            or grant.get("amended_maximum_attempts") != 4
            or grant.get("additional_attempts") != 1
            or not grant.get("verified_literal_diagnostics")
            or grant.get("validator_feedback", {}).get("sha256")
            != ledger.sha256_text(
                str(grant.get("validator_feedback", {}).get("text", ""))
            )
        ):
            raise IntegrityDriftError("literal retry grant drift")
        assignment_id = grant["assignment_id"]
        if assignment_id in exceptions:
            raise IntegrityDriftError("duplicate literal retry exception")
        exceptions[assignment_id] = 4
    return exceptions


def _effective_input_token_ceiling(run_dir: Path) -> int:
    """Apply content-addressed ceiling amendments over the locked base."""
    ceiling = INPUT_TOKEN_CEILING
    amendments = _validated_execution_amendments(run_dir)
    pending = [
        amendment["input_token_ceiling_amendment"]
        for amendment in amendments
        if amendment.get("input_token_ceiling_amendment") is not None
    ]
    while pending:
        match_index = next(
            (
                index
                for index, amendment in enumerate(pending)
                if amendment.get("previous_value") == ceiling
            ),
            None,
        )
        if match_index is None:
            raise IntegrityDriftError("input-token ceiling amendment drift")
        amendment = pending.pop(match_index)
        replacement = amendment.get("replacement_value")
        if (
            not isinstance(replacement, int)
            or replacement <= ceiling
            or replacement > COMPLETION_INPUT_TOKEN_CEILING
        ):
            raise IntegrityDriftError("input-token ceiling replacement drift")
        ceiling = replacement
    return ceiling


def validator_feedback_for_attempt(
    campaign_id: str,
    assignment_id: str,
    attempt_no: int,
    *,
    root: Path = ledger.LEDGER_ROOT,
) -> str | None:
    """Return fixed guidance only for an explicitly authorized repair retry."""
    campaign = load_provisional_campaign(campaign_id, root=root)
    run_dir = ledger.work_dir(campaign["run_id"], root)
    amendments = _validated_execution_amendments(run_dir)
    resolution_ids = {
        row.get("resolution_id")
        for row in amendments
    }
    grants = {
        row["assignment_id"]: row
        for row in _validated_literal_retry_grants(run_dir)
    }
    grant = grants.get(assignment_id)
    if (
        grant is not None
        and attempt_no == int(grant["amended_maximum_attempts"])
    ):
        return str(grant["validator_feedback"]["text"])
    connection = _connect(run_dir)
    try:
        assignment = connection.execute(
            "SELECT assignment_index FROM assignments WHERE assignment_id = ?",
            (assignment_id,),
        ).fetchone()
        if assignment is None:
            raise IntegrityDriftError("feedback assignment is absent")
        if (
            assignment["assignment_index"] == 12
            and attempt_no == 5
            and VALIDATOR_REPAIR_RESOLUTION_ID in resolution_ids
        ):
            return VALIDATOR_REPAIR_FEEDBACK
        assignment_index = int(assignment["assignment_index"])
        previous = connection.execute(
            """
            SELECT status, validation_code, validation_error
            FROM attempts
            WHERE assignment_id = ? AND attempt_no = ?
            """,
            (assignment_id, attempt_no - 1),
        ).fetchone()
    finally:
        connection.close()
    constituency_failure = (
        previous is not None
        and previous["status"] == "schema_invalid"
        and previous["validation_code"]
        == "registered_special_validator_failure"
        and str(previous["validation_error"]).startswith(
            "constituency claim "
        )
    )
    if (
        constituency_failure
        and TARGETED_LITERAL_RETRY_RESOLUTION_ID in resolution_ids
        and "exact target substring" in str(previous["validation_error"])
    ):
        diagnostics = [
            row
            for row in diagnose_constituency_literal_mismatches(
                campaign_id,
                assignment_index,
                root=root,
            )
            if int(row["attempt_no"]) < attempt_no
        ]
        if any(
            row.get("exact_candidate_verified") is True
            for row in diagnostics
        ):
            return targeted_literal_retry_feedback(diagnostics)
    if (
        constituency_failure
        and ENHANCED_VALIDATOR_RETRY_RESOLUTION_ID in resolution_ids
    ):
        return ENHANCED_VALIDATOR_REPAIR_FEEDBACK
    if (
        constituency_failure
        and VALIDATOR_RETRY_POLICY_RESOLUTION_ID in resolution_ids
    ):
        return VALIDATOR_REPAIR_FEEDBACK
    return None


def _exact_contexts_containing(
    target_text: str, group_text: str
) -> list[str]:
    """Return deterministic exact sentence-like contexts containing a group."""
    if not group_text:
        return []
    contexts: list[str] = []
    start = 0
    while True:
        occurrence = target_text.find(group_text, start)
        if occurrence < 0:
            break
        left = occurrence
        while left > 0 and target_text[left - 1] not in ".!?\n":
            left -= 1
        while left < occurrence and target_text[left].isspace():
            left += 1
        right = occurrence + len(group_text)
        while right < len(target_text) and target_text[right] not in ".!?\n":
            right += 1
        if right < len(target_text) and target_text[right] in ".!?":
            right += 1
        value = target_text[left:right].strip()
        if value and value in target_text and value not in contexts:
            contexts.append(value)
        start = occurrence + 1
    return contexts


def _closest_exact_evidence_span(
    target_text: str,
    group_text: str,
    submitted_span: str,
) -> str:
    """Select an exact target context without normalizing any characters."""
    contexts = _exact_contexts_containing(target_text, group_text)
    if contexts:
        return max(
            contexts,
            key=lambda value: (
                SequenceMatcher(
                    None, submitted_span, value, autojunk=False
                ).ratio(),
                -abs(len(value) - len(submitted_span)),
                -target_text.index(value),
            ),
        )
    match = SequenceMatcher(
        None, submitted_span, target_text, autojunk=False
    ).find_longest_match()
    return target_text[match.b : match.b + match.size]


def diagnose_constituency_literal_mismatches(
    campaign_id: str,
    assignment_index: int,
    *,
    root: Path = ledger.LEDGER_ROOT,
) -> list[dict[str, Any]]:
    """Diagnose recorded non-literal spans without changing response state."""
    campaign = load_provisional_campaign(campaign_id, root=root)
    run_dir = ledger.work_dir(campaign["run_id"], root)
    connection = _connect(run_dir)
    try:
        assignment_row = connection.execute(
            "SELECT * FROM assignments WHERE assignment_index = ?",
            (assignment_index,),
        ).fetchone()
        if assignment_row is None:
            raise KeyError(assignment_index)
        attempts = list(
            connection.execute(
                """
                SELECT attempt_no, raw_output_json, validation_code,
                       validation_error
                FROM attempts
                WHERE assignment_id = ?
                ORDER BY attempt_no
                """,
                (assignment_row["assignment_id"],),
            )
        )
    finally:
        connection.close()
    assignment = _load_json(
        run_dir
        / "assignments"
        / f"{assignment_row['assignment_id']}.json"
    )
    rendered = workflow.render_assignment(assignment)
    targets = {
        int(target["local_id"]): str(target["text"])
        for target in rendered["targets"]
    }
    diagnostics: list[dict[str, Any]] = []
    for attempt in attempts:
        if (
            attempt["validation_code"]
            != "registered_special_validator_failure"
            or not str(attempt["validation_error"]).startswith(
                "constituency claim "
            )
            or attempt["raw_output_json"] is None
        ):
            continue
        try:
            raw_text = json.loads(attempt["raw_output_json"])
            response = json.loads(raw_text)
        except (TypeError, json.JSONDecodeError):
            continue
        for annotation in response.get("annotations", []):
            local_id = int(annotation["local_id"])
            target_text = targets[local_id]
            claims = annotation.get("constituencies", {}).get("claims", [])
            for claim_index, claim in enumerate(claims):
                submitted_span = str(claim.get("evidence_span", ""))
                if submitted_span in target_text:
                    continue
                group_text = str(claim.get("group_text", ""))
                exact_candidate = _closest_exact_evidence_span(
                    target_text,
                    group_text,
                    submitted_span,
                )
                diagnostics.append(
                    {
                        "assignment_index": assignment_index,
                        "assignment_id": assignment_row["assignment_id"],
                        "attempt_no": int(attempt["attempt_no"]),
                        "local_id": local_id,
                        "claim_index": claim_index,
                        "group_text": group_text,
                        "submitted_span": submitted_span,
                        "exact_candidate": exact_candidate,
                        "exact_candidate_verified": (
                            bool(exact_candidate)
                            and exact_candidate in target_text
                            and group_text in exact_candidate
                        ),
                    }
                )
    return diagnostics


def targeted_literal_retry_feedback(
    diagnostics: Sequence[Mapping[str, Any]],
) -> str:
    """Build deterministic target-only hints from recorded mismatch evidence."""
    verified = {
        (
            int(row["local_id"]),
            int(row["claim_index"]),
            str(row["group_text"]),
            str(row["exact_candidate"]),
        )
        for row in diagnostics
        if row.get("exact_candidate_verified") is True
    }
    if not verified:
        raise ValueError("no verified exact literal candidates")
    hints = " ".join(
        (
            f"For local_id {local_id}, a validator-verified exact target "
            f"substring containing group_text {json.dumps(group_text)} is "
            f"{json.dumps(candidate)}."
        )
        for local_id, _claim_index, group_text, candidate in sorted(verified)
    )
    return (
        ENHANCED_VALIDATOR_REPAIR_FEEDBACK
        + " Before returning each claim, perform a character-for-character "
        "membership check against the target. If a proposed evidence_span is "
        "not found literally, shorten it to an exact target substring that "
        "still contains group_text; never repair the target. "
        + hints
    )


def _effective_protected_baseline(run_dir: Path) -> dict[str, Any]:
    baseline = _load_json(Path(run_dir) / BASELINE_FILENAME)
    pending = [
        amendment["protected_state_overlay"]
        for amendment in _validated_execution_amendments(run_dir)
        if amendment.get("protected_state_overlay") is not None
    ]
    while pending:
        match_index = next(
            (
                index
                for index, overlay in enumerate(pending)
                if overlay.get("field") == "docs_tree"
                and baseline.get("docs_tree") == overlay.get("previous_value")
            ),
            None,
        )
        if match_index is None:
            raise IntegrityDriftError("protected-state amendment drift")
        overlay = pending.pop(match_index)
        baseline = {**baseline, "docs_tree": overlay["replacement_value"]}
    return baseline


def _validate_exact_runtime_receipt(
    receipt: Mapping[str, Any]
) -> dict[str, Any]:
    validated = refresh.validate_model_receipt(
        receipt, required_effort=REQUIRED_REASONING_EFFORT
    )
    if validated.get("model_id") != REQUIRED_MODEL_ID:
        raise IntegrityDriftError("runtime model differs from locked plan")
    required = ("session_id", "source")
    if any(not str(validated.get(field, "")).strip() for field in required):
        raise IntegrityDriftError("runtime receipt lacks session/source identity")
    return dict(validated)


def _validate_worker_runtime_receipt(
    receipt: Mapping[str, Any]
) -> dict[str, Any]:
    validated = _validate_exact_runtime_receipt(receipt)
    if validated.get("fresh_restricted_labeling_session") is not True:
        raise IntegrityDriftError("runtime session is not attested fresh/restricted")
    if validated.get("sibling_responses_inspected") is not False:
        raise IntegrityDriftError("runtime session inspected sibling responses")
    return dict(validated)


def _validate_execution_receipt(
    receipt: Mapping[str, Any],
    *,
    runtime_receipt: Mapping[str, Any],
    invocation_error_code: str | None,
) -> dict[str, Any]:
    value = dict(receipt)
    if (
        value.get("model_id") != REQUIRED_MODEL_ID
        or value.get("reasoning_effort") != REQUIRED_REASONING_EFFORT
        or value.get("session_id") != runtime_receipt.get("session_id")
        or not str(value.get("source", "")).strip()
    ):
        raise IntegrityDriftError("execution receipt runtime identity drift")
    expected_status = (
        "failed" if invocation_error_code is not None else "completed"
    )
    if value.get("status") != expected_status:
        raise IntegrityDriftError("execution receipt completion status drift")
    return value


def verify_locked_plan(
    *,
    runtime_receipt: Mapping[str, Any],
    root: Path = ledger.LEDGER_ROOT,
    review_queue_path: Path | None = None,
) -> dict[str, Any]:
    """Re-derive every locked identity without creating executable state."""
    root = Path(root)
    review_queue_path = review_queue_path or (
        ledger.LEDGER_ROOT.parent.parent
        / "notes/annotation-refresh-campaign-review-queue-v1.json"
    )
    plan_path = root / provisional.PROVISIONAL_PLANS_DIR / f"{PLAN_ID}.json"
    reuse_path = root / provisional.REUSE_MANIFESTS_DIR / f"{REUSE_ID}.json"
    fresh_path = root / provisional.FRESH_SELECTIONS_DIR / f"{FRESH_ID}.json"
    plan = provisional._validate_artifact(
        _load_json(plan_path),
        prefix="pcplan",
        id_field="plan_id",
        sha_field="plan_sha256",
    )
    reuse = provisional._validate_artifact(
        _load_json(reuse_path),
        prefix="reuse",
        id_field="reuse_manifest_id",
        sha_field="reuse_manifest_sha256",
    )
    fresh = provisional._validate_artifact(
        _load_json(fresh_path),
        prefix="fresh",
        id_field="fresh_selection_id",
        sha_field="fresh_selection_sha256",
    )
    if (
        plan["plan_id"] != PLAN_ID
        or reuse["reuse_manifest_id"] != REUSE_ID
        or fresh["fresh_selection_id"] != FRESH_ID
    ):
        raise IntegrityDriftError("locked provisional artifact identity drift")

    rebuilt_reuse = provisional.build_reuse_manifest(root=root)
    rebuilt_fresh, token_receipt, bundle_receipt = (
        provisional.build_fresh_selection(rebuilt_reuse, root=root)
    )
    rebuilt_plan = provisional.build_provisional_plan(
        rebuilt_reuse,
        rebuilt_fresh,
        token_receipt,
        bundle_receipt,
        approval_id=provisional.PLANNING_APPROVAL_ID,
        root=root,
    )
    if rebuilt_reuse != reuse or rebuilt_fresh != fresh or rebuilt_plan != plan:
        raise IntegrityDriftError("live evidence no longer reproduces locked plan")

    if (
        reuse["counts"]["subjects"] != EXPECTED_REUSED_SUBJECTS
        or reuse["counts"]["subject_label_entries"] != EXPECTED_REUSED_FIELDS
        or fresh["n_subjects"] != EXPECTED_FRESH_SUBJECTS
        or fresh["n_assignments"] != EXPECTED_ASSIGNMENTS
        or len(fresh["batches"]) != EXPECTED_ASSIGNMENTS
        or len(fresh["identity_rows"]) != EXPECTED_FRESH_SUBJECTS
        or plan["token_receipt"]["fresh_input_tokens"]
        != EXPECTED_INPUT_TOKENS
        or plan["token_receipt"]["fresh_input_token_ceiling_10_percent"]
        != INPUT_TOKEN_CEILING
        or plan["required_execution_profile"]["model_id"]
        != REQUIRED_MODEL_ID
        or plan["required_execution_profile"]["reasoning_effort"]
        != REQUIRED_REASONING_EFFORT
    ):
        raise IntegrityDriftError("locked plan counts/runtime/token receipt drift")
    if plan.get("production_eligible") is not False:
        raise IntegrityDriftError("provisional plan became production eligible")

    queue = _load_json(review_queue_path)
    resolutions = {
        row.get("resolution_id"): row for row in queue.get("resolutions", [])
    }
    resolution = resolutions.get(AUTHORIZATION_RESOLUTION_ID)
    if (
        resolution is None
        or resolution.get("applies_to_items") != ["ARCV1-G007"]
        or resolution.get("authority_quote") != AUTHORIZATION_QUOTE
        or resolution.get("decided_by") != "repository_user"
        or resolution.get("decision")
        != "approved_exact_provisional_corpus_execution"
    ):
        raise IntegrityDriftError("ARCV1-G007 owner authorization is absent/drifted")
    item = next(
        (
            row
            for row in queue.get("items", [])
            if row.get("item_id") == "ARCV1-G007"
        ),
        None,
    )
    if item is None or item.get("status") != "satisfied":
        raise IntegrityDriftError("ARCV1-G007 is not satisfied")
    exact_runtime = _validate_exact_runtime_receipt(runtime_receipt)
    return {
        "status": "verified",
        "plan": plan,
        "reuse": reuse,
        "fresh": fresh,
        "bundle_receipt": bundle_receipt,
        "authorization_resolution": resolution,
        "runtime_receipt": exact_runtime,
        "paths": {
            "plan": str(plan_path),
            "reuse": str(reuse_path),
            "fresh": str(fresh_path),
            "review_queue": str(review_queue_path),
        },
    }


def _run_id() -> str:
    return (
        "provisional-"
        + PLAN_ID.removeprefix("pcplan_")[:24]
        + "-paragraph-judgment-v2-candidate"
    )


def _campaign_semantic(
    preflight: Mapping[str, Any], retry_policy: Mapping[str, Any]
) -> dict[str, Any]:
    plan = preflight["plan"]
    resolution = preflight["authorization_resolution"]
    return {
        "provisional_execution_version": EXECUTION_VERSION,
        "scope": "provisional-corpus",
        "status": "initialized",
        "layer_status": "provisional",
        "production_eligible": False,
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "reuse_manifest_id": preflight["reuse"]["reuse_manifest_id"],
        "reuse_manifest_sha256": preflight["reuse"]["reuse_manifest_sha256"],
        "fresh_selection_id": preflight["fresh"]["fresh_selection_id"],
        "fresh_selection_sha256": preflight["fresh"][
            "fresh_selection_sha256"
        ],
        "authorization_resolution_id": resolution["resolution_id"],
        "authorization_resolution_sha256": _canonical_sha(resolution),
        "required_execution_profile": plan["required_execution_profile"],
        "retry_policy_id": retry_policy["retry_policy_id"],
        "retry_policy_sha256": retry_policy["retry_policy_sha256"],
        "run_id": _run_id(),
        "n_runs": 1,
        "n_assignments": EXPECTED_ASSIGNMENTS,
        "n_subjects": EXPECTED_FRESH_SUBJECTS,
        "input_token_estimate": EXPECTED_INPUT_TOKENS,
        "input_token_ceiling": INPUT_TOKEN_CEILING,
        "prohibited_effects": [
            "adjudication_write",
            "deployment",
            "frozen_paid_artifact_modification",
            "materialized_generation",
            "active_materialization_pointer_change",
            "promotion",
            "site_generation",
        ],
    }


def _campaign_path(campaign_id: str, root: Path) -> Path:
    if not ledger.ID_PREFIX_RE.fullmatch(campaign_id) or not campaign_id.startswith(
        "pcamp_"
    ):
        raise ValueError("invalid provisional campaign id")
    return Path(root) / CAMPAIGNS_DIR / campaign_id


def _connect(run_dir: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        Path(run_dir) / CONTROL_FILENAME,
        timeout=30.0,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def _initialize_control(
    run_dir: Path,
    *,
    campaign_id: str,
    assignments: Sequence[Mapping[str, Any]],
) -> None:
    connection = _connect(run_dir)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = WAL;
            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );
            CREATE TABLE assignments (
                assignment_index INTEGER PRIMARY KEY,
                assignment_id TEXT NOT NULL UNIQUE,
                planned_batch_id TEXT NOT NULL UNIQUE,
                rendered_sha256 TEXT NOT NULL,
                input_proxy_tokens INTEGER NOT NULL CHECK(input_proxy_tokens > 0),
                status TEXT NOT NULL CHECK(status IN (
                    'available', 'claimed', 'accepted', 'terminal_failed'
                )),
                active_claim_id TEXT,
                lease_expires_at TEXT,
                accepted_attempt_id TEXT
            );
            CREATE TABLE sessions (
                session_receipt_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL UNIQUE,
                receipt_json TEXT NOT NULL
            );
            CREATE TABLE claim_events (
                event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_event_id TEXT NOT NULL UNIQUE,
                assignment_id TEXT NOT NULL,
                claim_id TEXT NOT NULL,
                session_receipt_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE attempts (
                attempt_id TEXT PRIMARY KEY,
                assignment_id TEXT NOT NULL,
                attempt_no INTEGER NOT NULL,
                session_receipt_id TEXT NOT NULL,
                rendered_sha256 TEXT NOT NULL,
                input_proxy_tokens INTEGER NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                submission_sha256 TEXT,
                raw_output_json TEXT,
                validation_code TEXT,
                validation_error TEXT,
                execution_receipt_json TEXT,
                accepted_package_path TEXT,
                UNIQUE(assignment_id, attempt_no)
            );
            CREATE TABLE attempt_events (
                event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_event_id TEXT NOT NULL UNIQUE,
                attempt_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        connection.execute("BEGIN IMMEDIATE")
        connection.executemany(
            "INSERT INTO meta(key, value_json) VALUES (?, ?)",
            [
                ("control_schema_version", ledger.canonical_json(CONTROL_SCHEMA_VERSION)),
                ("campaign_id", ledger.canonical_json(campaign_id)),
                ("run_id", ledger.canonical_json(_run_id())),
                ("execution_status", ledger.canonical_json("initialized")),
            ],
        )
        connection.executemany(
            """
            INSERT INTO assignments(
                assignment_index, assignment_id, planned_batch_id,
                rendered_sha256, input_proxy_tokens, status
            ) VALUES (?, ?, ?, ?, ?, 'available')
            """,
            [
                (
                    row["assignment_index"],
                    row["assignment_id"],
                    row["planned_batch_id"],
                    row["rendered_sha256"],
                    row["input_proxy_tokens"],
                )
                for row in assignments
            ],
        )
        connection.execute("COMMIT")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def _assignment_proxy_tokens(rendered: Mapping[str, Any]) -> int:
    serialized = json.dumps(
        rendered, indent=2, sort_keys=True, ensure_ascii=False
    )
    return math.ceil(len(serialized.encode("utf-8")) / 4)


def _precreate_assignments(
    run_id: str,
    *,
    candidate_root: Path,
    fresh: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> list[dict[str, Any]]:
    run_dir = ledger.work_dir(run_id, candidate_root)
    run = workflow._work_run(run_id, candidate_root)
    specs = workflow._specs_for_run(run, ledger.REGISTRY_V2_PATH)
    subjects = ledger.read_jsonl(run_dir / "subjects.jsonl")
    subjects_by_planned: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for subject in subjects:
        planned_batch_id = subject.get("planned_batch_id")
        if not isinstance(planned_batch_id, str) or not planned_batch_id:
            raise IntegrityDriftError("run subject lacks a planned batch ID")
        subjects_by_planned[planned_batch_id].append(subject)
    by_planned = {
        row["planned_batch_id"]: row for row in fresh["batches"]
    }
    if (
        len(by_planned) != EXPECTED_ASSIGNMENTS
        or set(subjects_by_planned) != set(by_planned)
    ):
        raise IntegrityDriftError("fresh batch IDs are not unique and complete")
    rows = []
    proxy_total = 0
    seen_subjects: set[str] = set()
    for assignment_index, batch in enumerate(fresh["batches"]):
        selected = sorted(
            subjects_by_planned[str(batch["planned_batch_id"])],
            key=lambda row: int(row["planned_batch_order"]),
        )
        assignment = workflow.build_assignment(
            run=run,
            specs=specs,
            bundle=bundle,
            subjects=subjects,
            selected_subjects=selected,
            context_window=int(
                bundle["context_policy"].get("neighbors_each_side", 0)
            ),
            assigned_at=ledger.utc_now(),
        )
        assignment_path = (
            run_dir
            / "assignments"
            / f"{assignment['assignment_id']}.json"
        )
        if assignment_path.exists():
            raise IntegrityDriftError("duplicate deterministic assignment ID")
        ledger.atomic_write_json(assignment_path, assignment)
        rendered = workflow.render_assignment(assignment)
        target_keys = [target["subject_key"] for target in assignment["targets"]]
        if (
            target_keys != batch["target_keys"]
            or [
                target["source_text_sha256"]
                for target in assignment["targets"]
            ]
            != batch["target_source_text_sha256"]
            or assignment["bundle_sha256"] != bundle["bundle_sha256"]
        ):
            raise IntegrityDriftError("assignment differs from locked batch")
        target_ids = [target["subject_id"] for target in assignment["targets"]]
        if seen_subjects.intersection(target_ids):
            raise IntegrityDriftError("fresh subject assigned more than once")
        seen_subjects.update(target_ids)
        proxy = _assignment_proxy_tokens(rendered)
        proxy_total += proxy
        rows.append(
            {
                "assignment_index": assignment_index,
                "assignment_id": assignment["assignment_id"],
                "planned_batch_id": batch["planned_batch_id"],
                "rendered_sha256": ledger.sha256_text(
                    json.dumps(
                        rendered,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                ),
                "input_proxy_tokens": proxy,
            }
        )
    if (
        len(rows) != EXPECTED_ASSIGNMENTS
        or len(seen_subjects) != EXPECTED_FRESH_SUBJECTS
        or proxy_total != EXPECTED_INPUT_TOKENS
    ):
        raise IntegrityDriftError(
            "precreated assignment count/coverage/token total drift"
        )
    ledger.write_jsonl(run_dir / "assignment-index.jsonl", rows)
    return rows


def load_provisional_campaign(
    campaign_id: str, *, root: Path = ledger.LEDGER_ROOT
) -> dict[str, Any]:
    path = _campaign_path(campaign_id, root) / "campaign.json"
    value = _load_json(path)
    semantic = {
        key: item
        for key, item in value.items()
        if key not in {"campaign_id", "campaign_sha256", "initialized_at"}
    }
    expected_sha = _canonical_sha(semantic)
    if (
        value.get("campaign_sha256") != expected_sha
        or value.get("campaign_id")
        != "pcamp_" + expected_sha.removeprefix("sha256:")
    ):
        raise IntegrityDriftError("provisional campaign identity drift")
    run_id = value["run_id"]
    if not ledger.work_dir(run_id, root).is_dir() and run_id not in ledger.sealed_runs(
        root
    ):
        raise IntegrityDriftError("provisional campaign run is missing")
    return value


def initialize_provisional_execution(
    *,
    runtime_receipt: Mapping[str, Any],
    initialized_by: str,
    root: Path = ledger.LEDGER_ROOT,
    review_queue_path: Path | None = None,
) -> dict[str, Any]:
    """Initialize exactly one campaign/run and all 9,186 locked assignments."""
    root = Path(root)
    preflight = verify_locked_plan(
        runtime_receipt=runtime_receipt,
        root=root,
        review_queue_path=review_queue_path,
    )
    retry_policy = _retry_policy()
    semantic = _campaign_semantic(preflight, retry_policy)
    campaign = _artifact(
        semantic,
        prefix="pcamp",
        id_field="campaign_id",
        sha_field="campaign_sha256",
    )
    campaign_id = campaign["campaign_id"]
    final_campaign = _campaign_path(campaign_id, root)
    campaigns_root = root / CAMPAIGNS_DIR
    existing_campaigns = (
        [path for path in campaigns_root.iterdir() if path.is_dir()]
        if campaigns_root.exists()
        else []
    )
    if final_campaign.exists():
        loaded = load_provisional_campaign(campaign_id, root=root)
        if {
            key: value
            for key, value in loaded.items()
            if key != "initialized_at"
        } != campaign:
            raise IntegrityDriftError("existing provisional campaign drift")
        return {
            "status": "already_initialized",
            "campaign_id": campaign_id,
            "run_id": loaded["run_id"],
        }
    if existing_campaigns:
        raise IntegrityDriftError(
            "another provisional execution campaign already exists"
        )
    if not str(initialized_by).strip():
        raise ValueError("initialized_by is required")

    candidate = campaigns_root / f".{campaign_id}.{os.getpid()}.tmp"
    if candidate.exists():
        raise FileExistsError(candidate)
    candidate_root = candidate / "candidate-ledger"
    run_id = campaign["run_id"]
    bundle = refresh.resolve_bundle(
        provisional.COMBINED_BUNDLE_ID,
        provisional.COMBINED_BUNDLE_VERSION,
        registry_path=refresh.BUNDLE_REGISTRY_PATH,
        label_registry_path=ledger.REGISTRY_V2_PATH,
    )
    baseline = _protected_state(root)
    try:
        refresh.initialize_bundle_run(
            bundle_id=bundle["bundle_id"],
            bundle_version=bundle["bundle_version"],
            run_id=run_id,
            subjects_path=Path(preflight["paths"]["fresh"]),
            model_receipt=preflight["runtime_receipt"],
            reasoning_effort=REQUIRED_REASONING_EFFORT,
            annotator_id="arcv1-provisional-fresh",
            root=candidate_root,
            bundle_registry_path=refresh.BUNDLE_REGISTRY_PATH,
            label_registry_path=ledger.REGISTRY_V2_PATH,
            canonical_corpus_fingerprint=preflight["plan"]["corpus_snapshot"][
                "fingerprint"
            ],
            selection_artifact_id=FRESH_ID,
            authorization_review_item_id="ARCV1-G007",
        )
        run_dir = ledger.work_dir(run_id, candidate_root)
        (run_dir / "accepted-responses").mkdir()
        (run_dir / "execution-session-receipts").mkdir()
        (run_dir / "attempt-receipts").mkdir()
        _publish_exact_json(run_dir / RETRY_POLICY_FILENAME, retry_policy)
        _publish_exact_json(run_dir / BASELINE_FILENAME, baseline)
        execution_manifest = _artifact(
            {
                "execution_manifest_version": EXECUTION_VERSION,
                "campaign_id": campaign_id,
                "run_id": run_id,
                "plan_id": PLAN_ID,
                "plan_sha256": preflight["plan"]["plan_sha256"],
                "fresh_selection_id": FRESH_ID,
                "fresh_selection_sha256": preflight["fresh"][
                    "fresh_selection_sha256"
                ],
                "reuse_manifest_id": REUSE_ID,
                "reuse_manifest_sha256": preflight["reuse"][
                    "reuse_manifest_sha256"
                ],
                "authorization_resolution_id": AUTHORIZATION_RESOLUTION_ID,
                "authorization_resolution_sha256": _canonical_sha(
                    preflight["authorization_resolution"]
                ),
                "required_model_id": REQUIRED_MODEL_ID,
                "required_reasoning_effort": REQUIRED_REASONING_EFFORT,
                "retry_policy_id": retry_policy["retry_policy_id"],
                "retry_policy_sha256": retry_policy["retry_policy_sha256"],
                "input_token_estimate": EXPECTED_INPUT_TOKENS,
                "input_token_ceiling": INPUT_TOKEN_CEILING,
                "n_assignments": EXPECTED_ASSIGNMENTS,
                "n_subjects": EXPECTED_FRESH_SUBJECTS,
                "production_eligible": False,
            },
            prefix="pexec",
            id_field="execution_manifest_id",
            sha_field="execution_manifest_sha256",
        )
        _publish_exact_json(
            run_dir / EXECUTION_MANIFEST_FILENAME, execution_manifest
        )
        assignment_rows = _precreate_assignments(
            run_id,
            candidate_root=candidate_root,
            fresh=preflight["fresh"],
            bundle=bundle,
        )
        _initialize_control(
            run_dir, campaign_id=campaign_id, assignments=assignment_rows
        )
        initialized = {
            **campaign,
            "initialized_at": _utc_now(),
        }
        _publish_exact_json(candidate / "campaign.json", initialized)
        with ledger.ledger_lock(root):
            if final_campaign.exists() or ledger.work_dir(run_id, root).exists():
                raise FileExistsError("provisional campaign/run appeared concurrently")
            campaigns_root.mkdir(parents=True, exist_ok=True)
            (root / "work").mkdir(parents=True, exist_ok=True)
            os.replace(
                ledger.work_dir(run_id, candidate_root),
                ledger.work_dir(run_id, root),
            )
            (candidate_root / "work").rmdir()
            candidate_root.rmdir()
            os.replace(candidate, final_campaign)
            ledger.fsync_directory(root / "work")
            ledger.fsync_directory(campaigns_root)
    except Exception:
        # Preserve candidates/orphans for explicit inspection.
        raise
    return {
        "status": "initialized",
        "campaign_id": campaign_id,
        "run_id": run_id,
        "assignments": EXPECTED_ASSIGNMENTS,
        "subjects": EXPECTED_FRESH_SUBJECTS,
    }


def _insert_event(
    connection: sqlite3.Connection,
    table: str,
    *,
    prefix: str,
    identity: Mapping[str, Any],
    values: Sequence[Any],
) -> str:
    event_id = ledger.content_id(prefix, identity)
    if table == "claim_events":
        connection.execute(
            """
            INSERT INTO claim_events(
                claim_event_id, assignment_id, claim_id, session_receipt_id,
                event_type, occurred_at, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (event_id, *values),
        )
    elif table == "attempt_events":
        connection.execute(
            """
            INSERT INTO attempt_events(
                attempt_event_id, attempt_id, event_type, occurred_at, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (event_id, *values),
        )
    else:  # pragma: no cover - internal vocabulary
        raise AssertionError(table)
    return event_id


def _register_session(
    connection: sqlite3.Connection, receipt: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    validated = _validate_worker_runtime_receipt(receipt)
    receipt_id = ledger.content_id("rcpt", validated)
    existing = connection.execute(
        "SELECT receipt_json FROM sessions WHERE session_receipt_id = ?",
        (receipt_id,),
    ).fetchone()
    if existing is not None:
        if json.loads(existing["receipt_json"]) != validated:
            raise IntegrityDriftError("session receipt identity drift")
        return receipt_id, validated
    try:
        connection.execute(
            """
            INSERT INTO sessions(session_receipt_id, session_id, receipt_json)
            VALUES (?, ?, ?)
            """,
            (
                receipt_id,
                validated["session_id"],
                ledger.canonical_json(validated),
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise IntegrityDriftError(
            "one runtime session supplied multiple receipts"
        ) from exc
    return receipt_id, validated


def _expire_claims(
    connection: sqlite3.Connection, *, now: str
) -> None:
    expired = connection.execute(
        """
        SELECT assignment_id, active_claim_id
        FROM assignments
        WHERE status = 'claimed' AND lease_expires_at <= ?
        ORDER BY assignment_index
        """,
        (now,),
    ).fetchall()
    for row in expired:
        session = connection.execute(
            """
            SELECT session_receipt_id FROM claim_events
            WHERE claim_id = ? ORDER BY event_seq LIMIT 1
            """,
            (row["active_claim_id"],),
        ).fetchone()
        payload = {
            "assignment_id": row["assignment_id"],
            "claim_id": row["active_claim_id"],
            "event_type": "claim_expired",
            "occurred_at": now,
        }
        _insert_event(
            connection,
            "claim_events",
            prefix="clmev",
            identity=payload,
            values=(
                row["assignment_id"],
                row["active_claim_id"],
                session["session_receipt_id"],
                "claim_expired",
                now,
                ledger.canonical_json(payload),
            ),
        )
        connection.execute(
            """
            UPDATE assignments
            SET status = 'available', active_claim_id = NULL,
                lease_expires_at = NULL
            WHERE assignment_id = ? AND status = 'claimed'
            """,
            (row["assignment_id"],),
        )


def claim_provisional_assignment(
    campaign_id: str,
    *,
    runtime_receipt: Mapping[str, Any],
    range_start: int = 0,
    range_end: int = EXPECTED_ASSIGNMENTS,
    lease_seconds: int = 1800,
    root: Path = ledger.LEDGER_ROOT,
    now: str | None = None,
) -> dict[str, Any]:
    """Atomically claim one unaccepted assignment and return only its rendering."""
    if (
        range_start < 0
        or range_end > EXPECTED_ASSIGNMENTS
        or range_start >= range_end
    ):
        raise ValueError("invalid assignment range")
    if not 60 <= lease_seconds <= 7200:
        raise ValueError("lease_seconds must be between 60 and 7200")
    campaign = load_provisional_campaign(campaign_id, root=root)
    run_dir = ledger.work_dir(campaign["run_id"], root)
    now = now or _utc_now()
    expires = (
        _parse_utc(now) + timedelta(seconds=lease_seconds)
    ).isoformat(timespec="microseconds").replace("+00:00", "Z")
    connection = _connect(run_dir)
    try:
        connection.execute("BEGIN IMMEDIATE")
        session_id, _ = _register_session(connection, runtime_receipt)
        _expire_claims(connection, now=now)
        row = connection.execute(
            """
            SELECT * FROM assignments
            WHERE assignment_index >= ? AND assignment_index < ?
              AND status = 'available'
            ORDER BY assignment_index
            LIMIT 1
            """,
            (range_start, range_end),
        ).fetchone()
        if row is None:
            connection.execute("COMMIT")
            return {
                "status": "range_complete",
                "campaign_id": campaign_id,
                "range_start": range_start,
                "range_end": range_end,
            }
        claim_semantic = {
            "campaign_id": campaign_id,
            "run_id": campaign["run_id"],
            "assignment_id": row["assignment_id"],
            "assignment_index": row["assignment_index"],
            "session_receipt_id": session_id,
            "claimed_at": now,
            "lease_expires_at": expires,
        }
        claim_id = ledger.content_id("clm", claim_semantic)
        updated = connection.execute(
            """
            UPDATE assignments
            SET status = 'claimed', active_claim_id = ?, lease_expires_at = ?
            WHERE assignment_id = ? AND status = 'available'
            """,
            (claim_id, expires, row["assignment_id"]),
        )
        if updated.rowcount != 1:
            raise IntegrityDriftError("atomic assignment claim was lost")
        payload = {**claim_semantic, "claim_id": claim_id, "event_type": "claimed"}
        _insert_event(
            connection,
            "claim_events",
            prefix="clmev",
            identity=payload,
            values=(
                row["assignment_id"],
                claim_id,
                session_id,
                "claimed",
                now,
                ledger.canonical_json(payload),
            ),
        )
        connection.execute("COMMIT")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    assignment = _load_json(
        run_dir / "assignments" / f"{row['assignment_id']}.json"
    )
    rendered = workflow.render_assignment(assignment)
    rendered_sha = ledger.sha256_text(
        json.dumps(rendered, indent=2, sort_keys=True, ensure_ascii=False)
    )
    if rendered_sha != row["rendered_sha256"]:
        raise IntegrityDriftError("claimed assignment render drift")
    return {
        "status": "claimed",
        "campaign_id": campaign_id,
        "run_id": campaign["run_id"],
        "assignment_index": row["assignment_index"],
        "assignment_id": row["assignment_id"],
        "claim_id": claim_id,
        "lease_expires_at": expires,
        "session_receipt_id": session_id,
        "rendered_assignment": rendered,
        "rendered_sha256": rendered_sha,
        "input_proxy_tokens": row["input_proxy_tokens"],
    }


def start_provisional_attempt(
    campaign_id: str,
    assignment_id: str,
    claim_id: str,
    *,
    runtime_receipt: Mapping[str, Any],
    root: Path = ledger.LEDGER_ROOT,
    now: str | None = None,
) -> dict[str, Any]:
    """Persist and charge an attempt before model invocation."""
    campaign = load_provisional_campaign(campaign_id, root=root)
    run_dir = ledger.work_dir(campaign["run_id"], root)
    effective_ceiling = _effective_input_token_ceiling(run_dir)
    now = now or _utc_now()
    connection = _connect(run_dir)
    try:
        connection.execute("BEGIN IMMEDIATE")
        session_id, _ = _register_session(connection, runtime_receipt)
        row = connection.execute(
            "SELECT * FROM assignments WHERE assignment_id = ?",
            (assignment_id,),
        ).fetchone()
        if row is None:
            raise KeyError(assignment_id)
        if (
            row["status"] != "claimed"
            or row["active_claim_id"] != claim_id
            or row["lease_expires_at"] <= now
        ):
            raise IntegrityDriftError("attempt does not own an active claim")
        claim_session = connection.execute(
            """
            SELECT session_receipt_id FROM claim_events
            WHERE claim_id = ? AND event_type = 'claimed'
            ORDER BY event_seq LIMIT 1
            """,
            (claim_id,),
        ).fetchone()
        if claim_session is None or claim_session["session_receipt_id"] != session_id:
            raise IntegrityDriftError("claim/runtime session mismatch")
        count = int(
            connection.execute(
                "SELECT COUNT(*) AS n FROM attempts WHERE assignment_id = ?",
                (assignment_id,),
            ).fetchone()["n"]
        )
        maximum = _attempt_limit(connection, assignment_id)
        if count >= maximum:
            raise RetryExhaustedError(assignment_id)
        reserved = int(
            connection.execute(
                "SELECT COALESCE(SUM(input_proxy_tokens), 0) AS n FROM attempts"
            ).fetchone()["n"]
        )
        if reserved + int(row["input_proxy_tokens"]) > effective_ceiling:
            connection.execute(
                "UPDATE meta SET value_json = ? WHERE key = 'execution_status'",
                (ledger.canonical_json("token_ceiling_blocked"),),
            )
            connection.execute("COMMIT")
            raise TokenCeilingError(
                f"{reserved + int(row['input_proxy_tokens'])} > "
                f"{effective_ceiling}"
            )
        attempt_no = count + 1
        semantic = {
            "campaign_id": campaign_id,
            "run_id": campaign["run_id"],
            "assignment_id": assignment_id,
            "attempt_no": attempt_no,
            "session_receipt_id": session_id,
            "rendered_sha256": row["rendered_sha256"],
            "input_proxy_tokens": row["input_proxy_tokens"],
        }
        attempt_id = ledger.content_id("att", semantic)
        connection.execute(
            """
            INSERT INTO attempts(
                attempt_id, assignment_id, attempt_no, session_receipt_id,
                rendered_sha256, input_proxy_tokens, status, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'started', ?)
            """,
            (
                attempt_id,
                assignment_id,
                attempt_no,
                session_id,
                row["rendered_sha256"],
                row["input_proxy_tokens"],
                now,
            ),
        )
        payload = {**semantic, "attempt_id": attempt_id, "event_type": "started"}
        _insert_event(
            connection,
            "attempt_events",
            prefix="attev",
            identity=payload,
            values=(
                attempt_id,
                "started",
                now,
                ledger.canonical_json(payload),
            ),
        )
        connection.execute("COMMIT")
        return {
            "status": "attempt_started",
            "attempt_id": attempt_id,
            "attempt_no": attempt_no,
            "assignment_id": assignment_id,
            "rendered_sha256": row["rendered_sha256"],
            "input_proxy_tokens": row["input_proxy_tokens"],
            "reserved_input_proxy_tokens": reserved
            + int(row["input_proxy_tokens"]),
        }
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def _field_rows(
    assignment: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    attempt_id: str,
    session_receipt_id: str,
    root: Path,
) -> list[dict[str, Any]]:
    annotations = workflow._response_pointer_value(response, "/annotations")
    by_local = {
        int(target["local_id"]): target for target in assignment["targets"]
    }
    bundle = _load_json(
        ledger.work_dir(str(assignment["run_id"]), root) / "bundle.json"
    )
    emitted = {
        row["label_type"]: row["response_pointer"]
        for row in bundle["emitted_label_specs"]
    }
    rows = []
    for annotation in annotations:
        target = by_local[int(annotation["local_id"])]
        for label_type in sorted(emitted):
            value = workflow._response_pointer_value(
                annotation, emitted[label_type]
            )
            value_json = ledger.canonical_json(value)
            rows.append(
                {
                    "canonical_subject_id": target["subject_id"],
                    "canonical_subject_key_json": ledger.canonical_json(
                        target["subject_key"]
                    ),
                    "label_type": label_type,
                    "value_json": value_json,
                    "value_sha256": ledger.sha256_text(value_json),
                    "provenance_kind": "fresh_single_pass",
                    "assignment_id": assignment["assignment_id"],
                    "attempt_id": attempt_id,
                    "session_receipt_id": session_receipt_id,
                }
            )
    return rows


def _classify_response_error(exc: Exception) -> str:
    message = str(exc).lower()
    if "local_id" in message or "incomplete" in message or "duplicate" in message:
        return "incomplete_duplicate_or_unknown_local_id"
    if "constituenc" in message or "evidence" in message:
        return "registered_special_validator_failure"
    return "response_schema_failure"


def complete_provisional_attempt(
    campaign_id: str,
    attempt_id: str,
    *,
    raw_output: str | Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
    execution_receipt: Mapping[str, Any],
    invocation_error_code: str | None = None,
    root: Path = ledger.LEDGER_ROOT,
    now: str | None = None,
) -> dict[str, Any]:
    """Persist every terminal attempt and accept at most one valid response."""
    campaign = load_provisional_campaign(campaign_id, root=root)
    run_dir = ledger.work_dir(campaign["run_id"], root)
    now = now or _utc_now()
    validated_runtime = _validate_worker_runtime_receipt(runtime_receipt)
    validated_execution = _validate_execution_receipt(
        execution_receipt,
        runtime_receipt=validated_runtime,
        invocation_error_code=invocation_error_code,
    )
    connection = _connect(run_dir)
    try:
        connection.execute("BEGIN IMMEDIATE")
        session_id, _ = _register_session(connection, validated_runtime)
        attempt = connection.execute(
            "SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()
        if attempt is None:
            raise KeyError(attempt_id)
        if attempt["session_receipt_id"] != session_id:
            raise IntegrityDriftError("attempt/runtime session mismatch")
        if attempt["status"] != "started":
            assignment_status = connection.execute(
                "SELECT status FROM assignments WHERE assignment_id = ?",
                (attempt["assignment_id"],),
            ).fetchone()
            if assignment_status is None:
                raise IntegrityDriftError("attempt assignment is absent")
            connection.execute("COMMIT")
            return {
                "status": attempt["status"],
                "attempt_id": attempt_id,
                "assignment_id": attempt["assignment_id"],
                "assignment_status": assignment_status["status"],
            }
        assignment_id = attempt["assignment_id"]
        assignment = _load_json(
            run_dir / "assignments" / f"{assignment_id}.json"
        )
        if isinstance(raw_output, str):
            raw_text = raw_output
            try:
                response: Mapping[str, Any] = json.loads(raw_output)
            except (json.JSONDecodeError, TypeError) as exc:
                response = {}
                validation_code = "json_parse_failure"
                validation_error: str | None = str(exc)
            else:
                validation_code = ""
                validation_error = None
        else:
            response = dict(raw_output)
            raw_text = ledger.canonical_json(response)
            validation_code = ""
            validation_error = None
        if invocation_error_code is not None:
            if invocation_error_code != "invocation_transient_failure":
                raise IntegrityDriftError("non-declared invocation error code")
            validation_code = invocation_error_code
            validation_error = str(validated_execution.get("error", ""))

        prepared: dict[str, Any] | None = None
        if not validation_code:
            try:
                prepared = workflow.prepare_response_artifacts(
                    campaign["run_id"],
                    assignment_id,
                    response,
                    root=root,
                    registry_path=ledger.REGISTRY_V2_PATH,
                    labeled_at=now,
                )
            except workflow.ResponseValidationError as exc:
                validation_code = _classify_response_error(exc)
                validation_error = str(exc)

        if validation_code:
            status = "invalid"
            terminal_status = "schema_invalid"
            count = int(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM attempts WHERE assignment_id = ?",
                    (assignment_id,),
                ).fetchone()["n"]
            )
            maximum = _attempt_limit(connection, assignment_id)
            assignment_status = (
                "terminal_failed" if count >= maximum else "available"
            )
            connection.execute(
                """
                UPDATE attempts SET status = ?, completed_at = ?,
                    submission_sha256 = ?, raw_output_json = ?,
                    validation_code = ?, validation_error = ?,
                    execution_receipt_json = ?
                WHERE attempt_id = ? AND status = 'started'
                """,
                (
                    terminal_status,
                    now,
                    ledger.sha256_text(raw_text),
                    ledger.canonical_json(raw_text),
                    validation_code,
                    validation_error,
                    ledger.canonical_json(validated_execution),
                    attempt_id,
                ),
            )
            connection.execute(
                """
                UPDATE assignments SET status = ?, active_claim_id = NULL,
                    lease_expires_at = NULL
                WHERE assignment_id = ?
                """,
                (assignment_status, assignment_id),
            )
            if assignment_status == "terminal_failed":
                connection.execute(
                    "UPDATE meta SET value_json = ? WHERE key = 'execution_status'",
                    (ledger.canonical_json("retry_exhausted_blocked"),),
                )
        else:
            assert prepared is not None
            package = {
                "package_version": "annotation-provisional-accepted-response-v1",
                "campaign_id": campaign_id,
                "run_id": campaign["run_id"],
                "assignment_id": assignment_id,
                "attempt_id": attempt_id,
                "session_receipt_id": session_id,
                "execution_receipt": validated_execution,
                "saved_response": prepared["saved_response"],
                "events": prepared["events"],
                "projections": prepared["projections"],
                "field_values": _field_rows(
                    assignment,
                    response,
                    attempt_id=attempt_id,
                    session_receipt_id=session_id,
                    root=root,
                ),
            }
            package = _artifact(
                package,
                prefix="acc",
                id_field="acceptance_id",
                sha_field="acceptance_sha256",
            )
            package_path = (
                run_dir / "accepted-responses" / f"{assignment_id}.json"
            )
            _publish_exact_json(package_path, package, compressed=True)
            updated = connection.execute(
                """
                UPDATE assignments
                SET status = 'accepted', accepted_attempt_id = ?,
                    active_claim_id = NULL, lease_expires_at = NULL
                WHERE assignment_id = ? AND status = 'claimed'
                  AND accepted_attempt_id IS NULL
                """,
                (attempt_id, assignment_id),
            )
            if updated.rowcount != 1:
                raise IntegrityDriftError(
                    "accepted response lost unique assignment race"
                )
            connection.execute(
                """
                UPDATE attempts SET status = 'accepted', completed_at = ?,
                    submission_sha256 = ?, raw_output_json = ?,
                    validation_code = 'accepted',
                    execution_receipt_json = ?, accepted_package_path = ?
                WHERE attempt_id = ? AND status = 'started'
                """,
                (
                    now,
                    ledger.sha256_text(raw_text),
                    ledger.canonical_json(raw_text),
                    ledger.canonical_json(validated_execution),
                    str(package_path.relative_to(run_dir)),
                    attempt_id,
                ),
            )
            status = "accepted"
            assignment_status = "accepted"
            validation_code = "accepted"
            validation_error = None
        event_payload = {
            "attempt_id": attempt_id,
            "assignment_id": assignment_id,
            "event_type": status,
            "occurred_at": now,
            "submission_sha256": ledger.sha256_text(raw_text),
            "validation_code": validation_code,
        }
        _insert_event(
            connection,
            "attempt_events",
            prefix="attev",
            identity=event_payload,
            values=(
                attempt_id,
                status,
                now,
                ledger.canonical_json(event_payload),
            ),
        )
        connection.execute("COMMIT")
        receipt = {
            "attempt_id": attempt_id,
            "assignment_id": assignment_id,
            "assignment_status": assignment_status,
            "status": status,
            "validation_code": validation_code,
            "validation_error": validation_error,
            "completed_at": now,
            "runtime_receipt_id": session_id,
            "execution_receipt": validated_execution,
            "submission_sha256": ledger.sha256_text(raw_text),
        }
        _publish_exact_json(
            run_dir / "attempt-receipts" / f"{attempt_id}.json", receipt
        )
        _publish_exact_json(
            run_dir
            / "execution-session-receipts"
            / f"{session_id}.json",
            validated_runtime,
        )
        return receipt
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def provisional_assignment_status(
    campaign_id: str,
    assignment_id: str,
    *,
    root: Path = ledger.LEDGER_ROOT,
) -> str:
    """Return one assignment's authoritative control-plane status."""
    campaign = load_provisional_campaign(campaign_id, root=root)
    run_dir = ledger.work_dir(campaign["run_id"], root)
    connection = _connect(run_dir)
    try:
        row = connection.execute(
            "SELECT status FROM assignments WHERE assignment_id = ?",
            (assignment_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise KeyError(assignment_id)
    return str(row["status"])


def provisional_execution_status(
    campaign_id: str, *, root: Path = ledger.LEDGER_ROOT
) -> dict[str, Any]:
    campaign = load_provisional_campaign(campaign_id, root=root)
    if campaign["run_id"] in ledger.sealed_runs(root):
        seal = ledger.verify_sealed_run(campaign["run_id"], root=root)
        return {
            "status": "sealed",
            "campaign_id": campaign_id,
            "run_id": campaign["run_id"],
            "artifact_set_sha256": seal["artifact_set_sha256"],
        }
    run_dir = ledger.work_dir(campaign["run_id"], root)
    connection = _connect(run_dir)
    try:
        assignment_counts = {
            row["status"]: int(row["n"])
            for row in connection.execute(
                "SELECT status, COUNT(*) AS n FROM assignments GROUP BY status"
            )
        }
        attempt_counts = {
            row["status"]: int(row["n"])
            for row in connection.execute(
                "SELECT status, COUNT(*) AS n FROM attempts GROUP BY status"
            )
        }
        reserved = int(
            connection.execute(
                "SELECT COALESCE(SUM(input_proxy_tokens), 0) AS n FROM attempts"
            ).fetchone()["n"]
        )
        sessions = int(
            connection.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()[
                "n"
            ]
        )
        execution_status = json.loads(
            connection.execute(
                "SELECT value_json FROM meta WHERE key = 'execution_status'"
            ).fetchone()["value_json"]
        )
    finally:
        connection.close()
    return {
        "status": execution_status,
        "campaign_id": campaign_id,
        "run_id": campaign["run_id"],
        "assignments": assignment_counts,
        "attempts": attempt_counts,
        "runtime_sessions": sessions,
        "reserved_input_proxy_tokens": reserved,
        "input_token_ceiling": _effective_input_token_ceiling(run_dir),
    }


def _export_control_receipts(
    connection: sqlite3.Connection, run_dir: Path
) -> None:
    tables = {
        "provisional-assignments.jsonl": (
            "SELECT * FROM assignments ORDER BY assignment_index"
        ),
        "claim-events.jsonl": "SELECT * FROM claim_events ORDER BY event_seq",
        "attempts.jsonl": (
            "SELECT * FROM attempts ORDER BY assignment_id, attempt_no"
        ),
        "attempt-events.jsonl": (
            "SELECT * FROM attempt_events ORDER BY event_seq"
        ),
        "runtime-sessions.jsonl": (
            "SELECT * FROM sessions ORDER BY session_receipt_id"
        ),
    }
    for filename, query in tables.items():
        ledger.write_gzip_jsonl(
            run_dir / filename,
            (dict(row) for row in connection.execute(query)),
        )


def compact_provisional_work_storage(
    campaign_id: str, *, root: Path = ledger.LEDGER_ROOT
) -> dict[str, Any]:
    """Deterministically gzip unsealed response JSON artifacts in place."""
    campaign = load_provisional_campaign(campaign_id, root=root)
    run_id = campaign["run_id"]
    if run_id in ledger.sealed_runs(root):
        raise IntegrityDriftError("sealed runs cannot be compacted in place")
    run_dir = ledger.work_dir(run_id, root)
    converted = 0
    bytes_before = 0
    bytes_after = 0
    for directory_name in ("assignments", "accepted-responses"):
        directory = run_dir / directory_name
        for path in sorted(directory.glob("*.json")):
            original_size = path.stat().st_size
            bytes_before += original_size
            with path.open("rb") as probe:
                already_compressed = probe.read(2) == b"\x1f\x8b"
            if not already_compressed:
                value = _load_json(path)
                temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
                try:
                    with ledger.open_deterministic_gzip_text(
                        temporary
                    ) as handle:
                        json.dump(
                            value,
                            handle,
                            indent=2,
                            sort_keys=True,
                            ensure_ascii=False,
                        )
                        handle.write("\n")
                    os.replace(temporary, path)
                    ledger.fsync_directory(path.parent)
                finally:
                    if temporary.exists():
                        temporary.unlink()
                converted += 1
            bytes_after += path.stat().st_size
    return {
        "status": "compacted",
        "campaign_id": campaign_id,
        "converted_files": converted,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
    }


def _verify_locked_execution_partition(
    *,
    campaign: Mapping[str, Any],
    run_dir: Path,
    connection: sqlite3.Connection,
    root: Path,
) -> dict[str, Any]:
    """Reconcile every persisted assignment with the locked fresh selection."""
    if campaign.get("scope") != "provisional-corpus":
        # Small unit-test campaigns intentionally exercise the control plane
        # without reproducing the repository's 35,154-subject plan.
        return {"status": "not_applicable_test_fixture"}
    expected_campaign = {
        "plan_id": PLAN_ID,
        "plan_sha256": "sha256:" + PLAN_ID.removeprefix("pcplan_"),
        "reuse_manifest_id": REUSE_ID,
        "reuse_manifest_sha256": "sha256:" + REUSE_ID.removeprefix("reuse_"),
        "fresh_selection_id": FRESH_ID,
        "fresh_selection_sha256": "sha256:"
        + FRESH_ID.removeprefix("fresh_"),
        "run_id": _run_id(),
        "n_assignments": EXPECTED_ASSIGNMENTS,
        "n_subjects": EXPECTED_FRESH_SUBJECTS,
        "input_token_estimate": EXPECTED_INPUT_TOKENS,
        "input_token_ceiling": INPUT_TOKEN_CEILING,
        "production_eligible": False,
    }
    if any(campaign.get(key) != value for key, value in expected_campaign.items()):
        raise IntegrityDriftError("campaign/locked execution identity drift")
    profile = campaign.get("required_execution_profile", {})
    if (
        profile.get("model_id") != REQUIRED_MODEL_ID
        or profile.get("reasoning_effort") != REQUIRED_REASONING_EFFORT
        or profile.get("fresh_single_pass") is not True
    ):
        raise IntegrityDriftError("campaign runtime profile drift")

    plan = provisional._validate_artifact(
        _load_json(
            Path(root)
            / provisional.PROVISIONAL_PLANS_DIR
            / f"{PLAN_ID}.json"
        ),
        prefix="pcplan",
        id_field="plan_id",
        sha_field="plan_sha256",
    )
    reuse = provisional._validate_artifact(
        _load_json(
            Path(root)
            / provisional.REUSE_MANIFESTS_DIR
            / f"{REUSE_ID}.json"
        ),
        prefix="reuse",
        id_field="reuse_manifest_id",
        sha_field="reuse_manifest_sha256",
    )
    fresh = provisional._validate_artifact(
        _load_json(
            Path(root)
            / provisional.FRESH_SELECTIONS_DIR
            / f"{FRESH_ID}.json"
        ),
        prefix="fresh",
        id_field="fresh_selection_id",
        sha_field="fresh_selection_sha256",
    )
    if (
        plan["plan_id"] != PLAN_ID
        or plan["reuse_manifest"]["reuse_manifest_id"] != REUSE_ID
        or plan["fresh_selection"]["fresh_selection_id"] != FRESH_ID
        or fresh["reuse_manifest_id"] != REUSE_ID
        or fresh["reuse_manifest_sha256"] != reuse["reuse_manifest_sha256"]
        or fresh["n_assignments"] != EXPECTED_ASSIGNMENTS
        or fresh["n_subjects"] != EXPECTED_FRESH_SUBJECTS
        or len(fresh["batches"]) != EXPECTED_ASSIGNMENTS
        or len(fresh["identity_rows"]) != EXPECTED_FRESH_SUBJECTS
    ):
        raise IntegrityDriftError("locked plan/reuse/fresh relationship drift")

    manifest = provisional._validate_artifact(
        _load_json(run_dir / EXECUTION_MANIFEST_FILENAME),
        prefix="pexec",
        id_field="execution_manifest_id",
        sha_field="execution_manifest_sha256",
    )
    expected_manifest = {
        "campaign_id": campaign["campaign_id"],
        "run_id": campaign["run_id"],
        "plan_id": PLAN_ID,
        "plan_sha256": plan["plan_sha256"],
        "fresh_selection_id": FRESH_ID,
        "fresh_selection_sha256": fresh["fresh_selection_sha256"],
        "reuse_manifest_id": REUSE_ID,
        "reuse_manifest_sha256": reuse["reuse_manifest_sha256"],
        "authorization_resolution_id": AUTHORIZATION_RESOLUTION_ID,
        "authorization_resolution_sha256": campaign[
            "authorization_resolution_sha256"
        ],
        "required_model_id": REQUIRED_MODEL_ID,
        "required_reasoning_effort": REQUIRED_REASONING_EFFORT,
        "retry_policy_id": campaign["retry_policy_id"],
        "retry_policy_sha256": campaign["retry_policy_sha256"],
        "input_token_estimate": EXPECTED_INPUT_TOKENS,
        "input_token_ceiling": INPUT_TOKEN_CEILING,
        "n_assignments": EXPECTED_ASSIGNMENTS,
        "n_subjects": EXPECTED_FRESH_SUBJECTS,
        "production_eligible": False,
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise IntegrityDriftError("execution manifest drift")
    policy = _load_json(run_dir / RETRY_POLICY_FILENAME)
    if policy != _retry_policy():
        raise IntegrityDriftError("frozen retry policy drift")

    index_rows = ledger.read_jsonl(run_dir / "assignment-index.jsonl")
    control_rows = [
        dict(row)
        for row in connection.execute(
            """
            SELECT assignment_index, assignment_id, planned_batch_id,
                   rendered_sha256, input_proxy_tokens
            FROM assignments ORDER BY assignment_index
            """
        )
    ]
    if index_rows != control_rows or len(index_rows) != EXPECTED_ASSIGNMENTS:
        raise IntegrityDriftError("assignment index/control reconciliation drift")
    assignment_paths = sorted((run_dir / "assignments").glob("*.json"))
    if len(assignment_paths) != EXPECTED_ASSIGNMENTS:
        raise IntegrityDriftError("assignment artifact count drift")

    observed_identities: set[tuple[str, str, str, str]] = set()
    proxy_total = 0
    for assignment_index, (row, batch) in enumerate(
        zip(index_rows, fresh["batches"], strict=True)
    ):
        if (
            row["assignment_index"] != assignment_index
            or row["planned_batch_id"] != batch["planned_batch_id"]
        ):
            raise IntegrityDriftError("assignment/fresh batch ordering drift")
        assignment = _load_json(
            run_dir / "assignments" / f"{row['assignment_id']}.json"
        )
        assignment_body = {
            key: value
            for key, value in assignment.items()
            if key
            not in {
                "assigned_at",
                "assignment_id",
                "assignment_input_sha256",
            }
        }
        input_sha = ledger.sha256_text(ledger.canonical_json(assignment_body))
        if (
            assignment.get("run_id") != campaign["run_id"]
            or assignment.get("assignment_id") != row["assignment_id"]
            or assignment.get("assignment_input_sha256") != input_sha
            or assignment.get("assignment_id")
            != ledger.content_id("asg", assignment_body)
        ):
            raise IntegrityDriftError("assignment content identity drift")
        targets = list(assignment["targets"])
        if (
            [target["planned_batch_id"] for target in targets]
            != [batch["planned_batch_id"]] * len(targets)
            or [target["planned_batch_order"] for target in targets]
            != list(range(1, len(targets) + 1))
            or [target["subject_key"] for target in targets]
            != batch["target_keys"]
            or [target["source_text_sha256"] for target in targets]
            != batch["target_source_text_sha256"]
            or [target["bundle_input_sha256"] for target in targets]
            != batch["target_bundle_input_sha256"]
        ):
            raise IntegrityDriftError("assignment targets differ from locked batch")
        rendered = workflow.render_assignment(assignment)
        rendered_sha = ledger.sha256_text(
            json.dumps(
                rendered,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
        )
        proxy = _assignment_proxy_tokens(rendered)
        if (
            row["rendered_sha256"] != rendered_sha
            or row["input_proxy_tokens"] != proxy
        ):
            raise IntegrityDriftError("rendered assignment/token receipt drift")
        proxy_total += proxy
        for target in targets:
            observed_identities.add(
                (
                    target["canonical_subject_id"],
                    ledger.canonical_json(target["subject_key"]),
                    target["source_text_sha256"],
                    target["bundle_input_sha256"],
                )
            )
    expected_identities = {
        (
            row["canonical_subject_id"],
            ledger.canonical_json(row["subject_key"]),
            row["source_text_sha256"],
            row["bundle_input_sha256"],
        )
        for row in fresh["identity_rows"]
    }
    if (
        observed_identities != expected_identities
        or len(observed_identities) != EXPECTED_FRESH_SUBJECTS
        or proxy_total != EXPECTED_INPUT_TOKENS
    ):
        raise IntegrityDriftError("fresh subject coverage/token total drift")
    return {
        "status": "verified",
        "assignments": len(index_rows),
        "subjects": len(observed_identities),
        "input_proxy_tokens": proxy_total,
    }


def recover_orphaned_attempts(
    campaign_id: str,
    *,
    reason: str,
    root: Path = ledger.LEDGER_ROOT,
) -> dict[str, Any]:
    """Close started attempts after a verified local coordinator failure."""
    if not str(reason).strip():
        raise ValueError("orphan recovery reason is required")
    campaign = load_provisional_campaign(campaign_id, root=root)
    run_dir = ledger.work_dir(campaign["run_id"], root)
    connection = _connect(run_dir)
    try:
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT attempts.attempt_id, attempts.session_receipt_id,
                       sessions.receipt_json
                FROM attempts
                JOIN sessions USING(session_receipt_id)
                WHERE attempts.status = 'started'
                ORDER BY attempts.started_at, attempts.attempt_id
                """
            )
        ]
    finally:
        connection.close()
    recovered = []
    for row in rows:
        runtime = json.loads(row["receipt_json"])
        execution_receipt = {
            "session_id": runtime["session_id"],
            "model_id": REQUIRED_MODEL_ID,
            "reasoning_effort": REQUIRED_REASONING_EFFORT,
            "source": "arcv1-local-coordinator-recovery",
            "status": "failed",
            "error": reason,
        }
        receipt = complete_provisional_attempt(
            campaign_id,
            row["attempt_id"],
            raw_output="",
            runtime_receipt=runtime,
            execution_receipt=execution_receipt,
            invocation_error_code="invocation_transient_failure",
            root=root,
        )
        recovered.append(receipt["attempt_id"])
    status = provisional_execution_status(campaign_id, root=root)
    return {
        "status": "recovered",
        "campaign_id": campaign_id,
        "recovered_attempt_ids": recovered,
        "recovered_attempts": len(recovered),
        "campaign_status": status,
    }


def apply_integrity_recovery_amendment(
    campaign_id: str,
    *,
    root: Path = ledger.LEDGER_ROOT,
    review_queue_path: Path | None = None,
) -> dict[str, Any]:
    """Apply the one-assignment RES018 retry/docs-baseline exception."""
    root = Path(root)
    review_queue_path = review_queue_path or (
        ledger.LEDGER_ROOT.parent.parent
        / "notes/annotation-refresh-campaign-review-queue-v1.json"
    )
    queue = _load_json(review_queue_path)
    resolution = next(
        (
            row
            for row in queue.get("resolutions", [])
            if row.get("resolution_id") == RECOVERY_AMENDMENT_RESOLUTION_ID
        ),
        None,
    )
    if (
        resolution is None
        or resolution.get("applies_to_items") != ["ARCV1-G007"]
        or resolution.get("authority_quote") != "space resolved, please continue"
        or resolution.get("decided_by") != "repository_user"
        or resolution.get("decision")
        != "approved_narrow_infrastructure_recovery_amendment"
    ):
        raise IntegrityDriftError("ARCV1-RES018 owner amendment is absent/drifted")

    campaign = load_provisional_campaign(campaign_id, root=root)
    run_dir = ledger.work_dir(campaign["run_id"], root)
    connection = _connect(run_dir)
    try:
        assignment = connection.execute(
            "SELECT * FROM assignments WHERE assignment_index = 12"
        ).fetchone()
        if assignment is None:
            raise IntegrityDriftError("amendment assignment index is absent")
        attempts = list(
            connection.execute(
                """
                SELECT attempt_no, status, validation_code, validation_error
                FROM attempts WHERE assignment_id = ? ORDER BY attempt_no
                """,
                (assignment["assignment_id"],),
            )
        )
    finally:
        connection.close()
    if (
        len(attempts) != 3
        or [row["attempt_no"] for row in attempts] != [1, 2, 3]
        or attempts[0]["validation_code"] != "invocation_transient_failure"
        or "ENOSPC" not in str(attempts[0]["validation_error"])
        or any(
            row["validation_code"] != "registered_special_validator_failure"
            for row in attempts[1:]
        )
        or assignment["status"] not in {"terminal_failed", "available", "accepted"}
    ):
        raise IntegrityDriftError("RES018 retry-exception precondition drift")

    original_baseline = _load_json(run_dir / BASELINE_FILENAME)
    current_state = _protected_state(root)
    if any(
        current_state.get(key) != value
        for key, value in original_baseline.items()
        if key != "docs_tree"
    ):
        raise IntegrityDriftError(
            "a non-docs protected component drifted before RES018"
        )
    semantic = {
        "amendment_version": "arcv1-provisional-integrity-amendment-v1",
        "campaign_id": campaign_id,
        "run_id": campaign["run_id"],
        "plan_id": PLAN_ID,
        "resolution_id": RECOVERY_AMENDMENT_RESOLUTION_ID,
        "resolution_sha256": _canonical_sha(resolution),
        "retry_exception": {
            "assignment_index": 12,
            "assignment_id": assignment["assignment_id"],
            "rendered_sha256": assignment["rendered_sha256"],
            "base_maximum_attempts": 3,
            "amended_maximum_attempts": 4,
            "additional_attempts": 1,
            "reason": "replace_ENOSPC_lost_delivery",
            "same_rendered_input_required": True,
            "repair_prompt": "prohibited",
            "payload_edit": "prohibited",
            "repartition": "prohibited",
        },
        "protected_state_overlay": {
            "field": "docs_tree",
            "previous_value": original_baseline["docs_tree"],
            "replacement_value": current_state["docs_tree"],
            "all_other_components_unchanged": True,
            "concurrent_output_owner": "repository_user",
        },
        "production_eligible": False,
        "prohibited_effects_unchanged": campaign["prohibited_effects"],
    }
    amendment = _artifact(
        semantic,
        prefix="pexam",
        id_field="amendment_id",
        sha_field="amendment_sha256",
    )
    amendment_dir = run_dir / AMENDMENTS_DIRNAME
    amendment_dir.mkdir(parents=True, exist_ok=True)
    _publish_exact_json(
        amendment_dir / f"{amendment['amendment_id']}.json",
        amendment,
    )

    connection = _connect(run_dir)
    try:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT value_json FROM meta WHERE key = 'attempt_limit_exceptions'"
        ).fetchone()
        exceptions = json.loads(existing["value_json"]) if existing else {}
        expected_limit = 4
        prior = exceptions.get(assignment["assignment_id"])
        if prior not in {None, expected_limit}:
            raise IntegrityDriftError("existing retry exception conflicts")
        exceptions[assignment["assignment_id"]] = expected_limit
        if existing is None:
            connection.execute(
                "INSERT INTO meta(key, value_json) VALUES (?, ?)",
                (
                    "attempt_limit_exceptions",
                    ledger.canonical_json(exceptions),
                ),
            )
        else:
            connection.execute(
                "UPDATE meta SET value_json = ? WHERE key = ?",
                (
                    ledger.canonical_json(exceptions),
                    "attempt_limit_exceptions",
                ),
            )
        connection.execute(
            """
            UPDATE assignments SET status = 'available',
                active_claim_id = NULL, lease_expires_at = NULL
            WHERE assignment_id = ? AND status = 'terminal_failed'
            """,
            (assignment["assignment_id"],),
        )
        if assignment["status"] != "accepted":
            connection.execute(
                "UPDATE meta SET value_json = ? WHERE key = 'execution_status'",
                (ledger.canonical_json("resumed_under_owner_amendment"),),
            )
        connection.execute("COMMIT")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return {
        "status": "applied",
        "campaign_id": campaign_id,
        "amendment_id": amendment["amendment_id"],
        "assignment_index": 12,
        "assignment_id": assignment["assignment_id"],
        "amended_maximum_attempts": 4,
        "protected_docs_tree": current_state["docs_tree"],
    }


def apply_validator_repair_amendment(
    campaign_id: str,
    *,
    root: Path = ledger.LEDGER_ROOT,
    review_queue_path: Path | None = None,
) -> dict[str, Any]:
    """Apply the single RES020 validator-feedback repair exception."""
    root = Path(root)
    review_queue_path = review_queue_path or (
        ledger.LEDGER_ROOT.parent.parent
        / "notes/annotation-refresh-campaign-review-queue-v1.json"
    )
    queue = _load_json(review_queue_path)
    resolution = next(
        (
            row
            for row in queue.get("resolutions", [])
            if row.get("resolution_id") == VALIDATOR_REPAIR_RESOLUTION_ID
        ),
        None,
    )
    if (
        resolution is None
        or resolution.get("applies_to_items") != ["ARCV1-G007"]
        or resolution.get("authority_quote")
        != "I would like to try running this again and getting the best data we can"
        or resolution.get("decided_by") != "repository_user"
        or resolution.get("decision")
        != "approved_single_validator_feedback_repair_attempt"
    ):
        raise IntegrityDriftError("ARCV1-RES020 owner amendment is absent/drifted")

    campaign = load_provisional_campaign(campaign_id, root=root)
    run_dir = ledger.work_dir(campaign["run_id"], root)
    prior_amendments = _validated_execution_amendments(run_dir)
    if not any(
        row.get("resolution_id") == RECOVERY_AMENDMENT_RESOLUTION_ID
        and row.get("retry_exception", {}).get("amended_maximum_attempts") == 4
        for row in prior_amendments
    ):
        raise IntegrityDriftError("RES020 requires the sealed RES018 amendment")
    connection = _connect(run_dir)
    try:
        assignment = connection.execute(
            "SELECT * FROM assignments WHERE assignment_index = 12"
        ).fetchone()
        if assignment is None:
            raise IntegrityDriftError("repair assignment index is absent")
        attempts = list(
            connection.execute(
                """
                SELECT attempt_no, status, validation_code, validation_error
                FROM attempts WHERE assignment_id = ? ORDER BY attempt_no
                """,
                (assignment["assignment_id"],),
            )
        )
        current_limit = _attempt_limit(
            connection, assignment["assignment_id"]
        )
    finally:
        connection.close()
    if (
        assignment is None
        or assignment["status"] not in {"terminal_failed", "available", "accepted"}
        or len(attempts) != 4
        or [row["attempt_no"] for row in attempts] != [1, 2, 3, 4]
        or attempts[-1]["status"] != "schema_invalid"
        or attempts[-1]["validation_code"]
        != "registered_special_validator_failure"
        or current_limit != 4
    ):
        raise IntegrityDriftError("RES020 repair precondition drift")
    effective_baseline = _effective_protected_baseline(run_dir)
    current_state = _protected_state(root)
    if any(
        current_state.get(key) != value
        for key, value in effective_baseline.items()
        if key != "docs_tree"
    ):
        raise IntegrityDriftError(
            "a non-docs protected component drifted before RES020"
        )

    semantic = {
        "amendment_version": "arcv1-provisional-validator-repair-v1",
        "campaign_id": campaign_id,
        "run_id": campaign["run_id"],
        "plan_id": PLAN_ID,
        "resolution_id": VALIDATOR_REPAIR_RESOLUTION_ID,
        "resolution_sha256": _canonical_sha(resolution),
        "retry_exception": {
            "assignment_index": 12,
            "assignment_id": assignment["assignment_id"],
            "rendered_sha256": assignment["rendered_sha256"],
            "previous_maximum_attempts": 4,
            "amended_maximum_attempts": 5,
            "additional_attempts": 1,
            "same_rendered_assignment_required": True,
            "fresh_reannotation_required": True,
            "prior_response_inspection": "prohibited",
            "payload_edit": "prohibited",
            "repartition": "prohibited",
        },
        "validator_feedback": {
            "text": VALIDATOR_REPAIR_FEEDBACK,
            "sha256": ledger.sha256_text(VALIDATOR_REPAIR_FEEDBACK),
            "scope": [
                "group_text_exact_substring",
                "evidence_span_exact_substring",
                "group_text_contained_in_evidence_span",
                "explicit_vs_context_resolved_consistency",
            ],
            "all_other_guidance_unchanged": True,
        },
        "protected_state_overlay": {
            "field": "docs_tree",
            "previous_value": effective_baseline["docs_tree"],
            "replacement_value": current_state["docs_tree"],
            "all_other_components_unchanged": True,
            "concurrent_output_owner": "repository_user",
        },
        "production_eligible": False,
        "prohibited_effects_unchanged": campaign["prohibited_effects"],
    }
    amendment = _artifact(
        semantic,
        prefix="pexam",
        id_field="amendment_id",
        sha_field="amendment_sha256",
    )
    amendment_dir = run_dir / AMENDMENTS_DIRNAME
    amendment_dir.mkdir(parents=True, exist_ok=True)
    _publish_exact_json(
        amendment_dir / f"{amendment['amendment_id']}.json",
        amendment,
    )

    connection = _connect(run_dir)
    try:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT value_json FROM meta WHERE key = 'attempt_limit_exceptions'"
        ).fetchone()
        if existing is None:
            raise IntegrityDriftError("RES018 attempt exception is missing")
        exceptions = json.loads(existing["value_json"])
        if exceptions.get(assignment["assignment_id"]) not in {4, 5}:
            raise IntegrityDriftError("existing repair exception conflicts")
        exceptions[assignment["assignment_id"]] = 5
        connection.execute(
            "UPDATE meta SET value_json = ? WHERE key = ?",
            (
                ledger.canonical_json(exceptions),
                "attempt_limit_exceptions",
            ),
        )
        connection.execute(
            """
            UPDATE assignments SET status = 'available',
                active_claim_id = NULL, lease_expires_at = NULL
            WHERE assignment_id = ? AND status = 'terminal_failed'
            """,
            (assignment["assignment_id"],),
        )
        if assignment["status"] != "accepted":
            connection.execute(
                "UPDATE meta SET value_json = ? WHERE key = 'execution_status'",
                (ledger.canonical_json("resumed_under_validator_repair"),),
            )
        connection.execute("COMMIT")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return {
        "status": "applied",
        "campaign_id": campaign_id,
        "amendment_id": amendment["amendment_id"],
        "assignment_index": 12,
        "assignment_id": assignment["assignment_id"],
        "amended_maximum_attempts": 5,
        "validator_feedback_sha256": semantic["validator_feedback"]["sha256"],
    }


def apply_validator_retry_policy_amendment(
    campaign_id: str,
    *,
    root: Path = ledger.LEDGER_ROOT,
    review_queue_path: Path | None = None,
) -> dict[str, Any]:
    """Apply RES021's recorded conditional-feedback retry policy."""
    root = Path(root)
    review_queue_path = review_queue_path or (
        ledger.LEDGER_ROOT.parent.parent
        / "notes/annotation-refresh-campaign-review-queue-v1.json"
    )
    queue = _load_json(review_queue_path)
    resolution = next(
        (
            row
            for row in queue.get("resolutions", [])
            if row.get("resolution_id")
            == VALIDATOR_RETRY_POLICY_RESOLUTION_ID
        ),
        None,
    )
    authority_quote = (
        "yes please do that, I just want it to make a call, it's okay if we "
        "don't have a ton of confidence in it quite yet, we just want to "
        "record all of this so we can keep improving the scripts"
    )
    if (
        resolution is None
        or resolution.get("applies_to_items") != ["ARCV1-G007"]
        or resolution.get("authority_quote") != authority_quote
        or resolution.get("decided_by") != "repository_user"
        or resolution.get("decision")
        != "approved_recorded_constituency_validator_feedback_retries"
    ):
        raise IntegrityDriftError("ARCV1-RES021 owner amendment is absent/drifted")

    campaign = load_provisional_campaign(campaign_id, root=root)
    run_dir = ledger.work_dir(campaign["run_id"], root)
    prior_amendments = _validated_execution_amendments(run_dir)
    if not any(
        row.get("resolution_id") == VALIDATOR_REPAIR_RESOLUTION_ID
        for row in prior_amendments
    ):
        raise IntegrityDriftError("RES021 requires the sealed RES020 amendment")
    connection = _connect(run_dir)
    try:
        assignment = connection.execute(
            "SELECT * FROM assignments WHERE assignment_index = 41"
        ).fetchone()
        if assignment is None:
            raise IntegrityDriftError("retry-policy assignment index is absent")
        attempts = list(
            connection.execute(
                """
                SELECT attempt_no, status, validation_code, validation_error
                FROM attempts WHERE assignment_id = ? ORDER BY attempt_no
                """,
                (assignment["assignment_id"],),
            )
        )
        current_limit = _attempt_limit(
            connection, assignment["assignment_id"]
        )
        counts = {
            row["status"]: int(row["n"])
            for row in connection.execute(
                "SELECT status, COUNT(*) AS n FROM assignments GROUP BY status"
            )
        }
    finally:
        connection.close()
    if (
        assignment["status"] != "terminal_failed"
        or len(attempts) != 3
        or [row["attempt_no"] for row in attempts] != [1, 2, 3]
        or any(row["status"] != "schema_invalid" for row in attempts)
        or any(
            row["validation_code"]
            != "registered_special_validator_failure"
            or not str(row["validation_error"]).startswith(
                "constituency claim "
            )
            for row in attempts
        )
        or current_limit != 3
        or counts.get("accepted") != 48
        or counts.get("terminal_failed") != 1
    ):
        raise IntegrityDriftError("RES021 retry-policy precondition drift")

    effective_baseline = _effective_protected_baseline(run_dir)
    current_state = _protected_state(root)
    if any(
        current_state.get(key) != value
        for key, value in effective_baseline.items()
        if key != "docs_tree"
    ):
        raise IntegrityDriftError(
            "a non-docs protected component drifted before RES021"
        )
    semantic = {
        "amendment_version": "arcv1-provisional-validator-retry-policy-v1",
        "campaign_id": campaign_id,
        "run_id": campaign["run_id"],
        "plan_id": PLAN_ID,
        "resolution_id": VALIDATOR_RETRY_POLICY_RESOLUTION_ID,
        "resolution_sha256": _canonical_sha(resolution),
        "retry_exception": {
            "assignment_index": 41,
            "assignment_id": assignment["assignment_id"],
            "rendered_sha256": assignment["rendered_sha256"],
            "base_maximum_attempts": 3,
            "amended_maximum_attempts": 4,
            "additional_attempts": 1,
            "additional_attempt_requires_validator_feedback": True,
        },
        "conditional_validator_feedback": {
            "trigger": {
                "immediately_previous_status": "schema_invalid",
                "immediately_previous_validation_code":
                "registered_special_validator_failure",
                "validation_error_prefix": "constituency claim ",
            },
            "applies_to": (
                "remaining fresh retries, including assignment 41 attempt 4"
            ),
            "ordinary_first_attempts_unchanged": True,
            "base_attempt_ceiling_unchanged_except_assignment_41": True,
            "prior_response_inspection": "prohibited",
            "validator_receipt_inspection_only": True,
            "text": VALIDATOR_REPAIR_FEEDBACK,
            "sha256": ledger.sha256_text(VALIDATOR_REPAIR_FEEDBACK),
        },
        "pilot_evidence": {
            "accepted_assignments": counts["accepted"],
            "terminal_failed_assignments": counts["terminal_failed"],
            "assignment_41_failure_codes": [
                row["validation_code"] for row in attempts
            ],
            "assignment_41_failure_messages": [
                row["validation_error"] for row in attempts
            ],
        },
        "confidence_status": "provisional_not_promoted",
        "recorded_for_script_improvement": True,
        "protected_state_overlay": {
            "field": "docs_tree",
            "previous_value": effective_baseline["docs_tree"],
            "replacement_value": current_state["docs_tree"],
            "all_other_components_unchanged": True,
            "concurrent_output_owner": "repository_user",
        },
        "production_eligible": False,
        "prohibited_effects_unchanged": campaign["prohibited_effects"],
    }
    amendment = _artifact(
        semantic,
        prefix="pexam",
        id_field="amendment_id",
        sha_field="amendment_sha256",
    )
    amendment_dir = run_dir / AMENDMENTS_DIRNAME
    amendment_dir.mkdir(parents=True, exist_ok=True)
    _publish_exact_json(
        amendment_dir / f"{amendment['amendment_id']}.json",
        amendment,
    )

    connection = _connect(run_dir)
    try:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT value_json FROM meta WHERE key = 'attempt_limit_exceptions'"
        ).fetchone()
        if existing is None:
            raise IntegrityDriftError("prior attempt exceptions are missing")
        exceptions = json.loads(existing["value_json"])
        prior = exceptions.get(assignment["assignment_id"])
        if prior not in {None, 4}:
            raise IntegrityDriftError("assignment 41 retry exception conflicts")
        exceptions[assignment["assignment_id"]] = 4
        connection.execute(
            "UPDATE meta SET value_json = ? WHERE key = ?",
            (
                ledger.canonical_json(exceptions),
                "attempt_limit_exceptions",
            ),
        )
        connection.execute(
            """
            UPDATE assignments SET status = 'available',
                active_claim_id = NULL, lease_expires_at = NULL
            WHERE assignment_id = ? AND status = 'terminal_failed'
            """,
            (assignment["assignment_id"],),
        )
        connection.execute(
            "UPDATE meta SET value_json = ? WHERE key = 'execution_status'",
            (
                ledger.canonical_json(
                    "resumed_under_validator_retry_policy"
                ),
            ),
        )
        connection.execute("COMMIT")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return {
        "status": "applied",
        "campaign_id": campaign_id,
        "amendment_id": amendment["amendment_id"],
        "assignment_index": 41,
        "assignment_id": assignment["assignment_id"],
        "amended_maximum_attempts": 4,
        "conditional_feedback_sha256": semantic[
            "conditional_validator_feedback"
        ]["sha256"],
        "confidence_status": semantic["confidence_status"],
    }


def apply_enhanced_validator_retry_amendment(
    campaign_id: str,
    *,
    root: Path = ledger.LEDGER_ROOT,
    review_queue_path: Path | None = None,
) -> dict[str, Any]:
    """Apply RES023's literal-transcript retry policy and 8912 exception."""
    root = Path(root)
    review_queue_path = review_queue_path or (
        ledger.LEDGER_ROOT.parent.parent
        / "notes/annotation-refresh-campaign-review-queue-v1.json"
    )
    queue = _load_json(review_queue_path)
    resolution = next(
        (
            row
            for row in queue.get("resolutions", [])
            if row.get("resolution_id")
            == ENHANCED_VALIDATOR_RETRY_RESOLUTION_ID
        ),
        None,
    )
    authority_quote = (
        "I explicitly authorize Codex to transmit ARCV1 assignment 8912 for "
        "one additional gpt-5.6-sol/high attempt, and remaining "
        "constituency-validator retry payloads, with an enhanced fixed "
        "instruction not to normalize, correct, or punctuate transcript "
        "wording. Store all responses and validation evidence locally as "
        "provisional data only. This does not authorize promotion, "
        "materialization, site generation, deployment, or modification of "
        "frozen paid artifacts."
    )
    if (
        resolution is None
        or resolution.get("applies_to_items") != ["ARCV1-G007"]
        or resolution.get("authority_quote") != authority_quote
        or resolution.get("decided_by") != "repository_user"
        or resolution.get("decision")
        != "approved_enhanced_literal_transcript_validator_retries"
    ):
        raise IntegrityDriftError("ARCV1-RES023 owner amendment is absent/drifted")

    campaign = load_provisional_campaign(campaign_id, root=root)
    run_dir = ledger.work_dir(campaign["run_id"], root)
    if not any(
        row.get("resolution_id") == VALIDATOR_RETRY_POLICY_RESOLUTION_ID
        for row in _validated_execution_amendments(run_dir)
    ):
        raise IntegrityDriftError("RES023 requires the sealed RES021 amendment")
    connection = _connect(run_dir)
    try:
        assignment = connection.execute(
            "SELECT * FROM assignments WHERE assignment_index = 8912"
        ).fetchone()
        if assignment is None:
            raise IntegrityDriftError("enhanced-retry assignment is absent")
        attempts = list(
            connection.execute(
                """
                SELECT attempt_no, status, validation_code, validation_error
                FROM attempts WHERE assignment_id = ? ORDER BY attempt_no
                """,
                (assignment["assignment_id"],),
            )
        )
        current_limit = _attempt_limit(
            connection, assignment["assignment_id"]
        )
        counts = {
            row["status"]: int(row["n"])
            for row in connection.execute(
                "SELECT status, COUNT(*) AS n FROM assignments GROUP BY status"
            )
        }
    finally:
        connection.close()
    expected_error = (
        "constituency claim 0: evidence_span is not an exact target substring"
    )
    if (
        assignment["status"] != "terminal_failed"
        or len(attempts) != 3
        or [row["attempt_no"] for row in attempts] != [1, 2, 3]
        or any(row["status"] != "schema_invalid" for row in attempts)
        or any(
            row["validation_code"]
            != "registered_special_validator_failure"
            or row["validation_error"] != expected_error
            for row in attempts
        )
        or current_limit != 3
        or counts
        != {
            "accepted": 708,
            "available": 8_477,
            "terminal_failed": 1,
        }
    ):
        raise IntegrityDriftError("RES023 enhanced-retry precondition drift")

    effective_baseline = _effective_protected_baseline(run_dir)
    current_state = _protected_state(root)
    if any(
        current_state.get(key) != value
        for key, value in effective_baseline.items()
        if key != "docs_tree"
    ):
        raise IntegrityDriftError(
            "a non-docs protected component drifted before RES023"
        )
    semantic = {
        "amendment_version": "arcv1-enhanced-literal-validator-retry-v1",
        "campaign_id": campaign_id,
        "run_id": campaign["run_id"],
        "plan_id": PLAN_ID,
        "resolution_id": ENHANCED_VALIDATOR_RETRY_RESOLUTION_ID,
        "resolution_sha256": _canonical_sha(resolution),
        "retry_exception": {
            "assignment_index": 8912,
            "assignment_id": assignment["assignment_id"],
            "rendered_sha256": assignment["rendered_sha256"],
            "base_maximum_attempts": 3,
            "amended_maximum_attempts": 4,
            "additional_attempts": 1,
            "additional_attempt_requires_enhanced_feedback": True,
        },
        "conditional_validator_feedback": {
            "supersedes_resolution_id":
            VALIDATOR_RETRY_POLICY_RESOLUTION_ID,
            "trigger": {
                "immediately_previous_status": "schema_invalid",
                "immediately_previous_validation_code":
                "registered_special_validator_failure",
                "validation_error_prefix": "constituency claim ",
            },
            "applies_to": (
                "remaining fresh retries, including assignment 8912 attempt 4"
            ),
            "ordinary_first_attempts_unchanged": True,
            "prior_response_inspection_by_model": "prohibited",
            "validator_receipt_inspection_only": True,
            "literal_transcript_copy_required": True,
            "text": ENHANCED_VALIDATOR_REPAIR_FEEDBACK,
            "sha256": ledger.sha256_text(
                ENHANCED_VALIDATOR_REPAIR_FEEDBACK
            ),
            "previous_feedback_sha256": ledger.sha256_text(
                VALIDATOR_REPAIR_FEEDBACK
            ),
        },
        "failure_diagnostic": {
            "assignment_index": 8912,
            "target_literal": "we protecting America at home",
            "rejected_normalization": "we're protecting America at home",
            "same_failure_on_all_three_attempts": True,
        },
        "confidence_status": "provisional_not_promoted",
        "recorded_for_script_improvement": True,
        "protected_state_overlay": {
            "field": "docs_tree",
            "previous_value": effective_baseline["docs_tree"],
            "replacement_value": current_state["docs_tree"],
            "all_other_components_unchanged": True,
            "concurrent_output_owner": "repository_user",
        },
        "production_eligible": False,
        "prohibited_effects_unchanged": campaign["prohibited_effects"],
    }
    amendment = _artifact(
        semantic,
        prefix="pexam",
        id_field="amendment_id",
        sha_field="amendment_sha256",
    )
    amendment_dir = run_dir / AMENDMENTS_DIRNAME
    amendment_dir.mkdir(parents=True, exist_ok=True)
    _publish_exact_json(
        amendment_dir / f"{amendment['amendment_id']}.json",
        amendment,
    )

    connection = _connect(run_dir)
    try:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT value_json FROM meta WHERE key = 'attempt_limit_exceptions'"
        ).fetchone()
        if existing is None:
            raise IntegrityDriftError("prior attempt exceptions are missing")
        exceptions = json.loads(existing["value_json"])
        prior = exceptions.get(assignment["assignment_id"])
        if prior not in {None, 4}:
            raise IntegrityDriftError("assignment 8912 retry exception conflicts")
        exceptions[assignment["assignment_id"]] = 4
        connection.execute(
            "UPDATE meta SET value_json = ? WHERE key = ?",
            (
                ledger.canonical_json(exceptions),
                "attempt_limit_exceptions",
            ),
        )
        connection.execute(
            """
            UPDATE assignments SET status = 'available',
                active_claim_id = NULL, lease_expires_at = NULL
            WHERE assignment_id = ? AND status = 'terminal_failed'
            """,
            (assignment["assignment_id"],),
        )
        connection.execute(
            "UPDATE meta SET value_json = ? WHERE key = 'execution_status'",
            (
                ledger.canonical_json(
                    "resumed_under_enhanced_validator_retry_policy"
                ),
            ),
        )
        connection.execute("COMMIT")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return {
        "status": "applied",
        "campaign_id": campaign_id,
        "amendment_id": amendment["amendment_id"],
        "assignment_index": 8912,
        "assignment_id": assignment["assignment_id"],
        "amended_maximum_attempts": 4,
        "enhanced_feedback_sha256": semantic[
            "conditional_validator_feedback"
        ]["sha256"],
        "confidence_status": semantic["confidence_status"],
    }


def grant_automatic_literal_retry_exceptions(
    campaign_id: str,
    *,
    root: Path = ledger.LEDGER_ROOT,
) -> dict[str, Any]:
    """Grant one targeted fourth attempt to every qualifying terminal failure."""
    root = Path(root)
    campaign = load_provisional_campaign(campaign_id, root=root)
    run_dir = ledger.work_dir(campaign["run_id"], root)
    amendments = _validated_execution_amendments(run_dir)
    policy = next(
        (
            row
            for row in amendments
            if row.get("resolution_id")
            == TARGETED_LITERAL_RETRY_RESOLUTION_ID
        ),
        None,
    )
    if policy is None:
        return {
            "status": "policy_not_active",
            "campaign_id": campaign_id,
            "granted": [],
            "unresolved_terminal_failures": [],
        }

    granted: list[dict[str, Any]] = []
    with _LITERAL_GRANT_THREAD_LOCK, ledger.ledger_lock(root):
        connection = _connect(run_dir)
        try:
            terminal_rows = list(
                connection.execute(
                    """
                    SELECT * FROM assignments
                    WHERE status = 'terminal_failed'
                    ORDER BY assignment_index
                    """
                )
            )
        finally:
            connection.close()
        existing_grants = {
            row["assignment_id"]: row
            for row in _validated_literal_retry_grants(run_dir)
        }
        for assignment in terminal_rows:
            assignment_id = str(assignment["assignment_id"])
            connection = _connect(run_dir)
            try:
                attempts = [
                    dict(row)
                    for row in connection.execute(
                        """
                        SELECT attempt_id, attempt_no, status, validation_code,
                               validation_error, submission_sha256
                        FROM attempts
                        WHERE assignment_id = ?
                        ORDER BY attempt_no
                        """,
                        (assignment_id,),
                    )
                ]
                current_limit = _attempt_limit(connection, assignment_id)
            finally:
                connection.close()
            qualifies = (
                current_limit == 3
                and len(attempts) == 3
                and [row["attempt_no"] for row in attempts] == [1, 2, 3]
                and all(row["status"] == "schema_invalid" for row in attempts)
                and all(
                    row["validation_code"]
                    == "registered_special_validator_failure"
                    and str(row["validation_error"]).startswith(
                        "constituency claim "
                    )
                    and "exact target substring"
                    in str(row["validation_error"])
                    for row in attempts
                )
            )
            if not qualifies:
                continue
            diagnostics = diagnose_constituency_literal_mismatches(
                campaign_id,
                int(assignment["assignment_index"]),
                root=root,
            )
            verified = [
                row
                for row in diagnostics
                if row.get("exact_candidate_verified") is True
            ]
            if {
                int(row["attempt_no"]) for row in verified
            } != {1, 2, 3}:
                continue
            feedback = targeted_literal_retry_feedback(verified)
            semantic = {
                "grant_version": "arcv1-targeted-literal-retry-grant-v1",
                "campaign_id": campaign_id,
                "run_id": campaign["run_id"],
                "plan_id": PLAN_ID,
                "resolution_id": TARGETED_LITERAL_RETRY_RESOLUTION_ID,
                "policy_amendment_id": policy["amendment_id"],
                "policy_amendment_sha256": policy["amendment_sha256"],
                "assignment_index": int(assignment["assignment_index"]),
                "assignment_id": assignment_id,
                "rendered_sha256": assignment["rendered_sha256"],
                "base_maximum_attempts": 3,
                "amended_maximum_attempts": 4,
                "additional_attempts": 1,
                "qualifying_attempts": attempts,
                "verified_literal_diagnostics": verified,
                "validator_feedback": {
                    "text": feedback,
                    "sha256": ledger.sha256_text(feedback),
                    "prior_responses_transmitted": False,
                    "literal_target_candidates_only": True,
                },
                "same_rendered_assignment_required": True,
                "fresh_reannotation_required": True,
                "input_token_ceiling_unchanged": INPUT_TOKEN_CEILING,
                "production_eligible": False,
            }
            grant = _artifact(
                semantic,
                prefix="plitgrant",
                id_field="grant_id",
                sha_field="grant_sha256",
            )
            existing = existing_grants.get(assignment_id)
            if existing is not None and existing != grant:
                raise IntegrityDriftError(
                    "existing literal retry grant conflicts"
                )
            grant_dir = run_dir / LITERAL_RETRY_GRANTS_DIRNAME
            grant_dir.mkdir(parents=True, exist_ok=True)
            _publish_exact_json(
                grant_dir / f"{grant['grant_id']}.json",
                grant,
            )
            connection = _connect(run_dir)
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT status FROM assignments WHERE assignment_id = ?
                    """,
                    (assignment_id,),
                ).fetchone()
                exception_row = connection.execute(
                    """
                    SELECT value_json FROM meta
                    WHERE key = 'attempt_limit_exceptions'
                    """
                ).fetchone()
                exceptions = (
                    json.loads(exception_row["value_json"])
                    if exception_row is not None
                    else {}
                )
                prior_limit = exceptions.get(assignment_id)
                if prior_limit not in {None, 4}:
                    raise IntegrityDriftError(
                        "automatic literal retry exception conflicts"
                    )
                exceptions[assignment_id] = 4
                if exception_row is None:
                    connection.execute(
                        "INSERT INTO meta(key, value_json) VALUES (?, ?)",
                        (
                            "attempt_limit_exceptions",
                            ledger.canonical_json(exceptions),
                        ),
                    )
                else:
                    connection.execute(
                        "UPDATE meta SET value_json = ? WHERE key = ?",
                        (
                            ledger.canonical_json(exceptions),
                            "attempt_limit_exceptions",
                        ),
                    )
                if row is None:
                    raise IntegrityDriftError(
                        "literal retry assignment disappeared"
                    )
                if row["status"] == "terminal_failed":
                    connection.execute(
                        """
                        UPDATE assignments
                        SET status = 'available', active_claim_id = NULL,
                            lease_expires_at = NULL
                        WHERE assignment_id = ?
                        """,
                        (assignment_id,),
                    )
                elif row["status"] not in {"available", "accepted"}:
                    raise IntegrityDriftError(
                        "literal retry assignment status drift"
                    )
                connection.execute(
                    """
                    UPDATE meta SET value_json = ?
                    WHERE key = 'execution_status'
                    """,
                    (
                        ledger.canonical_json(
                            "resumed_under_automatic_literal_retry_policy"
                        ),
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()
            existing_grants[assignment_id] = grant
            granted.append(
                {
                    "assignment_index": int(assignment["assignment_index"]),
                    "assignment_id": assignment_id,
                    "grant_id": grant["grant_id"],
                    "amended_maximum_attempts": 4,
                    "validator_feedback_sha256": grant[
                        "validator_feedback"
                    ]["sha256"],
                }
            )

    connection = _connect(run_dir)
    try:
        unresolved = [
            int(row["assignment_index"])
            for row in connection.execute(
                """
                SELECT assignment_index FROM assignments
                WHERE status = 'terminal_failed'
                ORDER BY assignment_index
                """
            )
        ]
    finally:
        connection.close()
    return {
        "status": "granted" if granted else "no_new_grants",
        "campaign_id": campaign_id,
        "granted": granted,
        "unresolved_terminal_failures": unresolved,
    }


def apply_targeted_literal_retry_policy_amendment(
    campaign_id: str,
    *,
    root: Path = ledger.LEDGER_ROOT,
    review_queue_path: Path | None = None,
) -> dict[str, Any]:
    """Apply RES025's bounded automatic targeted-literal retry policy."""
    root = Path(root)
    review_queue_path = review_queue_path or (
        ledger.LEDGER_ROOT.parent.parent
        / "notes/annotation-refresh-campaign-review-queue-v1.json"
    )
    queue = _load_json(review_queue_path)
    resolution = next(
        (
            row
            for row in queue.get("resolutions", [])
            if row.get("resolution_id")
            == TARGETED_LITERAL_RETRY_RESOLUTION_ID
        ),
        None,
    )
    authority_quote = (
        "I want to change the policy so it continues automatically in this "
        "scenario:\n\nI explicitly authorize Codex to transmit ARCV1 "
        "assignment 8177 for one additional gpt-5.6-sol/high attempt, and "
        "remaining constituency-validator retry payloads, with an "
        "assignment-specific literal hint and an instruction to shorten "
        "evidence spans until they exactly match the target text. Store all "
        "results locally as provisional evidence only. This does not expand "
        "the token ceiling or authorize promotion, materialization, site "
        "generation, deployment, or modification of frozen paid artifacts."
    )
    if (
        resolution is None
        or resolution.get("applies_to_items") != ["ARCV1-G007"]
        or resolution.get("authority_quote") != authority_quote
        or resolution.get("decided_by") != "repository_user"
        or resolution.get("decision")
        != "approved_automatic_targeted_literal_constituency_retries"
    ):
        raise IntegrityDriftError("ARCV1-RES025 owner amendment is absent/drifted")

    campaign = load_provisional_campaign(campaign_id, root=root)
    run_dir = ledger.work_dir(campaign["run_id"], root)
    prior_amendments = _validated_execution_amendments(run_dir)
    existing_policy = next(
        (
            row
            for row in prior_amendments
            if row.get("resolution_id")
            == TARGETED_LITERAL_RETRY_RESOLUTION_ID
        ),
        None,
    )
    if existing_policy is not None:
        grants = grant_automatic_literal_retry_exceptions(
            campaign_id, root=root
        )
        return {
            "status": "already_applied",
            "campaign_id": campaign_id,
            "amendment_id": existing_policy["amendment_id"],
            "automatic_grants": grants,
        }
    if not any(
        row.get("resolution_id") == ENHANCED_VALIDATOR_RETRY_RESOLUTION_ID
        for row in prior_amendments
    ):
        raise IntegrityDriftError("RES025 requires the sealed RES023 amendment")

    connection = _connect(run_dir)
    try:
        assignment = connection.execute(
            "SELECT * FROM assignments WHERE assignment_index = 8177"
        ).fetchone()
        if assignment is None:
            raise IntegrityDriftError("targeted-retry assignment is absent")
        attempts = [
            dict(row)
            for row in connection.execute(
                """
                SELECT attempt_no, status, validation_code, validation_error
                FROM attempts WHERE assignment_id = ? ORDER BY attempt_no
                """,
                (assignment["assignment_id"],),
            )
        ]
        current_limit = _attempt_limit(
            connection, assignment["assignment_id"]
        )
        counts = {
            row["status"]: int(row["n"])
            for row in connection.execute(
                "SELECT status, COUNT(*) AS n FROM assignments GROUP BY status"
            )
        }
    finally:
        connection.close()
    if (
        assignment["status"] != "terminal_failed"
        or current_limit != 3
        or counts
        != {
            "accepted": 3_584,
            "available": 5_601,
            "terminal_failed": 1,
        }
        or len(attempts) != 3
        or [row["attempt_no"] for row in attempts] != [1, 2, 3]
        or any(row["status"] != "schema_invalid" for row in attempts)
        or any(
            row["validation_code"]
            != "registered_special_validator_failure"
            or "evidence_span is not an exact target substring"
            not in str(row["validation_error"])
            for row in attempts
        )
    ):
        raise IntegrityDriftError("RES025 targeted-retry precondition drift")
    diagnostics = diagnose_constituency_literal_mismatches(
        campaign_id, 8177, root=root
    )
    if {
        int(row["attempt_no"])
        for row in diagnostics
        if row.get("exact_candidate_verified") is True
    } != {1, 2, 3}:
        raise IntegrityDriftError("RES025 literal diagnostic precondition drift")

    effective_baseline = _effective_protected_baseline(run_dir)
    current_state = _protected_state(root)
    if any(
        current_state.get(key) != value
        for key, value in effective_baseline.items()
        if key != "docs_tree"
    ):
        raise IntegrityDriftError(
            "a non-docs protected component drifted before RES025"
        )
    semantic = {
        "amendment_version": "arcv1-targeted-literal-retry-policy-v1",
        "campaign_id": campaign_id,
        "run_id": campaign["run_id"],
        "plan_id": PLAN_ID,
        "resolution_id": TARGETED_LITERAL_RETRY_RESOLUTION_ID,
        "resolution_sha256": _canonical_sha(resolution),
        "automatic_retry_policy": {
            "trigger": {
                "assignment_status": "terminal_failed",
                "base_attempts_exhausted": 3,
                "all_attempt_statuses": "schema_invalid",
                "all_validation_codes":
                "registered_special_validator_failure",
                "validation_error_contains":
                "exact target substring",
                "verified_literal_candidate_required": True,
            },
            "amended_maximum_attempts": 4,
            "additional_attempts": 1,
            "assignment_specific_literal_hint": True,
            "shorten_until_exact_target_membership": True,
            "ordinary_attempts_unchanged": True,
            "failed_fourth_attempt_remains_terminal": True,
            "prior_response_inspection_by_model": "prohibited",
            "automatic_continuation": True,
        },
        "authorized_initial_assignment_index": 8177,
        "input_token_ceiling": INPUT_TOKEN_CEILING,
        "input_token_ceiling_expanded": False,
        "confidence_status": "provisional_not_promoted",
        "recorded_for_script_improvement": True,
        "protected_state_overlay": {
            "field": "docs_tree",
            "previous_value": effective_baseline["docs_tree"],
            "replacement_value": current_state["docs_tree"],
            "all_other_components_unchanged": True,
            "concurrent_output_owner": "repository_user",
        },
        "production_eligible": False,
        "prohibited_effects_unchanged": campaign["prohibited_effects"],
    }
    amendment = _artifact(
        semantic,
        prefix="pexam",
        id_field="amendment_id",
        sha_field="amendment_sha256",
    )
    amendment_dir = run_dir / AMENDMENTS_DIRNAME
    amendment_dir.mkdir(parents=True, exist_ok=True)
    _publish_exact_json(
        amendment_dir / f"{amendment['amendment_id']}.json",
        amendment,
    )
    grants = grant_automatic_literal_retry_exceptions(
        campaign_id, root=root
    )
    if [row["assignment_index"] for row in grants["granted"]] != [8177]:
        raise IntegrityDriftError("RES025 did not grant assignment 8177")
    return {
        "status": "applied",
        "campaign_id": campaign_id,
        "amendment_id": amendment["amendment_id"],
        "automatic_grants": grants,
        "protected_docs_tree": current_state["docs_tree"],
    }


def apply_completion_amendment(
    campaign_id: str,
    *,
    root: Path = ledger.LEDGER_ROOT,
    review_queue_path: Path | None = None,
) -> dict[str, Any]:
    """Apply RES026's ceiling, terminal-retry, and docs-overlay authority."""
    root = Path(root)
    review_queue_path = review_queue_path or (
        ledger.LEDGER_ROOT.parent.parent
        / "notes/annotation-refresh-campaign-review-queue-v1.json"
    )
    queue = _load_json(review_queue_path)
    resolution = next(
        (
            row
            for row in queue.get("resolutions", [])
            if row.get("resolution_id")
            == COMPLETION_AMENDMENT_RESOLUTION_ID
        ),
        None,
    )
    if (
        resolution is None
        or resolution.get("applies_to_items") != ["ARCV1-G007"]
        or resolution.get("authority_quote") != COMPLETION_AUTHORITY_QUOTE
        or resolution.get("decided_by") != "repository_user"
        or resolution.get("decision")
        != "approved_provisional_corpus_completion_amendment"
    ):
        raise IntegrityDriftError("ARCV1-RES026 owner amendment is absent/drifted")

    campaign = load_provisional_campaign(campaign_id, root=root)
    run_dir = ledger.work_dir(campaign["run_id"], root)
    prior_amendments = _validated_execution_amendments(run_dir)
    existing = next(
        (
            row
            for row in prior_amendments
            if row.get("resolution_id")
            == COMPLETION_AMENDMENT_RESOLUTION_ID
        ),
        None,
    )
    if existing is not None:
        return {
            "status": "already_applied",
            "campaign_id": campaign_id,
            "amendment_id": existing["amendment_id"],
            "input_token_ceiling":
            _effective_input_token_ceiling(run_dir),
        }
    if not any(
        row.get("resolution_id") == TARGETED_LITERAL_RETRY_RESOLUTION_ID
        for row in prior_amendments
    ):
        raise IntegrityDriftError("RES026 requires the sealed RES025 amendment")

    now = _utc_now()
    connection = _connect(run_dir)
    try:
        connection.execute("BEGIN IMMEDIATE")
        _expire_claims(connection, now=now)
        counts = {
            row["status"]: int(row["n"])
            for row in connection.execute(
                "SELECT status, COUNT(*) AS n FROM assignments GROUP BY status"
            )
        }
        attempt_counts = {
            row["status"]: int(row["n"])
            for row in connection.execute(
                "SELECT status, COUNT(*) AS n FROM attempts GROUP BY status"
            )
        }
        reserved = int(
            connection.execute(
                "SELECT COALESCE(SUM(input_proxy_tokens), 0) AS n FROM attempts"
            ).fetchone()["n"]
        )
        terminal_rows = list(
            connection.execute(
                """
                SELECT * FROM assignments
                WHERE status = 'terminal_failed'
                ORDER BY assignment_index
                """
            )
        )
        if (
            counts
            != {
                "accepted": 9_070,
                "available": 109,
                "terminal_failed": 7,
            }
            or attempt_counts
            != {
                "accepted": 9_070,
                "schema_invalid": 1_027,
            }
            or reserved != 89_824_184
            or tuple(
                int(row["assignment_index"]) for row in terminal_rows
            )
            != COMPLETION_RETRY_ASSIGNMENT_INDICES
        ):
            raise IntegrityDriftError("RES026 execution checkpoint drift")

        retry_exceptions: list[dict[str, Any]] = []
        for assignment in terminal_rows:
            attempts = list(
                connection.execute(
                    """
                    SELECT attempt_no, status, validation_code,
                           validation_error
                    FROM attempts
                    WHERE assignment_id = ?
                    ORDER BY attempt_no
                    """,
                    (assignment["assignment_id"],),
                )
            )
            assignment_index = int(assignment["assignment_index"])
            if (
                len(attempts) != 3
                or [row["attempt_no"] for row in attempts] != [1, 2, 3]
                or any(
                    row["status"] != "schema_invalid" for row in attempts
                )
                or _attempt_limit(
                    connection, assignment["assignment_id"]
                ) != 3
            ):
                raise IntegrityDriftError(
                    "RES026 terminal attempt history drift"
                )
            retry_mode = (
                "enhanced_constituency_validator_feedback"
                if assignment_index in {844, 3907}
                else "unchanged_locked_payload_after_infrastructure_failure"
            )
            if retry_mode.startswith("enhanced"):
                if attempts[-1]["validation_code"] != (
                    "registered_special_validator_failure"
                ):
                    raise IntegrityDriftError(
                        "RES026 semantic terminal classification drift"
                    )
            elif attempts[-1]["validation_code"] != (
                "invocation_transient_failure"
            ):
                raise IntegrityDriftError(
                    "RES026 infrastructure terminal classification drift"
                )
            retry_exceptions.append(
                {
                    "assignment_index": assignment_index,
                    "assignment_id": assignment["assignment_id"],
                    "rendered_sha256": assignment["rendered_sha256"],
                    "base_maximum_attempts": 3,
                    "amended_maximum_attempts": 6,
                    "additional_attempts": 3,
                    "retry_mode": retry_mode,
                }
            )
        connection.execute("COMMIT")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()

    effective_baseline = _effective_protected_baseline(run_dir)
    current_state = _protected_state(root)
    if any(
        current_state.get(key) != value
        for key, value in effective_baseline.items()
        if key != "docs_tree"
    ):
        raise IntegrityDriftError(
            "a non-docs protected component drifted before RES026"
        )
    expected_docs_tree = {
        "files": 348,
        "sha256": (
            "sha256:"
            "a7215af43f9c35378a45aae7fbd9dd251a415140f2e805569bcb43a7938a8af4"
        ),
    }

    semantic = {
        "amendment_version": "arcv1-provisional-completion-amendment-v1",
        "campaign_id": campaign_id,
        "run_id": campaign["run_id"],
        "plan_id": PLAN_ID,
        "resolution_id": COMPLETION_AMENDMENT_RESOLUTION_ID,
        "resolution_sha256": _canonical_sha(resolution),
        "execution_checkpoint": {
            "accepted_assignments": 9_070,
            "available_assignments": 109,
            "terminal_failed_assignments": 7,
            "accepted_attempts": 9_070,
            "invalid_attempts": 1_027,
            "reserved_input_proxy_tokens": 89_824_184,
        },
        "input_token_ceiling_amendment": {
            "previous_value": INPUT_TOKEN_CEILING,
            "replacement_value": COMPLETION_INPUT_TOKEN_CEILING,
            "expanded_by": (
                COMPLETION_INPUT_TOKEN_CEILING - INPUT_TOKEN_CEILING
            ),
        },
        "retry_exceptions": retry_exceptions,
        "semantic_retry_feedback": {
            "assignment_indices": [844, 3907],
            "text": ENHANCED_VALIDATOR_REPAIR_FEEDBACK,
            "sha256": ledger.sha256_text(
                ENHANCED_VALIDATOR_REPAIR_FEEDBACK
            ),
            "no_normalization_or_local_repair": True,
        },
        "infrastructure_retry_payload": {
            "assignment_indices": [2265, 2266, 4282, 6496, 8895],
            "locked_payload_unchanged": True,
        },
        "confidence_status": "provisional_not_promoted",
        "protected_state_overlay": {
            "field": "docs_tree",
            "previous_value": effective_baseline["docs_tree"],
            "replacement_value": expected_docs_tree,
            "all_other_components_unchanged": True,
            "concurrent_output_owner": "repository_user",
            "observed_value_at_application": current_state["docs_tree"],
            "observed_value_matches_authority": (
                current_state["docs_tree"] == expected_docs_tree
            ),
        },
        "production_eligible": False,
        "prohibited_effects_unchanged": campaign["prohibited_effects"],
    }
    amendment = _artifact(
        semantic,
        prefix="pexam",
        id_field="amendment_id",
        sha_field="amendment_sha256",
    )
    amendment_dir = run_dir / AMENDMENTS_DIRNAME
    amendment_dir.mkdir(parents=True, exist_ok=True)
    _publish_exact_json(
        amendment_dir / f"{amendment['amendment_id']}.json",
        amendment,
    )

    connection = _connect(run_dir)
    try:
        connection.execute("BEGIN IMMEDIATE")
        exception_row = connection.execute(
            """
            SELECT value_json FROM meta
            WHERE key = 'attempt_limit_exceptions'
            """
        ).fetchone()
        exceptions = (
            json.loads(exception_row["value_json"])
            if exception_row is not None
            else {}
        )
        for retry in retry_exceptions:
            assignment_id = retry["assignment_id"]
            prior_limit = exceptions.get(assignment_id)
            if prior_limit not in {None, 6}:
                raise IntegrityDriftError(
                    "RES026 retry exception conflicts"
                )
            exceptions[assignment_id] = 6
        if exception_row is None:
            connection.execute(
                "INSERT INTO meta(key, value_json) VALUES (?, ?)",
                (
                    "attempt_limit_exceptions",
                    ledger.canonical_json(exceptions),
                ),
            )
        else:
            connection.execute(
                "UPDATE meta SET value_json = ? WHERE key = ?",
                (
                    ledger.canonical_json(exceptions),
                    "attempt_limit_exceptions",
                ),
            )
        connection.execute(
            """
            UPDATE assignments
            SET status = 'available', active_claim_id = NULL,
                lease_expires_at = NULL
            WHERE status = 'terminal_failed'
              AND assignment_index IN (844, 2265, 2266, 3907, 4282, 6496, 8895)
            """
        )
        connection.execute(
            "UPDATE meta SET value_json = ? WHERE key = 'execution_status'",
            (
                ledger.canonical_json(
                    "resumed_under_completion_amendment"
                ),
            ),
        )
        connection.execute("COMMIT")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()

    declared = _declared_attempt_limit_exceptions(run_dir)
    for retry in retry_exceptions:
        if declared.get(retry["assignment_id"]) != 6:
            raise IntegrityDriftError(
                "RES026 retry exception provenance drift"
            )
    return {
        "status": "applied",
        "campaign_id": campaign_id,
        "amendment_id": amendment["amendment_id"],
        "input_token_ceiling": _effective_input_token_ceiling(run_dir),
        "reopened_assignment_indices": list(
            COMPLETION_RETRY_ASSIGNMENT_INDICES
        ),
        "protected_docs_tree": expected_docs_tree,
        "observed_docs_tree": current_state["docs_tree"],
    }


def apply_postcompletion_docs_overlay(
    campaign_id: str,
    *,
    root: Path = ledger.LEDGER_ROOT,
    review_queue_path: Path | None = None,
) -> dict[str, Any]:
    """Apply RES027's exact docs-only overlay after execution completes."""
    root = Path(root)
    review_queue_path = review_queue_path or (
        ledger.LEDGER_ROOT.parent.parent
        / "notes/annotation-refresh-campaign-review-queue-v1.json"
    )
    queue = _load_json(review_queue_path)
    resolution = next(
        (
            row
            for row in queue.get("resolutions", [])
            if row.get("resolution_id")
            == POSTCOMPLETION_DOCS_OVERLAY_RESOLUTION_ID
        ),
        None,
    )
    if (
        resolution is None
        or resolution.get("applies_to_items") != ["ARCV1-G007"]
        or resolution.get("authority_quote")
        != POSTCOMPLETION_DOCS_OVERLAY_AUTHORITY_QUOTE
        or resolution.get("decided_by") != "repository_user"
        or resolution.get("decision")
        != "approved_postcompletion_docs_protected_state_overlay"
    ):
        raise IntegrityDriftError("ARCV1-RES027 owner amendment is absent/drifted")

    campaign = load_provisional_campaign(campaign_id, root=root)
    run_dir = ledger.work_dir(campaign["run_id"], root)
    prior_amendments = _validated_execution_amendments(run_dir)
    existing = next(
        (
            row
            for row in prior_amendments
            if row.get("resolution_id")
            == POSTCOMPLETION_DOCS_OVERLAY_RESOLUTION_ID
        ),
        None,
    )
    if existing is not None:
        return {
            "status": "already_applied",
            "campaign_id": campaign_id,
            "amendment_id": existing["amendment_id"],
            "protected_docs_tree":
            _effective_protected_baseline(run_dir)["docs_tree"],
        }
    if not any(
        row.get("resolution_id") == COMPLETION_AMENDMENT_RESOLUTION_ID
        for row in prior_amendments
    ):
        raise IntegrityDriftError("RES027 requires the sealed RES026 amendment")

    connection = _connect(run_dir)
    try:
        counts = {
            row["status"]: int(row["n"])
            for row in connection.execute(
                "SELECT status, COUNT(*) AS n FROM assignments GROUP BY status"
            )
        }
        attempt_counts = {
            row["status"]: int(row["n"])
            for row in connection.execute(
                "SELECT status, COUNT(*) AS n FROM attempts GROUP BY status"
            )
        }
        reserved = int(
            connection.execute(
                "SELECT COALESCE(SUM(input_proxy_tokens), 0) AS n FROM attempts"
            ).fetchone()["n"]
        )
    finally:
        connection.close()
    if (
        counts != {"accepted": EXPECTED_ASSIGNMENTS}
        or attempt_counts
        != {
            "accepted": EXPECTED_ASSIGNMENTS,
            "schema_invalid": 1_047,
        }
        or reserved != 91_043_250
        or _effective_input_token_ceiling(run_dir)
        != COMPLETION_INPUT_TOKEN_CEILING
    ):
        raise IntegrityDriftError("RES027 completion checkpoint drift")

    effective_baseline = _effective_protected_baseline(run_dir)
    current_state = _protected_state(root)
    if any(
        current_state.get(key) != value
        for key, value in effective_baseline.items()
        if key != "docs_tree"
    ):
        raise IntegrityDriftError(
            "a non-docs protected component drifted before RES027"
        )
    authorized_docs_tree = {
        "files": 348,
        "sha256": (
            "sha256:"
            "e96417db8b1b8e9b8bddb05f3755df498ba8842ab3fc5c097e38591152ba4705"
        ),
    }
    if current_state.get("docs_tree") != authorized_docs_tree:
        raise IntegrityDriftError("RES027 authorized docs-tree digest drift")

    semantic = {
        "amendment_version":
        "arcv1-postcompletion-docs-protected-state-overlay-v1",
        "campaign_id": campaign_id,
        "run_id": campaign["run_id"],
        "plan_id": PLAN_ID,
        "resolution_id": POSTCOMPLETION_DOCS_OVERLAY_RESOLUTION_ID,
        "resolution_sha256": _canonical_sha(resolution),
        "execution_checkpoint": {
            "accepted_assignments": EXPECTED_ASSIGNMENTS,
            "accepted_subjects": EXPECTED_FRESH_SUBJECTS,
            "accepted_attempts": EXPECTED_ASSIGNMENTS,
            "invalid_attempts": 1_047,
            "reserved_input_proxy_tokens": 91_043_250,
            "input_token_ceiling": COMPLETION_INPUT_TOKEN_CEILING,
        },
        "protected_state_overlay": {
            "field": "docs_tree",
            "previous_value": effective_baseline["docs_tree"],
            "replacement_value": authorized_docs_tree,
            "all_other_components_unchanged": True,
            "concurrent_output_owner": "repository_user",
        },
        "docs_modified_or_regenerated": False,
        "confidence_status": "provisional_not_promoted",
        "production_eligible": False,
        "prohibited_effects_unchanged": campaign["prohibited_effects"],
    }
    amendment = _artifact(
        semantic,
        prefix="pexam",
        id_field="amendment_id",
        sha_field="amendment_sha256",
    )
    amendment_dir = run_dir / AMENDMENTS_DIRNAME
    amendment_dir.mkdir(parents=True, exist_ok=True)
    _publish_exact_json(
        amendment_dir / f"{amendment['amendment_id']}.json",
        amendment,
    )
    if _effective_protected_baseline(run_dir) != current_state:
        raise IntegrityDriftError("RES027 protected-state overlay did not reconcile")

    connection = _connect(run_dir)
    try:
        connection.execute(
            "UPDATE meta SET value_json = ? WHERE key = 'execution_status'",
            (ledger.canonical_json("completed_ready_for_audit"),),
        )
        connection.commit()
    finally:
        connection.close()
    return {
        "status": "applied",
        "campaign_id": campaign_id,
        "amendment_id": amendment["amendment_id"],
        "protected_docs_tree": authorized_docs_tree,
        "assignments": EXPECTED_ASSIGNMENTS,
        "subjects": EXPECTED_FRESH_SUBJECTS,
    }


def audit_provisional_execution(
    campaign_id: str,
    *,
    root: Path = ledger.LEDGER_ROOT,
    consolidate: bool = True,
) -> dict[str, Any]:
    """Verify locked partition, receipts, fields, ceiling, and side effects."""
    campaign = load_provisional_campaign(campaign_id, root=root)
    run_id = campaign["run_id"]
    run_dir = ledger.work_dir(run_id, root)
    connection = _connect(run_dir)
    issues: list[str] = []
    try:
        connection.execute("BEGIN IMMEDIATE")
        partition_audit = _verify_locked_execution_partition(
            campaign=campaign,
            run_dir=run_dir,
            connection=connection,
            root=Path(root),
        )
        counts = {
            row["status"]: int(row["n"])
            for row in connection.execute(
                "SELECT status, COUNT(*) AS n FROM assignments GROUP BY status"
            )
        }
        if counts != {"accepted": EXPECTED_ASSIGNMENTS}:
            raise IntegrityDriftError(f"assignment completion mismatch: {counts}")
        attempts = list(
            connection.execute(
                "SELECT * FROM attempts ORDER BY assignment_id, attempt_no"
            )
        )
        if any(row["status"] == "started" for row in attempts):
            raise IntegrityDriftError("started attempt remains unresolved")
        attempts_by_assignment = Counter(
            row["assignment_id"] for row in attempts
        )
        for assignment_id, count in attempts_by_assignment.items():
            if count > _attempt_limit(connection, assignment_id):
                raise IntegrityDriftError("assignment attempt-limit drift")
        exception_row = connection.execute(
            "SELECT value_json FROM meta WHERE key = 'attempt_limit_exceptions'"
        ).fetchone()
        actual_exceptions = (
            json.loads(exception_row["value_json"])
            if exception_row is not None
            else {}
        )
        if actual_exceptions != _declared_attempt_limit_exceptions(run_dir):
            raise IntegrityDriftError("attempt-limit exception provenance drift")
        reserved = sum(int(row["input_proxy_tokens"]) for row in attempts)
        effective_ceiling = _effective_input_token_ceiling(run_dir)
        if reserved > effective_ceiling:
            raise TokenCeilingError(f"{reserved} > {effective_ceiling}")
        sessions = {
            row["session_receipt_id"]: json.loads(row["receipt_json"])
            for row in connection.execute("SELECT * FROM sessions")
        }
        for receipt in sessions.values():
            _validate_worker_runtime_receipt(receipt)
        accepted_attempts = {
            row["attempt_id"]: row
            for row in attempts
            if row["status"] == "accepted"
        }
        if len(accepted_attempts) != EXPECTED_ASSIGNMENTS:
            raise IntegrityDriftError("accepted-attempt count mismatch")

        index_rows = ledger.read_jsonl(run_dir / "assignment-index.jsonl")
        if len(index_rows) != EXPECTED_ASSIGNMENTS:
            raise IntegrityDriftError("assignment index count mismatch")
        assignments = {
            row["assignment_id"]: row
            for row in connection.execute(
                "SELECT * FROM assignments ORDER BY assignment_index"
            )
        }
        if set(assignments) != {
            path.stem for path in (run_dir / "assignments").glob("*.json")
        }:
            raise IntegrityDriftError("assignment artifact/control key drift")

        all_events: list[dict[str, Any]] = []
        all_projections: list[dict[str, Any]] = []
        all_fields: list[dict[str, Any]] = []
        packages = []
        for row in index_rows:
            assignment_id = row["assignment_id"]
            package = _load_json(
                run_dir / "accepted-responses" / f"{assignment_id}.json"
            )
            semantic = dict(package)
            declared_id = semantic.pop("acceptance_id")
            declared_sha = semantic.pop("acceptance_sha256")
            observed_sha = _canonical_sha(semantic)
            if (
                declared_sha != observed_sha
                or declared_id
                != "acc_" + observed_sha.removeprefix("sha256:")
            ):
                raise IntegrityDriftError("accepted package identity drift")
            attempt_id = package["attempt_id"]
            attempt = accepted_attempts.get(attempt_id)
            if (
                attempt is None
                or attempt["assignment_id"] != assignment_id
                or package["session_receipt_id"] not in sessions
                or attempt["session_receipt_id"]
                != package["session_receipt_id"]
                or attempt["accepted_package_path"]
                != f"accepted-responses/{assignment_id}.json"
            ):
                raise IntegrityDriftError("acceptance attempt/session drift")
            raw_text = json.loads(attempt["raw_output_json"])
            execution_receipt = json.loads(attempt["execution_receipt_json"])
            if (
                not isinstance(raw_text, str)
                or attempt["submission_sha256"]
                != ledger.sha256_text(raw_text)
                or package["execution_receipt"] != execution_receipt
                or package["saved_response"]["response"] != json.loads(raw_text)
                or package["saved_response"]["run_id"] != run_id
                or package["saved_response"]["assignment_id"] != assignment_id
                or package["saved_response"]["labeled_at"]
                != attempt["completed_at"]
            ):
                raise IntegrityDriftError("accepted response/attempt receipt drift")
            _validate_execution_receipt(
                execution_receipt,
                runtime_receipt=sessions[package["session_receipt_id"]],
                invocation_error_code=None,
            )
            assignment = _load_json(
                run_dir / "assignments" / f"{assignment_id}.json"
            )
            expected_fields = _field_rows(
                assignment,
                package["saved_response"]["response"],
                attempt_id=attempt_id,
                session_receipt_id=package["session_receipt_id"],
                root=Path(root),
            )
            if package["field_values"] != expected_fields:
                raise IntegrityDriftError("accepted package field projection drift")
            packages.append(package)
            all_events.extend(package["events"])
            all_projections.extend(package["projections"])
            all_fields.extend(package["field_values"])
        field_keys = [
            (row["canonical_subject_id"], row["label_type"])
            for row in all_fields
        ]
        by_subject: dict[str, set[str]] = defaultdict(set)
        for subject_id, label_type in field_keys:
            by_subject[subject_id].add(label_type)
        if (
            len(all_fields) != EXPECTED_FRESH_FIELDS
            or len(set(field_keys)) != EXPECTED_FRESH_FIELDS
            or len(by_subject) != EXPECTED_FRESH_SUBJECTS
            or any(len(labels) != 7 for labels in by_subject.values())
        ):
            raise IntegrityDriftError("fresh subject-label completeness drift")

        baseline = _effective_protected_baseline(run_dir)
        if _protected_state(root) != baseline:
            raise IntegrityDriftError("prohibited side-effect baseline changed")
        if consolidate:
            for package in packages:
                response = package["saved_response"]
                _publish_exact_json(
                    run_dir
                    / "responses"
                    / f"{response['assignment_id']}.json",
                    response,
                )
            ledger.write_gzip_jsonl(run_dir / "label_events.jsonl", all_events)
            ledger.write_gzip_jsonl(
                run_dir / "canonical_label_projection.jsonl", all_projections
            )
            ledger.write_gzip_jsonl(
                run_dir / SUBJECT_FIELDS_FILENAME,
                sorted(
                    all_fields,
                    key=lambda row: (
                        row["canonical_subject_id"],
                        row["label_type"],
                    ),
                ),
            )
            _export_control_receipts(connection, run_dir)
        connection.execute("COMMIT")
        generic = ledger.audit_work_run(
            run_id, root=root, registry_path=ledger.REGISTRY_V2_PATH
        )
        if generic["status"] != "ok":
            raise IntegrityDriftError(
                f"generic ledger audit failed: {generic['issues']}"
            )
        return {
            "status": "ok",
            "campaign_id": campaign_id,
            "run_id": run_id,
            "assignments": EXPECTED_ASSIGNMENTS,
            "subjects": EXPECTED_FRESH_SUBJECTS,
            "subject_label_fields": EXPECTED_FRESH_FIELDS,
            "duplicate_or_missing_field_keys": 0,
            "attempts": len(attempts),
            "runtime_sessions": len(sessions),
            "reserved_input_proxy_tokens": reserved,
            "input_token_ceiling": effective_ceiling,
            "locked_partition_audit": partition_audit,
            "execution_amendment_ids": [
                row["amendment_id"]
                for row in _validated_execution_amendments(run_dir)
            ],
            "generic_ledger_audit": generic,
            "issues": [],
        }
    except Exception as exc:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        issues.append(str(exc))
        return {
            "status": "failed",
            "campaign_id": campaign_id,
            "run_id": run_id,
            "issues": issues,
        }
    finally:
        connection.close()


def seal_provisional_execution(
    campaign_id: str, *, root: Path = ledger.LEDGER_ROOT
) -> dict[str, Any]:
    campaign = load_provisional_campaign(campaign_id, root=root)
    run_id = campaign["run_id"]
    if run_id in ledger.sealed_runs(root):
        seal = ledger.verify_sealed_run(run_id, root=root)
        return {
            "status": "already_sealed",
            "run_id": run_id,
            "artifact_set_sha256": seal["artifact_set_sha256"],
        }
    audit = audit_provisional_execution(
        campaign_id, root=root, consolidate=True
    )
    if audit["status"] != "ok":
        raise IntegrityDriftError(f"provisional audit failed: {audit['issues']}")
    run_dir = ledger.work_dir(run_id, root)
    connection = _connect(run_dir)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode = DELETE")
    finally:
        connection.close()
    result = ledger.seal_run(
        run_id, root=root, registry_path=ledger.REGISTRY_V2_PATH
    )
    verified = ledger.verify_sealed_run(run_id, root=root)
    if result["artifact_set_sha256"] != verified["artifact_set_sha256"]:
        raise IntegrityDriftError("fresh run seal verification drift")
    return result


def _composite_fields(
    reuse: Mapping[str, Any],
    fresh_rows: Iterable[Mapping[str, Any]],
    *,
    fresh_run_id: str,
    fresh_seal_sha256: str,
) -> list[dict[str, Any]]:
    rows = []
    for entry in reuse["entries"]:
        rows.append(
            {
                "canonical_subject_id": entry["canonical_subject_id"],
                "label_type": entry["label_type"],
                "value_json": ledger.canonical_json(entry["selected_value"]),
                "value_sha256": entry["selected_value_sha256"],
                "provenance_kind": "evaluated_reuse",
                "source_artifact_id": reuse["reuse_manifest_id"],
                "source_artifact_sha256": reuse["reuse_manifest_sha256"],
                "confidence_provenance": entry["confidence_provenance"],
                "fresh_run_id": None,
                "fresh_run_artifact_set_sha256": None,
                "assignment_id": None,
                "attempt_id": None,
                "session_receipt_id": None,
            }
        )
    for row in fresh_rows:
        value = dict(row)
        value.update(
            {
                "source_artifact_id": fresh_run_id,
                "source_artifact_sha256": fresh_seal_sha256,
                "confidence_provenance": "fresh_single_pass",
                "fresh_run_id": fresh_run_id,
                "fresh_run_artifact_set_sha256": fresh_seal_sha256,
            }
        )
        rows.append(value)
    rows.sort(key=lambda row: (row["canonical_subject_id"], row["label_type"]))
    return rows


def publish_provisional_composite(
    campaign_id: str, *, root: Path = ledger.LEDGER_ROOT
) -> dict[str, Any]:
    """Publish the content-addressed non-promoted reuse+fresh composite."""
    campaign = load_provisional_campaign(campaign_id, root=root)
    run_id = campaign["run_id"]
    seal = ledger.verify_sealed_run(run_id, root=root)
    sealed_dir = ledger.sealed_artifact_dir(run_id, root)
    baseline = _effective_protected_baseline(sealed_dir)
    if _protected_state(root) != baseline:
        raise IntegrityDriftError(
            "prohibited side-effect baseline changed before composite publication"
        )
    reuse = provisional._validate_artifact(
        _load_json(
            Path(root)
            / provisional.REUSE_MANIFESTS_DIR
            / f"{REUSE_ID}.json"
        ),
        prefix="reuse",
        id_field="reuse_manifest_id",
        sha_field="reuse_manifest_sha256",
    )
    fresh_rows = ledger.read_jsonl(sealed_dir / SUBJECT_FIELDS_FILENAME)
    fields = _composite_fields(
        reuse,
        fresh_rows,
        fresh_run_id=run_id,
        fresh_seal_sha256=seal["artifact_set_sha256"],
    )
    keys = [(row["canonical_subject_id"], row["label_type"]) for row in fields]
    provenance = Counter(row["provenance_kind"] for row in fields)
    subjects = {row["canonical_subject_id"] for row in fields}
    if (
        len(fields) != EXPECTED_TOTAL_FIELDS
        or len(set(keys)) != EXPECTED_TOTAL_FIELDS
        or len(subjects) != EXPECTED_TOTAL_SUBJECTS
        or provenance
        != {
            "evaluated_reuse": EXPECTED_REUSED_FIELDS,
            "fresh_single_pass": EXPECTED_FRESH_FIELDS,
        }
    ):
        raise IntegrityDriftError("composite key/count/provenance audit failed")

    composites_root = Path(root) / COMPOSITES_DIR
    candidate = composites_root / f".candidate.{os.getpid()}.tmp"
    if candidate.exists():
        raise FileExistsError(candidate)
    candidate.mkdir(parents=True)
    try:
        fields_path = candidate / "subject_label_fields.jsonl.gz"
        ledger.write_gzip_jsonl(fields_path, fields)
        fields_sha = ledger.sha256_bytes(fields_path.read_bytes())
        manifest = _artifact(
            {
                "composite_version": COMPOSITE_VERSION,
                "scope": "provisional-corpus",
                "layer_status": "provisional",
                "production_eligible": False,
                "promotion_status": "not_promoted",
                "materialization_status": "not_materialized",
                "plan_id": PLAN_ID,
                "plan_sha256": campaign["plan_sha256"],
                "reuse_manifest_id": REUSE_ID,
                "reuse_manifest_sha256": reuse["reuse_manifest_sha256"],
                "fresh_run_id": run_id,
                "fresh_run_artifact_set_sha256": seal[
                    "artifact_set_sha256"
                ],
                "join_keys": ["canonical_subject_id", "label_type"],
                "reuse_and_fresh_provenance_distinct": True,
                "fields_file": fields_path.name,
                "fields_file_sha256": fields_sha,
                "counts": {
                    "fresh_assignments": EXPECTED_ASSIGNMENTS,
                    "fresh_subjects": EXPECTED_FRESH_SUBJECTS,
                    "fresh_subject_label_fields": EXPECTED_FRESH_FIELDS,
                    "reused_subjects": EXPECTED_REUSED_SUBJECTS,
                    "reused_subject_label_fields": EXPECTED_REUSED_FIELDS,
                    "total_subjects": EXPECTED_TOTAL_SUBJECTS,
                    "total_subject_label_fields": EXPECTED_TOTAL_FIELDS,
                    "missing_or_duplicate_keys": 0,
                },
            },
            prefix="pcomp",
            id_field="composite_id",
            sha_field="composite_sha256",
        )
        _publish_exact_json(candidate / "manifest.json", manifest)
        inventory = ledger._member_inventory(candidate, exclude={"seal.json"})
        artifact_set_sha = ledger.artifact_root(inventory)
        _publish_exact_json(
            candidate / "seal.json",
            {
                "composite_id": manifest["composite_id"],
                "artifact_set_sha256": artifact_set_sha,
                "members": inventory,
            },
        )
        final = composites_root / manifest["composite_id"]
        composites_root.mkdir(parents=True, exist_ok=True)
        with ledger.ledger_lock(root):
            if final.exists():
                existing = verify_provisional_composite(
                    manifest["composite_id"], root=root
                )
                shutil.rmtree(candidate)
                return {"status": "already_published", **existing}
            os.replace(candidate, final)
            ledger.fsync_directory(composites_root)
        return {
            "status": "published",
            "composite_id": manifest["composite_id"],
            "composite_sha256": manifest["composite_sha256"],
            "artifact_set_sha256": artifact_set_sha,
            "path": str(final),
            "counts": manifest["counts"],
        }
    except Exception:
        # Preserve the candidate for explicit inspection.
        raise


def verify_provisional_composite(
    composite_id: str, *, root: Path = ledger.LEDGER_ROOT
) -> dict[str, Any]:
    if not ledger.ID_PREFIX_RE.fullmatch(composite_id) or not composite_id.startswith(
        "pcomp_"
    ):
        raise ValueError("invalid composite id")
    directory = Path(root) / COMPOSITES_DIR / composite_id
    manifest = _load_json(directory / "manifest.json")
    semantic = dict(manifest)
    declared_id = semantic.pop("composite_id")
    declared_sha = semantic.pop("composite_sha256")
    observed_sha = _canonical_sha(semantic)
    if (
        declared_id != composite_id
        or declared_sha != observed_sha
        or composite_id != "pcomp_" + observed_sha.removeprefix("sha256:")
    ):
        raise IntegrityDriftError("composite manifest identity drift")
    seal = _load_json(directory / "seal.json")
    inventory = ledger._member_inventory(directory, exclude={"seal.json"})
    if (
        seal.get("composite_id") != composite_id
        or seal.get("members") != inventory
        or seal.get("artifact_set_sha256") != ledger.artifact_root(inventory)
    ):
        raise IntegrityDriftError("composite artifact-set drift")
    fields_path = directory / manifest["fields_file"]
    if ledger.sha256_bytes(fields_path.read_bytes()) != manifest[
        "fields_file_sha256"
    ]:
        raise IntegrityDriftError("composite fields hash drift")
    if (
        manifest.get("scope") != "provisional-corpus"
        or manifest.get("layer_status") != "provisional"
        or manifest.get("production_eligible") is not False
        or manifest.get("promotion_status") != "not_promoted"
        or manifest.get("materialization_status") != "not_materialized"
        or manifest.get("plan_id") != PLAN_ID
    ):
        raise IntegrityDriftError("composite provisional-status drift")
    fresh_run_id = manifest["fresh_run_id"]
    fresh_seal = ledger.verify_sealed_run(fresh_run_id, root=root)
    if (
        fresh_seal["artifact_set_sha256"]
        != manifest["fresh_run_artifact_set_sha256"]
    ):
        raise IntegrityDriftError("composite fresh-run seal drift")
    sealed_dir = ledger.sealed_artifact_dir(fresh_run_id, root)
    if _protected_state(root) != _effective_protected_baseline(sealed_dir):
        raise IntegrityDriftError(
            "prohibited side-effect baseline changed after composite publication"
        )
    fields = ledger.read_jsonl(fields_path)
    keys = [(row["canonical_subject_id"], row["label_type"]) for row in fields]
    provenance = Counter(row["provenance_kind"] for row in fields)
    if (
        len(fields) != manifest["counts"]["total_subject_label_fields"]
        or len(set(keys)) != len(fields)
        or len({row["canonical_subject_id"] for row in fields})
        != manifest["counts"]["total_subjects"]
        or provenance
        != {
            "evaluated_reuse": manifest["counts"][
                "reused_subject_label_fields"
            ],
            "fresh_single_pass": manifest["counts"][
                "fresh_subject_label_fields"
            ],
        }
        or any(
            ledger.sha256_text(row["value_json"]) != row["value_sha256"]
            for row in fields
        )
    ):
        raise IntegrityDriftError("composite field/key/provenance drift")
    return {
        "composite_id": composite_id,
        "composite_sha256": declared_sha,
        "artifact_set_sha256": seal["artifact_set_sha256"],
        "path": str(directory),
        "counts": manifest["counts"],
    }


def _read_receipt(path: Path) -> dict[str, Any]:
    return _load_json(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ledger.LEDGER_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-plan")
    verify.add_argument("--runtime-receipt", type=Path, required=True)
    verify.add_argument("--review-queue", type=Path)

    initialize = subparsers.add_parser("init")
    initialize.add_argument("--runtime-receipt", type=Path, required=True)
    initialize.add_argument("--review-queue", type=Path)
    initialize.add_argument("--initialized-by", required=True)

    claim = subparsers.add_parser("claim")
    claim.add_argument("--campaign-id", required=True)
    claim.add_argument("--runtime-receipt", type=Path, required=True)
    claim.add_argument("--range-start", type=int, default=0)
    claim.add_argument("--range-end", type=int, default=EXPECTED_ASSIGNMENTS)
    claim.add_argument("--lease-seconds", type=int, default=1800)

    start = subparsers.add_parser("start-attempt")
    start.add_argument("--campaign-id", required=True)
    start.add_argument("--assignment-id", required=True)
    start.add_argument("--claim-id", required=True)
    start.add_argument("--runtime-receipt", type=Path, required=True)

    complete = subparsers.add_parser("complete-attempt")
    complete.add_argument("--campaign-id", required=True)
    complete.add_argument("--attempt-id", required=True)
    complete.add_argument("--raw-output", type=Path, required=True)
    complete.add_argument("--runtime-receipt", type=Path, required=True)
    complete.add_argument("--execution-receipt", type=Path, required=True)
    complete.add_argument(
        "--invocation-error-code",
        choices=["invocation_transient_failure"],
    )

    status = subparsers.add_parser("status")
    status.add_argument("--campaign-id", required=True)

    compact = subparsers.add_parser("compact")
    compact.add_argument("--campaign-id", required=True)

    recover = subparsers.add_parser("recover-orphans")
    recover.add_argument("--campaign-id", required=True)
    recover.add_argument("--reason", required=True)

    amendment = subparsers.add_parser("apply-integrity-amendment")
    amendment.add_argument("--campaign-id", required=True)
    amendment.add_argument("--review-queue", type=Path)

    validator_repair = subparsers.add_parser(
        "apply-validator-repair-amendment"
    )
    validator_repair.add_argument("--campaign-id", required=True)
    validator_repair.add_argument("--review-queue", type=Path)

    validator_retry_policy = subparsers.add_parser(
        "apply-validator-retry-policy-amendment"
    )
    validator_retry_policy.add_argument("--campaign-id", required=True)
    validator_retry_policy.add_argument("--review-queue", type=Path)

    enhanced_validator_retry = subparsers.add_parser(
        "apply-enhanced-validator-retry-amendment"
    )
    enhanced_validator_retry.add_argument("--campaign-id", required=True)
    enhanced_validator_retry.add_argument("--review-queue", type=Path)

    targeted_literal_retry = subparsers.add_parser(
        "apply-targeted-literal-retry-policy-amendment"
    )
    targeted_literal_retry.add_argument("--campaign-id", required=True)
    targeted_literal_retry.add_argument("--review-queue", type=Path)

    completion_amendment = subparsers.add_parser(
        "apply-completion-amendment"
    )
    completion_amendment.add_argument("--campaign-id", required=True)
    completion_amendment.add_argument("--review-queue", type=Path)

    docs_overlay = subparsers.add_parser(
        "apply-postcompletion-docs-overlay"
    )
    docs_overlay.add_argument("--campaign-id", required=True)
    docs_overlay.add_argument("--review-queue", type=Path)

    grant_literal_retries = subparsers.add_parser(
        "grant-automatic-literal-retries"
    )
    grant_literal_retries.add_argument("--campaign-id", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--campaign-id", required=True)

    seal = subparsers.add_parser("seal")
    seal.add_argument("--campaign-id", required=True)

    composite = subparsers.add_parser("publish-composite")
    composite.add_argument("--campaign-id", required=True)

    verify_composite = subparsers.add_parser("verify-composite")
    verify_composite.add_argument("--composite-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify-plan":
        result = verify_locked_plan(
            runtime_receipt=_read_receipt(args.runtime_receipt),
            root=args.root,
            review_queue_path=args.review_queue,
        )
        result = {
            "status": result["status"],
            "plan_id": result["plan"]["plan_id"],
            "reuse_manifest_id": result["reuse"]["reuse_manifest_id"],
            "fresh_selection_id": result["fresh"]["fresh_selection_id"],
            "counts": {
                "reused_subjects": result["reuse"]["counts"]["subjects"],
                "reused_fields": result["reuse"]["counts"][
                    "subject_label_entries"
                ],
                "fresh_subjects": result["fresh"]["n_subjects"],
                "fresh_assignments": result["fresh"]["n_assignments"],
            },
        }
    elif args.command == "init":
        result = initialize_provisional_execution(
            runtime_receipt=_read_receipt(args.runtime_receipt),
            initialized_by=args.initialized_by,
            root=args.root,
            review_queue_path=args.review_queue,
        )
    elif args.command == "claim":
        result = claim_provisional_assignment(
            args.campaign_id,
            runtime_receipt=_read_receipt(args.runtime_receipt),
            range_start=args.range_start,
            range_end=args.range_end,
            lease_seconds=args.lease_seconds,
            root=args.root,
        )
    elif args.command == "start-attempt":
        result = start_provisional_attempt(
            args.campaign_id,
            args.assignment_id,
            args.claim_id,
            runtime_receipt=_read_receipt(args.runtime_receipt),
            root=args.root,
        )
    elif args.command == "complete-attempt":
        result = complete_provisional_attempt(
            args.campaign_id,
            args.attempt_id,
            raw_output=args.raw_output.read_text(encoding="utf-8"),
            runtime_receipt=_read_receipt(args.runtime_receipt),
            execution_receipt=_read_receipt(args.execution_receipt),
            invocation_error_code=args.invocation_error_code,
            root=args.root,
        )
    elif args.command == "status":
        result = provisional_execution_status(
            args.campaign_id, root=args.root
        )
    elif args.command == "compact":
        result = compact_provisional_work_storage(
            args.campaign_id, root=args.root
        )
    elif args.command == "recover-orphans":
        result = recover_orphaned_attempts(
            args.campaign_id,
            reason=args.reason,
            root=args.root,
        )
    elif args.command == "apply-integrity-amendment":
        result = apply_integrity_recovery_amendment(
            args.campaign_id,
            root=args.root,
            review_queue_path=args.review_queue,
        )
    elif args.command == "apply-validator-repair-amendment":
        result = apply_validator_repair_amendment(
            args.campaign_id,
            root=args.root,
            review_queue_path=args.review_queue,
        )
    elif args.command == "apply-validator-retry-policy-amendment":
        result = apply_validator_retry_policy_amendment(
            args.campaign_id,
            root=args.root,
            review_queue_path=args.review_queue,
        )
    elif args.command == "apply-enhanced-validator-retry-amendment":
        result = apply_enhanced_validator_retry_amendment(
            args.campaign_id,
            root=args.root,
            review_queue_path=args.review_queue,
        )
    elif args.command == "apply-targeted-literal-retry-policy-amendment":
        result = apply_targeted_literal_retry_policy_amendment(
            args.campaign_id,
            root=args.root,
            review_queue_path=args.review_queue,
        )
    elif args.command == "apply-completion-amendment":
        result = apply_completion_amendment(
            args.campaign_id,
            root=args.root,
            review_queue_path=args.review_queue,
        )
    elif args.command == "apply-postcompletion-docs-overlay":
        result = apply_postcompletion_docs_overlay(
            args.campaign_id,
            root=args.root,
            review_queue_path=args.review_queue,
        )
    elif args.command == "grant-automatic-literal-retries":
        result = grant_automatic_literal_retry_exceptions(
            args.campaign_id,
            root=args.root,
        )
    elif args.command == "audit":
        result = audit_provisional_execution(
            args.campaign_id, root=args.root
        )
    elif args.command == "seal":
        result = seal_provisional_execution(
            args.campaign_id, root=args.root
        )
    elif args.command == "publish-composite":
        result = publish_provisional_composite(
            args.campaign_id, root=args.root
        )
    elif args.command == "verify-composite":
        result = verify_provisional_composite(
            args.composite_id, root=args.root
        )
    else:  # pragma: no cover - argparse owns the vocabulary
        raise AssertionError(args.command)
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
