"""Restricted Codex launcher for ARCV1 provisional assignments.

This is an execution adapter around the provider-neutral provisional control
plane.  Each assignment is sent to a fresh ephemeral ``codex exec`` session in
an empty read-only working directory.  The session receives only the claimed
rendered assignment and its locked output schema.  JSONL lifecycle events,
explicit model/effort configuration, usage, and the actual Codex thread ID are
persisted through the provisional attempt receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import annotation_ledger as ledger
from . import annotation_refresh_provisional_execution as execution


LAUNCHER_VERSION = "arcv1-codex-ephemeral-worker-v1"
MODEL_ID = "gpt-5.6-sol"
REASONING_EFFORT = "high"
DESKTOP_CODEX_BINARY = Path(
    "/Applications/ChatGPT.app/Contents/Resources/codex"
)
UNSUPPORTED_TRANSPORT_SCHEMA_KEYS = frozenset({"uniqueItems"})


def _codex_binary() -> str:
    """Prefer the desktop-bundled CLI that supports the active Sol runtime."""
    if DESKTOP_CODEX_BINARY.is_file():
        return str(DESKTOP_CODEX_BINARY)
    discovered = shutil.which("codex")
    if discovered is None:
        raise FileNotFoundError("Codex CLI is unavailable")
    return discovered


def _codex_version(codex_binary: str) -> str:
    result = subprocess.run(
        [codex_binary, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _runtime_receipt(
    launcher_session_id: str,
    *,
    codex_version: str,
    range_start: int,
    range_end: int,
) -> dict[str, Any]:
    return {
        "receipt_version": "codex-exec-runtime-profile-v1",
        "model_id": MODEL_ID,
        "specificity": "exact_runtime_identifier",
        "reasoning_effort": REASONING_EFFORT,
        "session_id": launcher_session_id,
        "source": "codex-exec-explicit-profile",
        "codex_version": codex_version,
        "model_identity_source": "command_line_model_flag",
        "reasoning_effort_source": "command_line_config_override",
        "sandbox": "read-only",
        "approval_policy": "never",
        "ephemeral": True,
        "ignore_user_config": True,
        "ignore_rules": True,
        "fresh_restricted_labeling_session": True,
        "sibling_responses_inspected": False,
        "assignment_range": [range_start, range_end],
        "launcher_version": LAUNCHER_VERSION,
    }


def _prompt(
    rendered: Mapping[str, Any], *, repair_feedback: str | None = None
) -> str:
    value = (
        "You are a restricted annotation worker. Use only the rendered "
        "assignment below. Do not use tools, inspect files, browse, or seek "
        "other context. Do not discuss the task. Return only one JSON object "
        "that conforms exactly to response_schema and covers every local_id. "
        "Apply the supplied prompt and target text literally.\n\n"
        + json.dumps(
            rendered,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    if repair_feedback is not None:
        value += "\n\n" + repair_feedback
    return value


def _transport_schema(value: Any) -> Any:
    """Project the locked schema onto OpenAI's structured-output subset.

    The complete locked schema remains in the model-facing prompt and is
    always enforced by the repository's local response validator.  This
    projection only removes provider-unsupported enforcement keywords.
    """
    if isinstance(value, Mapping):
        return {
            key: _transport_schema(item)
            for key, item in value.items()
            if key not in UNSUPPORTED_TRANSPORT_SCHEMA_KEYS
        }
    if isinstance(value, list):
        return [_transport_schema(item) for item in value]
    return value


def _parse_events(stdout: str) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    events = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("Codex JSONL event is not an object")
        events.append(value)
    thread_ids = [
        row.get("thread_id")
        for row in events
        if row.get("type") == "thread.started"
    ]
    completed = [
        row for row in events if row.get("type") == "turn.completed"
    ]
    prohibited_items = []
    for row in events:
        item = row.get("item")
        if isinstance(item, dict) and item.get("type") in {
            "command_execution",
            "file_change",
            "mcp_tool_call",
            "web_search",
        }:
            prohibited_items.append(item.get("type"))
    if prohibited_items:
        raise execution.IntegrityDriftError(
            f"restricted labeling worker used tools: {prohibited_items}"
        )
    if len(thread_ids) != 1 or not str(thread_ids[0]).strip():
        raise execution.IntegrityDriftError(
            "Codex worker did not emit one exact thread ID"
        )
    if len(completed) != 1 or not isinstance(completed[0].get("usage"), dict):
        raise execution.IntegrityDriftError(
            "Codex worker did not emit one completed usage receipt"
        )
    return events, str(thread_ids[0]), dict(completed[0]["usage"])


def _launch(
    rendered: Mapping[str, Any],
    *,
    launcher_session_id: str,
    codex_binary: str,
    codex_version: str,
    timeout_seconds: int,
    repair_feedback: str | None = None,
) -> tuple[str, dict[str, Any]]:
    prompt = _prompt(rendered, repair_feedback=repair_feedback)
    prompt_sha = ledger.sha256_text(prompt)
    with tempfile.TemporaryDirectory(prefix="arcv1-worker-") as temp_name:
        temp = Path(temp_name)
        isolated_codex_home = temp / "codex-home"
        isolated_codex_home.mkdir()
        source_codex_home = Path(
            os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))
        )
        auth_path = source_codex_home / "auth.json"
        if not auth_path.is_file():
            raise FileNotFoundError("Codex authentication receipt is unavailable")
        (isolated_codex_home / "auth.json").symlink_to(auth_path)
        schema_path = temp / "schema.json"
        output_path = temp / "response.json"
        locked_schema = rendered["response_schema"]
        transport_schema = _transport_schema(locked_schema)
        schema_path.write_text(
            json.dumps(
                transport_schema,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        command = [
            codex_binary,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--json",
            "--model",
            MODEL_ID,
            "--config",
            f'model_reasoning_effort="{REASONING_EFFORT}"',
            "--config",
            'approval_policy="never"',
            "--sandbox",
            "read-only",
            "--cd",
            str(temp),
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "-",
        ]
        command_profile = {
            "launcher_version": LAUNCHER_VERSION,
            "codex_binary": codex_binary,
            "codex_version": codex_version,
            "model_flag": MODEL_ID,
            "model_reasoning_effort": REASONING_EFFORT,
            "ephemeral": True,
            "ignore_user_config": True,
            "ignore_rules": True,
            "sandbox": "read-only",
            "approval_policy": "never",
            "working_directory_policy": "fresh_empty_temporary_directory",
            "codex_home_policy": "fresh_temporary_home_with_auth_symlink_only",
            "locked_response_schema_sha256": ledger.sha256_text(
                ledger.canonical_json(locked_schema)
            ),
            "output_schema_sha256": ledger.sha256_bytes(
                schema_path.read_bytes()
            ),
            "transport_schema_removed_keywords": sorted(
                UNSUPPORTED_TRANSPORT_SCHEMA_KEYS
            ),
            "prompt_sha256": prompt_sha,
            "repair_feedback_present": repair_feedback is not None,
            "repair_feedback_sha256": (
                ledger.sha256_text(repair_feedback)
                if repair_feedback is not None
                else None
            ),
        }
        try:
            result = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                env={**os.environ, "CODEX_HOME": str(isolated_codex_home)},
            )
        except subprocess.TimeoutExpired as exc:
            return "", {
                "session_id": launcher_session_id,
                "model_id": MODEL_ID,
                "reasoning_effort": REASONING_EFFORT,
                "source": "codex-exec-jsonl",
                "status": "failed",
                "error": f"timeout after {timeout_seconds}s",
                "command_profile": command_profile,
                "stdout_sha256": ledger.sha256_text(exc.stdout or ""),
                "stderr_sha256": ledger.sha256_text(exc.stderr or ""),
            }
        if result.returncode != 0 or not output_path.exists():
            return "", {
                "session_id": launcher_session_id,
                "model_id": MODEL_ID,
                "reasoning_effort": REASONING_EFFORT,
                "source": "codex-exec-jsonl",
                "status": "failed",
                "error": (
                    f"codex exec exit={result.returncode}; "
                    f"stderr_sha256={ledger.sha256_text(result.stderr)}"
                ),
                "command_profile": command_profile,
                "stdout_sha256": ledger.sha256_text(result.stdout),
                "stderr_sha256": ledger.sha256_text(result.stderr),
            }
        events, thread_id, usage = _parse_events(result.stdout)
        raw_output = output_path.read_text(encoding="utf-8")
        return raw_output, {
            "session_id": launcher_session_id,
            "codex_thread_id": thread_id,
            "model_id": MODEL_ID,
            "reasoning_effort": REASONING_EFFORT,
            "source": "codex-exec-jsonl",
            "status": "completed",
            "usage": usage,
            "command_profile": command_profile,
            "stdout_sha256": ledger.sha256_text(result.stdout),
            "stderr_sha256": ledger.sha256_text(result.stderr),
            "event_types": [row.get("type") for row in events],
            "raw_output_sha256": ledger.sha256_text(raw_output),
        }


def run_worker(
    campaign_id: str,
    *,
    range_start: int,
    range_end: int,
    max_assignments: int | None,
    timeout_seconds: int,
    root: Path,
) -> dict[str, Any]:
    root = Path(root).resolve()
    codex_binary = _codex_binary()
    codex_version = _codex_version(codex_binary)
    completed = 0
    invalid = 0
    while max_assignments is None or completed + invalid < max_assignments:
        launcher_session_id = "codex-launch-" + uuid.uuid4().hex
        runtime = _runtime_receipt(
            launcher_session_id,
            codex_version=codex_version,
            range_start=range_start,
            range_end=range_end,
        )
        claim = execution.claim_provisional_assignment(
            campaign_id,
            runtime_receipt=runtime,
            range_start=range_start,
            range_end=range_end,
            lease_seconds=min(7200, max(600, timeout_seconds + 300)),
            root=root,
        )
        if claim["status"] == "range_complete":
            break
        attempt = execution.start_provisional_attempt(
            campaign_id,
            claim["assignment_id"],
            claim["claim_id"],
            runtime_receipt=runtime,
            root=root,
        )
        raw_output, execution_receipt = _launch(
            claim["rendered_assignment"],
            launcher_session_id=launcher_session_id,
            codex_binary=codex_binary,
            codex_version=codex_version,
            timeout_seconds=timeout_seconds,
            repair_feedback=execution.validator_feedback_for_attempt(
                campaign_id,
                claim["assignment_id"],
                attempt["attempt_no"],
                root=root,
            ),
        )
        invocation_error = (
            None
            if execution_receipt["status"] == "completed"
            else "invocation_transient_failure"
        )
        receipt = execution.complete_provisional_attempt(
            campaign_id,
            attempt["attempt_id"],
            raw_output=raw_output,
            runtime_receipt=runtime,
            execution_receipt=execution_receipt,
            invocation_error_code=invocation_error,
            root=root,
        )
        if receipt["status"] == "accepted":
            completed += 1
        else:
            invalid += 1
        progress = {
            "worker_range": [range_start, range_end],
            "accepted_this_worker": completed,
            "invalid_this_worker": invalid,
            "last_assignment_index": claim["assignment_index"],
            "last_attempt_status": receipt["status"],
            "last_validation_code": receipt["validation_code"],
        }
        print(json.dumps(progress, sort_keys=True), flush=True)
        if receipt["status"] != "accepted":
            # The next loop retries mechanically with the same locked rendering.
            automatic_recovery = (
                execution.grant_automatic_literal_retry_exceptions(
                    campaign_id, root=root
                )
            )
            if automatic_recovery["granted"]:
                print(
                    json.dumps(
                        {
                            "automatic_literal_retry_grants":
                            automatic_recovery["granted"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            assignment_status = execution.provisional_assignment_status(
                campaign_id,
                claim["assignment_id"],
                root=root,
            )
            if assignment_status == "terminal_failed":
                raise execution.RetryExhaustedError(
                    "an assignment exhausted the frozen retry policy"
                )
            status = execution.provisional_execution_status(
                campaign_id, root=root
            )
            if status["status"] == "token_ceiling_blocked":
                raise execution.TokenCeilingError(
                    "the frozen input-token ceiling blocked execution"
                )
            if invocation_error is not None:
                # A short fixed delay prevents an infrastructure-level failure
                # from consuming all three mechanical attempts immediately.
                time.sleep(10)
            continue
    return {
        "status": "worker_complete",
        "range_start": range_start,
        "range_end": range_end,
        "accepted": completed,
        "invalid": invalid,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--range-start", type=int, required=True)
    parser.add_argument("--range-end", type=int, required=True)
    parser.add_argument("--max-assignments", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--root", type=Path, default=ledger.LEDGER_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_worker(
        args.campaign_id,
        range_start=args.range_start,
        range_end=args.range_end,
        max_assignments=args.max_assignments,
        timeout_seconds=args.timeout_seconds,
        root=args.root,
    )
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
