"""Bounded concurrent coordinator for exact ARCV1 Codex workers."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from . import annotation_ledger as ledger
from . import annotation_refresh_codex_worker as worker
from . import annotation_refresh_provisional_execution as execution


def partition_ranges(
    range_start: int, range_end: int, workers: int
) -> list[tuple[int, int]]:
    if (
        range_start < 0
        or range_end <= range_start
        or workers < 1
        or range_end > execution.EXPECTED_ASSIGNMENTS
    ):
        raise ValueError("invalid coordinator range/workers")
    width = range_end - range_start
    count = min(workers, width)
    base, remainder = divmod(width, count)
    ranges = []
    cursor = range_start
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        ranges.append((cursor, cursor + size))
        cursor += size
    if cursor != range_end:
        raise AssertionError("range partition drift")
    return ranges


def run_coordinator(
    campaign_id: str,
    *,
    range_start: int,
    range_end: int,
    workers: int,
    timeout_seconds: int,
    root: Path,
) -> dict[str, Any]:
    root = Path(root).resolve()
    ranges = partition_ranges(range_start, range_end, workers)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str | int]] = []
    with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
        pending = {
            pool.submit(
                worker.run_worker,
                campaign_id,
                range_start=start,
                range_end=end,
                max_assignments=None,
                timeout_seconds=timeout_seconds,
                root=root,
            ): (start, end)
            for start, end in ranges
        }
        for future in as_completed(pending):
            start, end = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                errors.append(
                    {
                        "range_start": start,
                        "range_end": end,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            else:
                results.append(result)
            print(
                json.dumps(
                    {
                        "coordinator_range_complete": [start, end],
                        "completed_ranges": len(results),
                        "failed_ranges": len(errors),
                        "total_ranges": len(ranges),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    status = execution.provisional_execution_status(campaign_id, root=root)
    return {
        "status": "failed" if errors else "coordinator_complete",
        "range_start": range_start,
        "range_end": range_end,
        "workers": len(ranges),
        "worker_results": sorted(
            results, key=lambda row: int(row["range_start"])
        ),
        "worker_errors": sorted(
            errors, key=lambda row: int(row["range_start"])
        ),
        "campaign_status": status,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--range-start", type=int, required=True)
    parser.add_argument("--range-end", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--root", type=Path, default=ledger.LEDGER_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_coordinator(
        args.campaign_id,
        range_start=args.range_start,
        range_end=args.range_end,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
        root=args.root,
    )
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
