"""Small, explicit frequentist inference receipts used by public artifacts.

The site deliberately calls a p value a *no-change surprise rate*.  This
module keeps that language out of estimators while making the resolution and
multiple-testing family machine-readable.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import corpus


def p_value_from_draws(draws: Iterable[float], null: float = 0.0,
                       alternative: str = "two-sided") -> tuple[float, float]:
    """Tail area of bootstrap estimates relative to a null value."""
    values = np.asarray(list(draws), dtype=float)
    if values.size == 0:
        raise ValueError("at least one bootstrap/permutation draw is required")
    if alternative == "greater":
        extreme = np.count_nonzero(values <= null)
    elif alternative == "less":
        extreme = np.count_nonzero(values >= null)
    elif alternative == "two-sided":
        lower = np.count_nonzero(values <= null)
        upper = np.count_nonzero(values >= null)
        p = min(1.0, 2 * (min(lower, upper) + 1) / (values.size + 1))
        return p, 1 / (values.size + 1)
    else:
        raise ValueError("alternative must be greater, less, or two-sided")
    return (extreme + 1) / (values.size + 1), 1 / (values.size + 1)


def bootstrap_p_value(null_draws: Iterable[float], observed: float,
                      alternative: str = "two-sided") -> tuple[float, float]:
    values = np.asarray(list(null_draws), dtype=float)
    if alternative == "greater":
        extreme = np.count_nonzero(values >= observed)
    elif alternative == "less":
        extreme = np.count_nonzero(values <= observed)
    elif alternative == "two-sided":
        extreme = np.count_nonzero(np.abs(values) >= abs(observed))
    else:
        raise ValueError("alternative must be greater, less, or two-sided")
    return (extreme + 1) / (len(values) + 1), 1 / (len(values) + 1)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm step-down adjustment, preserving input order."""
    p = np.asarray(list(p_values), dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p values must fall in [0, 1]")
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    result = np.empty_like(p)
    result[order] = adjusted_sorted
    return result


def surprise_label(p_value: float, floor: float | None = None) -> str:
    """Plain-language display with an honest Monte-Carlo resolution floor."""
    if floor is not None and p_value <= floor + 1e-15:
        return f"No-change surprise: ≤{floor * 100:.3g}%, at the test’s resolution floor"
    return f"No-change surprise: {p_value * 100:.3g}%"


@dataclass(frozen=True)
class InferenceReceipt:
    metric: str
    estimate: float
    unit: str
    interval_low: float | None
    interval_high: float | None
    p_value: float | None
    p_adjusted: float | None
    p_resolution: float | None
    test_family: str
    bootstrap_unit: str
    bootstrap_draws: int
    treatment: str
    taxonomy: str
    n_presidents: int
    n_speeches: int
    status: str
    caveat: str = ""

    def record(self) -> dict:
        row = asdict(self)
        row["surprise_label"] = (surprise_label(row["p_value"], row["p_resolution"])
                                 if row["p_value"] is not None else "Not tested")
        return row


def receipts_frame(receipts: Iterable[InferenceReceipt]) -> pd.DataFrame:
    return pd.DataFrame([r.record() for r in receipts])


def build_common_receipts(data_dir: Path) -> pd.DataFrame:
    """Unify confirmatory, exploratory and descriptive inference metadata."""
    rows: list[dict] = []
    coverage = data_dir / "coverage_pressure" / "inference_receipts.parquet"
    if coverage.exists():
        frame = pd.read_parquet(coverage)
        for record in frame.to_dict("records"):
            record.update({"analysis": "coverage_pressure",
                           "family_size": 2 if record["status"] == "confirmatory" else 0})
            rows.append(record)

    upgraded = data_dir / "register" / "trends_20000.parquet"
    register_path = upgraded if upgraded.exists() else data_dir / "register" / "trends.parquet"
    if register_path.exists():
        register = pd.read_parquet(register_path)
        tests = register[register.p_value.notna()].copy()
        # The full set is disclosed as one exploratory family. Holm values are
        # useful for sorting but are never relabeled confirmatory.
        tests["p_adjusted"] = holm_adjust(tests.p_value)
        for record in tests.to_dict("records"):
            floor = 1 / (int(record["n_bootstrap"]) + 1)
            rows.append({
                "analysis": "register", "metric": record["measure"],
                "estimate": record["value"], "unit": record["measure"],
                "interval_low": record["ci_low"], "interval_high": record["ci_high"],
                "p_value": record["p_value"], "p_adjusted": record["p_adjusted"],
                "p_resolution": floor, "test_family": "registered-trends exploratory 444",
                "family_size": len(tests), "bootstrap_unit": "speech",
                "bootstrap_draws": int(record["n_bootstrap"]),
                "treatment": f"{record['genre_treatment']} · {record['statistic']}",
                "taxonomy": record["taxonomy"], "n_presidents": None,
                "n_speeches": (int(record["n_speeches"])
                               if pd.notna(record["n_speeches"]) else None), "status": (
                    "post-hoc" if record["statistic"] == "delta_modern_minus_postbellum"
                    else "exploratory"),
                "caveat": "One of 444 disclosed exploratory tests.",
                "surprise_label": surprise_label(record["p_value"], floor),
            })

    combat_path = data_dir / "combat" / "combativeness.parquet"
    if combat_path.exists():
        combat = pd.read_parquet(combat_path)
        for record in combat.to_dict("records"):
            rows.append({
                "analysis": "combativeness", "metric": record["flag"],
                "estimate": record["rate"], "unit": "share of paragraphs",
                "interval_low": record["ci_lo"], "interval_high": record["ci_hi"],
                "p_value": None, "p_adjusted": None, "p_resolution": None,
                "test_family": "descriptive intervals", "family_size": 0,
                "bootstrap_unit": "speech", "bootstrap_draws": int(record["n_bootstrap"]),
                "treatment": f"{record['era']} · {record['treatment']}",
                "taxonomy": "closed AI rubric", "n_presidents": None,
                "n_speeches": int(record["n_speeches"]), "status": "exploratory",
                "caveat": str(record["ci_status"]), "surprise_label": "Not tested",
            })

    output = pd.DataFrame(rows)
    out_path = data_dir / "inference_receipts.parquet"
    output.to_parquet(out_path, index=False)
    output.to_csv(data_dir / "inference_receipts.csv", index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the common frequentist inference-receipt artifact"
    )
    parser.add_argument("--data-dir", type=Path, default=corpus.DATA_DIR)
    args = parser.parse_args()
    receipts = build_common_receipts(args.data_dir)
    print(f"wrote {len(receipts)} inference receipts to {args.data_dir}")


if __name__ == "__main__":
    main()
