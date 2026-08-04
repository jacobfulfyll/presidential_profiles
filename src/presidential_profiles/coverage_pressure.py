"""Registered breadth/depth analysis for presidential coverage pressure.

All functions are pure enough to test on tiny paragraph tables.  The command
builds a compact public extract plus a full parquet and receipt table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import attention, corpus, inference
from .llm_annotations import load_paragraph_annotations

OUT_DIR = corpus.DATA_DIR / "coverage_pressure"
PRIMARY_ERAS = (1860, 1890, 1920)
MODERN_ERAS = (1950, 1980, 2010)
N_BOOTSTRAP = 20_000


def effective_topics(weights: np.ndarray | list[float]) -> float:
    """exp(Shannon entropy); zero weights are harmless and a one-topic item is 1."""
    w = np.asarray(weights, dtype=float)
    w = w[w > 0]
    if not len(w):
        return 0.0
    p = w / w.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def split_topic_words(word_count: int, topics: list[str]) -> dict[str, float]:
    """Equal word exposure for a multi-label paragraph, conserving its words."""
    unique = list(dict.fromkeys(topics))
    return {topic: word_count / len(unique) for topic in unique} if unique else {}


def topic_episodes(rows: pd.DataFrame) -> pd.DataFrame:
    """Consecutive paragraph runs per topic, including multi-label exposure."""
    episodes: list[dict] = []
    for topic, group in rows.sort_values("para_idx").groupby("topic", sort=False):
        run_words = 0.0
        run_paragraphs = 0
        previous = None
        for row in group.itertuples(index=False):
            idx = int(row.para_idx)
            if previous is not None and idx != previous + 1:
                episodes.append({"topic": topic, "run_words": run_words,
                                 "run_paragraphs": run_paragraphs})
                run_words = 0.0
                run_paragraphs = 0
            run_words += float(row.exposure_words)
            run_paragraphs += 1
            previous = idx
        if previous is not None:
            episodes.append({"topic": topic, "run_words": run_words,
                             "run_paragraphs": run_paragraphs})
    return pd.DataFrame(episodes, columns=["topic", "run_words", "run_paragraphs"])


def speech_metrics(assignments: pd.DataFrame) -> pd.DataFrame:
    """Compute specified coverage measures one speech at a time."""
    rows = []
    for doc, speech in assignments.groupby("doc_name", sort=False):
        exposure = speech.groupby("topic")["exposure_words"].sum()
        runs = topic_episodes(speech)
        total = float(exposure.sum())
        run_words = runs["run_words"].to_numpy(float)
        paragraph_topics = (speech.groupby("para_idx")["topic"]
                            .apply(set).sort_index().tolist())
        similarities = []
        for left, right in zip(paragraph_topics, paragraph_topics[1:]):
            union = left | right
            similarities.append(len(left & right) / len(union) if union else 0.0)
        rows.append({
            "doc_name": doc,
            "effective_topics": effective_topics(exposure.to_numpy()),
            "depth_words": float((run_words ** 2).sum() / run_words.sum()) if len(run_words) else np.nan,
            "leading_topic_share": float(exposure.max() / total) if total else np.nan,
            "median_episode_words": float(np.median(run_words)) if len(run_words) else np.nan,
            "singleton_run_share": float(np.mean(runs["run_paragraphs"] == 1)) if len(run_words) else np.nan,
            "adjacent_topic_similarity": float(np.mean(similarities)) if similarities else np.nan,
            "episodes_250_share": float(run_words[run_words >= 250].sum() / total) if total else np.nan,
            "episodes_500_share": float(run_words[run_words >= 500].sum() / total) if total else np.nan,
            "n_topics": int(len(exposure)),
        })
    return pd.DataFrame(rows)


def _era(year: int) -> int:
    return (year // 10) * 10


def hierarchical_difference(frame: pd.DataFrame, metric: str, draws: int = N_BOOTSTRAP,
                            seed: int = 20260722, weight_col: str | None = None,
                            controls: tuple[str, ...] = ()) -> tuple[float, np.ndarray]:
    """President-then-speech bootstrap, modern minus postbellum mean."""
    f = frame.dropna(subset=[metric]).copy()
    def estimate(part: pd.DataFrame) -> float:
        modern = (part.group == "modern").astype(float).to_numpy()
        if controls:
            x = np.column_stack([np.ones(len(part)), modern] +
                                [part[c].astype(float).to_numpy() for c in controls])
            y = part[metric].to_numpy(float)
            if weight_col:
                sw = np.sqrt(part[weight_col].to_numpy(float))
                x, y = x * sw[:, None], y * sw
            return float(np.linalg.lstsq(x, y, rcond=None)[0][1])
        weights = part[weight_col] if weight_col else None
        means = {g: np.average(p[metric], weights=(p[weight_col] if weight_col else None))
                 for g, p in part.groupby("group")}
        return float(means["modern"] - means["postbellum"])

    observed = estimate(f)
    rng = np.random.default_rng(seed)
    output = np.empty(draws)
    for i in range(draws):
        pieces = []
        for group, group_rows in f.groupby("group"):
            presidents = list(group_rows.president.unique())
            selected = rng.choice(presidents, len(presidents), replace=True)
            for draw_president, name in enumerate(selected):
                rows = group_rows[group_rows.president == name]
                sampled = rows.iloc[rng.integers(0, len(rows), len(rows))].copy()
                # Repeated president selections are distinct bootstrap clusters.
                sampled["_bootstrap_president"] = f"{group}:{draw_president}"
                pieces.append(sampled)
        output[i] = estimate(pd.concat(pieces, ignore_index=True))
    return float(observed), output


def hierarchical_differences(
    frame: pd.DataFrame, metric_names: list[str], draws: int = N_BOOTSTRAP,
    seed: int = 20260722, weight_col: str | None = None,
    controls: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Shared-resample bootstrap for many outcomes.

    Resampling indices once per arm preserves the exact president→speech design
    while avoiding a separate, expensive pandas reconstruction for every metric.
    """
    f = frame.dropna(subset=metric_names).reset_index(drop=True)
    y = f[metric_names].to_numpy(float)
    modern = (f.group == "modern").astype(float).to_numpy()
    weights = f[weight_col].to_numpy(float) if weight_col else np.ones(len(f))
    control_values = f[list(controls)].to_numpy(float) if controls else np.empty((len(f), 0))
    if controls:
        means, scales = control_values.mean(axis=0), control_values.std(axis=0)
        scales[scales == 0] = 1
        control_values = (control_values - means) / scales

    def estimate(indices: np.ndarray) -> np.ndarray:
        yi, mi, wi = y[indices], modern[indices], weights[indices]
        if controls:
            x = np.column_stack([np.ones(len(indices)), mi, control_values[indices]])
            sw = np.sqrt(wi)
            return np.linalg.lstsq(x * sw[:, None], yi * sw[:, None], rcond=None)[0][1]
        return np.array([
            np.average(yi[mi == 1, column], weights=wi[mi == 1]) -
            np.average(yi[mi == 0, column], weights=wi[mi == 0])
            for column in range(yi.shape[1])
        ])

    observed = estimate(np.arange(len(f)))
    pools = {}
    for group, group_rows in f.groupby("group"):
        pools[group] = [group_rows.index[group_rows.president == president].to_numpy()
                        for president in group_rows.president.unique()]
    rng = np.random.default_rng(seed)
    output = np.empty((draws, len(metric_names)))
    for draw in range(draws):
        indices = []
        for president_pools in pools.values():
            selected = rng.integers(0, len(president_pools), len(president_pools))
            for position in selected:
                speech_pool = president_pools[position]
                indices.extend(rng.choice(speech_pool, len(speech_pool), replace=True))
        output[draw] = estimate(np.asarray(indices, dtype=int))
    return observed, output


def joint_decision(breadth_adjusted: float, depth_adjusted: float,
                   breadth_estimate: float, depth_estimate: float,
                   alpha: float = .05) -> str:
    if breadth_adjusted <= alpha and depth_adjusted <= alpha:
        return "supported" if breadth_estimate > 0 and depth_estimate < 0 else "mixed"
    return "inconclusive"


def build(draws: int = N_BOOTSTRAP, out_dir: Path = OUT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    speeches = corpus.load()
    annotations = load_paragraph_annotations("paragraph_annotations")
    taxonomy = attention.load_taxonomy()
    names = attention.canonical_label_map(taxonomy)
    paras = pd.read_parquet(corpus.DATA_DIR / "paragraphs.parquet")
    merged = annotations.merge(paras, on=["doc_name", "para_idx"], validate="one_to_one")
    records = []
    for row in merged.itertuples(index=False):
        topics = attention.normalize_topics(row.topics, names)
        for topic, value in split_topic_words(int(row.word_count), topics).items():
            records.append((row.doc_name, int(row.para_idx), topic, value))
    assignments = pd.DataFrame(records, columns=["doc_name", "para_idx", "topic", "exposure_words"])
    speech_labels = pd.read_parquet(corpus.DATA_DIR / "llm_annotations" / "speech_annotations.parquet")
    metrics = speech_metrics(assignments).merge(
        speeches[["doc_name", "president", "year", "title", "word_count"]], on="doc_name", validate="one_to_one")
    metrics = metrics.merge(speech_labels[["doc_name", "speech_type", "medium"]],
                            on="doc_name", validate="one_to_one")
    metrics["era"] = metrics.year.map(_era)
    metrics["is_annual"] = metrics.speech_type.eq("state_of_the_union_or_annual_message")
    nparas = merged.groupby("doc_name").size()
    mean_para = merged.groupby("doc_name").word_count.mean()
    metrics["n_paragraphs"] = metrics.doc_name.map(nparas).astype(int)
    metrics["average_paragraph_words"] = metrics.doc_name.map(mean_para)
    metrics["log_speech_words"] = np.log(metrics.word_count.clip(lower=1))
    metrics["spoken_medium"] = metrics.medium.astype(str).str.contains(
        "spoken|radio|tv|television|broadcast|press|interview", case=False, regex=True).astype(int)
    metrics["group"] = np.select([metrics.era.isin(PRIMARY_ERAS), metrics.era.isin(MODERN_ERAS)],
                                 ["postbellum", "modern"], default="other")
    eligible = metrics[metrics.group != "other"].copy()
    primary = eligible[eligible.is_annual].copy()
    # paragraph-count rule belongs to the comparable sample, not arbitrary title text.
    primary = primary[primary.n_paragraphs >= 12]
    # Fixed common genre mix for the all-speech standardized sensitivity arm.
    common = eligible.speech_type.value_counts(normalize=True)
    within = eligible.groupby("group").speech_type.value_counts(normalize=True)
    eligible["genre_weight"] = [common.get(t, 0) / within.get((g, t), np.nan)
                                for g, t in zip(eligible.group, eligible.speech_type)]
    receipts = []
    raw_p = []
    primary_specs = [("effective_topics", "greater", "effective topics"),
                     ("depth_words", "less", "equivalent words")]
    primary_estimates, primary_boot = hierarchical_differences(
        primary, [item[0] for item in primary_specs], draws=draws)
    for column, (metric, direction, unit) in enumerate(primary_specs):
        estimate, boot = primary_estimates[column], primary_boot[:, column]
        p, floor = inference.bootstrap_p_value(boot - estimate, estimate, alternative=direction)
        raw_p.append(p)
        receipts.append(inference.InferenceReceipt(metric, estimate, unit,
            float(np.quantile(boot, .025)), float(np.quantile(boot, .975)), p, None, floor,
            "coverage-pressure primary outcomes", "president then speech", draws,
            "modern eras minus postbellum eras", "AI taxonomy v1", primary.president.nunique(),
            len(primary), "confirmatory", "Annual-message sample with at least 12 paragraphs."))
    adjusted = inference.holm_adjust(raw_p)
    receipts = [inference.InferenceReceipt(**{**r.__dict__, "p_adjusted": float(a)})
                for r, a in zip(receipts, adjusted)]
    # Publish every secondary outcome and sensitivity arm, clearly separated
    # from the two confirmatory tests.  Their intervals are descriptive.
    secondary = [("leading_topic_share", "share"), ("median_episode_words", "words"),
                 ("singleton_run_share", "share"), ("adjacent_topic_similarity", "Jaccard"),
                 ("episodes_250_share", "share"), ("episodes_500_share", "share")]
    arms = [("all speeches (raw)", eligible, None, ()),
            ("all speeches (genre-standardized)", eligible, "genre_weight", ()),
            ("annual messages with length/paragraph/medium controls", primary, None,
             ("log_speech_words", "average_paragraph_words", "spoken_medium"))]
    sensitivity_specs = [("effective_topics", "effective topics"),
                         ("depth_words", "equivalent words"), *secondary]
    for treatment, frame, weight, controls in arms:
        estimates, bootstraps = hierarchical_differences(
            frame, [item[0] for item in sensitivity_specs], draws=draws,
            weight_col=weight, controls=controls)
        for column, (metric, unit) in enumerate(sensitivity_specs):
            estimate, boot = estimates[column], bootstraps[:, column]
            receipts.append(inference.InferenceReceipt(metric, estimate, unit,
                float(np.quantile(boot, .025)), float(np.quantile(boot, .975)), None, None,
                1 / (draws + 1), "coverage-pressure sensitivity outcomes",
                "president then speech", draws, treatment, "AI taxonomy v1",
                frame.president.nunique(), len(frame), "exploratory",
                "Sensitivity estimate; not part of the two-outcome joint decision."))
    table = inference.receipts_frame(receipts)
    decision = joint_decision(float(adjusted[0]), float(adjusted[1]),
                              float(table.iloc[0].estimate), float(table.iloc[1].estimate))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(out_dir / "coverage_pressure.parquet", index=False)
    table.to_parquet(out_dir / "inference_receipts.parquet", index=False)
    table.to_csv(out_dir / "inference_receipts.csv", index=False)
    era_summary = (metrics.groupby(["era", "is_annual"])
                   [["effective_topics", "depth_words", "leading_topic_share",
                     "adjacent_topic_similarity"]].mean().reset_index())
    def clean_records(frame: pd.DataFrame) -> list[dict]:
        return [{key: (None if pd.isna(value) else value) for key, value in row.items()}
                for row in frame.to_dict("records")]
    (out_dir / "coverage_pressure.json").write_text(json.dumps({
        "decision": decision, "receipts": clean_records(table),
        "era_summary": clean_records(era_summary)}, indent=2, allow_nan=False))
    (out_dir / "manifest.json").write_text(json.dumps({"version": "coverage-pressure-v1", "bootstrap_draws": draws,
        "primary_sample": "annual messages / State of the Union with >=12 paragraphs", "postbellum_eras": PRIMARY_ERAS,
        "modern_eras": MODERN_ERAS, "decision": decision,
        "published_arms": ["primary", *[a[0] for a in arms]],
        "exclusions": {"under_12_paragraphs_from_primary": int((eligible.is_annual & (eligible.n_paragraphs < 12)).sum())}}, indent=2))
    return metrics, table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    _, receipts = build(args.draws, args.out_dir)
    print(receipts[["metric", "estimate", "p_value", "p_adjusted"]].to_string(index=False))


if __name__ == "__main__":
    main()
