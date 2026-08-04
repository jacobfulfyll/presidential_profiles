"""Deterministic public network and enrichment artifacts."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from . import attention, corpus, indices, issues, rhetoric, trends
from .llm_annotations import load_paragraph_annotations, load_speech_annotations

OUT_DIR = corpus.DATA_DIR / "networks"
DEFAULT_MIN_PARAGRAPHS = 20
DEFAULT_MIN_SPEECHES = 5
LAYOUT_SEED = 20260722


def assign_era(year: int) -> str:
    for name, start, end in trends.ERAS:
        if start <= int(year) <= end:
            return name
    raise ValueError(f"year {year} falls outside the declared era axis")


def topic_edges(
    assignments: pd.DataFrame,
    speeches: pd.DataFrame,
    n_paragraphs_total: int | None = None,
) -> pd.DataFrame:
    """Co-occurrence pairs with paragraph support, speech support, Jaccard and lift."""
    required = {"doc_name", "para_idx", "topic"}
    if not required.issubset(assignments):
        raise ValueError(f"assignments missing {sorted(required - set(assignments))}")
    unique = assignments.drop_duplicates(["doc_name", "para_idx", "topic"])
    n_paragraphs = (
        int(n_paragraphs_total)
        if n_paragraphs_total is not None
        else unique[["doc_name", "para_idx"]].drop_duplicates().shape[0]
    )
    marginal = unique.groupby("topic").size()
    pair_rows: list[tuple[str, str, str, int]] = []
    for (doc, para), group in unique.groupby(["doc_name", "para_idx"], sort=False):
        topics = sorted(group.topic.unique())
        pair_rows.extend((a, b, doc, int(para)) for a, b in itertools.combinations(topics, 2))
    pairs = pd.DataFrame(pair_rows, columns=["source", "target", "doc_name", "para_idx"])
    columns = ["source", "target", "shared_paragraphs", "shared_speeches",
               "jaccard", "lift", "source_marginal_rate", "target_marginal_rate"]
    if pairs.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (source, target), group in pairs.groupby(["source", "target"], sort=True):
        shared = len(group)
        union = int(marginal[source] + marginal[target] - shared)
        source_rate, target_rate = marginal[source] / n_paragraphs, marginal[target] / n_paragraphs
        rows.append({
            "source": source, "target": target,
            "shared_paragraphs": shared,
            "shared_speeches": int(group.doc_name.nunique()),
            "jaccard": shared / union if union else 0.0,
            "lift": (shared / n_paragraphs) / (source_rate * target_rate),
            "source_marginal_rate": source_rate,
            "target_marginal_rate": target_rate,
        })
    return pd.DataFrame(rows, columns=columns)


def stable_layout(edges: pd.DataFrame, seed: int = LAYOUT_SEED,
                  nodes: list[str] | None = None) -> dict[str, list[float]]:
    graph = nx.Graph()
    if nodes:
        graph.add_nodes_from(nodes)
    if len(edges):
        graph.add_weighted_edges_from(
            [(r.source, r.target, float(r.shared_speeches))
             for r in edges.itertuples(index=False)]
        )
    positions = nx.spring_layout(graph, seed=seed, weight="weight")
    return {node: [round(float(x), 6), round(float(y), 6)]
            for node, (x, y) in sorted(positions.items())}


def _assignments() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    labels = load_paragraph_annotations("paragraph_annotations")
    taxonomy = attention.load_taxonomy()
    canonical = attention.canonical_label_map(taxonomy)
    parent = {entry["name"]: entry["level1"] for entry in taxonomy["level2"]}
    rows = []
    for row in labels.itertuples(index=False):
        for topic in attention.normalize_topics(row.topics, canonical):
            rows.append((row.doc_name, int(row.para_idx), topic, parent[topic]))
    universe = labels[["doc_name", "para_idx"]].drop_duplicates()
    return (
        pd.DataFrame(rows, columns=["doc_name", "para_idx", "level2", "level1"]),
        taxonomy,
        universe,
    )


def build_topic_networks(
    assignments: pd.DataFrame, meta: pd.DataFrame, universe: pd.DataFrame
) -> pd.DataFrame:
    joined = assignments.merge(meta, on="doc_name", validate="many_to_one")
    universe_meta = universe.merge(
        meta[["doc_name", "era", "speech_type"]],
        on="doc_name", validate="many_to_one",
    )
    output = []
    for level in ("level2", "level1"):
        base = joined.rename(columns={level: "topic"}).drop_duplicates(
            ["doc_name", "para_idx", "topic", "era", "speech_type"])
        slices = [("all", "all", base)]
        slices += [(era, "all", group) for era, group in base.groupby("era")]
        slices += [("all", speech_type, group) for speech_type, group in base.groupby("speech_type")]
        slices += [
            (era, speech_type, group)
            for (era, speech_type), group in base.groupby(["era", "speech_type"])
        ]
        for era, speech_type, frame in slices:
            universe_slice = universe_meta
            if era != "all":
                universe_slice = universe_slice[universe_slice.era.eq(era)]
            if speech_type != "all":
                universe_slice = universe_slice[
                    universe_slice.speech_type.eq(speech_type)
                ]
            edges = topic_edges(
                frame, meta,
                n_paragraphs_total=len(universe_slice),
            )
            if edges.empty:
                continue
            edges.insert(0, "taxonomy_level", level)
            edges.insert(1, "era", era)
            edges.insert(2, "speech_type", speech_type)
            output.append(edges)
    return pd.concat(output, ignore_index=True)


def speech_type_enrichment(assignments: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    joined = assignments.merge(meta[["doc_name", "speech_type"]], on="doc_name",
                               validate="many_to_one")
    denominators = meta.groupby("speech_type").doc_name.nunique()
    all_den = meta.doc_name.nunique()
    rows = []
    for level in ("level2", "level1"):
        pairs = joined[["doc_name", "speech_type", level]].drop_duplicates()
        overall = pairs.groupby(level).doc_name.nunique() / all_den
        by_type = pairs.groupby(["speech_type", level]).doc_name.nunique()
        for (speech_type, topic), count in by_type.items():
            rate = count / denominators[speech_type]
            rows.append({"taxonomy_level": level, "speech_type": speech_type,
                         "topic": topic, "n_speeches_with_topic": int(count),
                         "n_speeches_in_type": int(denominators[speech_type]),
                         "topic_rate": rate, "overall_rate": overall[topic],
                         "lift": rate / overall[topic] if overall[topic] else np.nan})
    return pd.DataFrame(rows)


def _cosine_edges(names: list[str], matrix: np.ndarray, kind: str,
                  counts: pd.Series, *, standardized: bool = False) -> pd.DataFrame:
    if standardized:
        eligible = np.array([counts.get(name, 0) >= 5 for name in names])
        reference = matrix[eligible]
        center = reference.mean(axis=0)
        spread = reference.std(axis=0)
        spread[spread == 0] = 1
        matrix = (matrix - center) / spread
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    values = matrix / np.where(norms == 0, 1, norms)
    similarity = values @ values.T
    rows = []
    for i, j in itertools.combinations(range(len(names)), 2):
        rows.append({"network": kind, "source": names[i], "target": names[j],
                     "similarity": float(similarity[i, j]),
                     "source_n_speeches": int(counts.get(names[i], 0)),
                     "target_n_speeches": int(counts.get(names[j], 0)),
                     "default_visible": bool(counts.get(names[i], 0) >= 5 and
                                             counts.get(names[j], 0) >= 5)})
    return pd.DataFrame(rows)


def president_networks(assignments: pd.DataFrame, speeches: pd.DataFrame) -> pd.DataFrame:
    counts = speeches.groupby("president").doc_name.nunique()
    joined = assignments.merge(speeches[["doc_name", "president"]], on="doc_name",
                               validate="many_to_one")
    fine = (joined.drop_duplicates(["doc_name", "para_idx", "level2"])
            .groupby(["president", "level2"]).size().unstack(fill_value=0))
    broad = (joined.drop_duplicates(["doc_name", "para_idx", "level1"])
             .groupby(["president", "level1"]).size().unstack(fill_value=0))
    fine_edges = _cosine_edges(
        fine.index.tolist(), fine.to_numpy(float), "ai_topics", counts
    )
    broad_edges = _cosine_edges(
        broad.index.tolist(), broad.to_numpy(float), "ai_domains", counts
    )

    annotations = load_paragraph_annotations("paragraph_annotations").merge(
        speeches[["doc_name", "president"]], on="doc_name",
        how="inner", validate="many_to_one",
    )
    flags = annotations.groupby("president")[
        ["party_attack", "enemy_naming", "zero_sum"]
    ].mean() * 100
    proposal_values = (
        pd.crosstab(
            annotations.president,
            annotations.proposal_values,
            normalize="index",
        ).reindex(columns=["proposal", "values"], fill_value=0) * 100
    )
    fine_probability = fine.div(fine.sum(axis=1), axis=0)
    breadth = fine_probability.apply(
        lambda row: float(np.exp(-(row[row > 0] * np.log(row[row > 0])).sum())),
        axis=1,
    ).rename("topic_breadth")
    ai_rhetoric = (
        flags.join(proposal_values, how="outer")
        .join(breadth, how="outer")
        .reindex(columns=[
            "party_attack", "enemy_naming", "zero_sum",
            "proposal", "values", "topic_breadth",
        ]).fillna(0)
    )
    ai_rhetoric_edges = _cosine_edges(
        ai_rhetoric.index.tolist(), ai_rhetoric.to_numpy(float),
        "ai_rhetoric", counts, standardized=True,
    )

    stats = rhetoric.build_stats(speeches)
    markers = indices.build_markers(speeches)
    scores = indices.president_scores(markers, stats, speeches).set_index("president")
    fingerprint_columns = [
        "nrc_hope", "nrc_fear", "certainty", "us_them",
        "self_reference", "fk_grade", "ttr", "religiosity",
    ]
    fingerprint_edges = _cosine_edges(
        scores.index.tolist(), scores[fingerprint_columns].to_numpy(float),
        "rhetorical_fingerprint", counts, standardized=True,
    )

    issue_table, _ = issues.build_issues()
    issue_table = issue_table.set_index("president")
    issue_columns = [column for column in issue_table if column.startswith("share_")]
    legacy_edges = _cosine_edges(
        issue_table.index.tolist(), issue_table[issue_columns].to_numpy(float),
        "legacy_issues", counts,
    )
    return pd.concat([
        fine_edges, broad_edges, ai_rhetoric_edges,
        fingerprint_edges, legacy_edges,
    ], ignore_index=True)


def _write_dictionary(path: Path, columns: dict[str, str]) -> None:
    pd.DataFrame([{"column": key, "definition": value}
                  for key, value in columns.items()]).to_csv(path, index=False)


def build(out_dir: Path = OUT_DIR) -> pd.DataFrame:
    speeches = corpus.load()
    speech_labels = load_speech_annotations("speech_annotations")
    meta = speeches[["doc_name", "president", "year"]].merge(
        speech_labels[["doc_name", "speech_type"]], on="doc_name", validate="one_to_one")
    meta["era"] = meta.year.map(assign_era)
    assignments, taxonomy, universe = _assignments()
    topic_table = build_topic_networks(assignments, meta, universe)
    enrichment = speech_type_enrichment(assignments, meta)
    presidents = president_networks(assignments, speeches)
    out_dir.mkdir(parents=True, exist_ok=True)
    topic_table.to_parquet(out_dir / "topic_edges.parquet", index=False)
    topic_table.to_csv(out_dir / "topic_edges.csv", index=False)
    enrichment.to_parquet(out_dir / "speech_type_enrichment.parquet", index=False)
    enrichment.to_csv(out_dir / "speech_type_enrichment.csv", index=False)
    presidents.to_parquet(out_dir / "president_edges.parquet", index=False)
    presidents.to_csv(out_dir / "president_edges.csv", index=False)
    invocation_path = corpus.DATA_DIR / "invocations_v2" / "edges.parquet"
    invocation_edges = pd.read_parquet(invocation_path) if invocation_path.exists() else pd.DataFrame()
    if len(invocation_edges):
        invocation_edges.to_csv(out_dir / "invocation_edges.csv", index=False)
    candidate_path = corpus.DATA_DIR / "invocations_v2" / "candidates.parquet"
    classification_path = corpus.DATA_DIR / "invocations_v2" / "classifications.parquet"
    invocation_evidence = pd.DataFrame()
    if candidate_path.exists() and classification_path.exists():
        candidates = pd.read_parquet(candidate_path)
        classifications = pd.read_parquet(classification_path)
        invocation_evidence = candidates.merge(
            classifications, on="candidate_id", validate="one_to_one"
        )
        invocation_evidence = invocation_evidence[
            invocation_evidence.excluded_reason.eq("")
        ].copy()
        invocation_evidence["era"] = pd.to_datetime(
            invocation_evidence.speech_date
        ).dt.year.map(assign_era)
        evidence_columns = [
            "candidate_id", "speaker", "target", "doc_name", "para_idx",
            "speech_date", "era", "raw_mention", "function", "stance",
            "evidence_span", "rationale", "quotation_status", "target_status",
            "evidence_status",
        ]
        invocation_evidence = invocation_evidence[evidence_columns]
        invocation_evidence.to_parquet(
            out_dir / "invocation_evidence.parquet", index=False
        )
        invocation_evidence.to_csv(
            out_dir / "invocation_evidence.csv", index=False
        )
        (out_dir / "invocation_evidence.json").write_text(
            invocation_evidence.to_json(orient="records", date_format="iso")
        )

    overall = topic_table[(topic_table.era == "all") &
                          (topic_table.speech_type == "all")]
    layouts = {}
    for level, entries in overall.groupby("taxonomy_level"):
        names = [entry["name"] for entry in taxonomy[level.replace("level", "level")]]
        layouts[level] = stable_layout(entries, nodes=names)
    default_edges = topic_table[(topic_table.shared_paragraphs >= DEFAULT_MIN_PARAGRAPHS) &
                                (topic_table.shared_speeches >= DEFAULT_MIN_SPEECHES)]
    invocation_layout = {}
    if len(invocation_edges):
        igraph = nx.Graph()
        for row in invocation_edges.itertuples(index=False):
            igraph.add_edge(row.speaker, row.target, weight=float(row.raw_mentions))
        invocation_layout = {node: [round(float(x), 6), round(float(y), 6)]
                             for node, (x, y) in nx.spring_layout(
                                 igraph, seed=LAYOUT_SEED, weight="weight").items()}
    president_layouts = {}
    for network, frame in presidents[presidents.default_visible].groupby("network"):
        graph = nx.Graph()
        strongest = frame.nlargest(140, "similarity")
        for row in strongest.itertuples(index=False):
            graph.add_edge(row.source, row.target, weight=float(row.similarity))
        president_layouts[network] = {
            node: [round(float(x), 6), round(float(y), 6)]
            for node, (x, y) in nx.spring_layout(
                graph, seed=LAYOUT_SEED, weight="weight").items()}
    payload = {
        "schema_version": "network-atlas-v2",
        "defaults": {"min_paragraphs": DEFAULT_MIN_PARAGRAPHS,
                     "min_speeches": DEFAULT_MIN_SPEECHES},
        "topic_edges": topic_table.to_dict("records"),
        "topic_layouts": layouts,
        "president_edges": presidents.to_dict("records"),
        "president_layouts": president_layouts,
        "invocation_edges": invocation_edges.to_dict("records"),
        "invocation_layout": invocation_layout,
    }
    (out_dir / "network_atlas.json").write_text(json.dumps(payload, indent=2))
    # Compatibility shard retained for the first site implementation.
    (out_dir / "topic_network.json").write_text(json.dumps(
        {"edges": default_edges.to_dict("records"), "layout": layouts}, indent=2))
    _write_dictionary(out_dir / "topic_edges_dictionary.csv", {
        "shared_paragraphs": "Distinct paragraphs carrying both topics.",
        "shared_speeches": "Distinct speeches containing a shared paragraph.",
        "jaccard": "Shared paragraphs divided by paragraphs carrying either topic.",
        "lift": "Observed co-occurrence divided by independence expectation.",
    })
    _write_dictionary(out_dir / "invocation_evidence_dictionary.csv", {
        "candidate_id": "Stable semantic identifier for the extracted mention.",
        "function": "Exploratory AI classification of what the invocation does.",
        "stance": "Exploratory AI classification of the speaker's stance.",
        "evidence_span": "Verbatim evidence used for the classification.",
        "quotation_status": "Whether the mention occurs in narration or a deliberate quotation.",
        "target_status": "The target president's chronological status on the speech date.",
        "evidence_status": "Validation status of the classification.",
    })
    _write_dictionary(out_dir / "president_edges_dictionary.csv", {
        "network": (
            "Declared feature space: ai_topics (50), ai_domains (17), "
            "ai_rhetoric (6 standardized), rhetorical_fingerprint "
            "(8 standardized), or legacy_issues (16)."
        ),
        "similarity": (
            "Cosine similarity within that feature space; it is not an overall "
            "president likeness or evidence of influence."
        ),
        "source_n_speeches": "Distinct corpus speeches for the source president.",
        "target_n_speeches": "Distinct corpus speeches for the target president.",
        "default_visible": "Both presidents have at least five corpus speeches.",
    })
    (out_dir / "manifest.json").write_text(json.dumps({
        "schema_version": "network-atlas-v2", "layout_seed": LAYOUT_SEED,
        "default_support": {"paragraphs": DEFAULT_MIN_PARAGRAPHS,
                            "speeches": DEFAULT_MIN_SPEECHES},
        "taxonomy": "taxonomy_v1", "era_axis": [x[0] for x in trends.ERAS],
        "president_similarity_spaces": {
            "ai_topics": 50,
            "ai_domains": 17,
            "ai_rhetoric": 6,
            "rhetorical_fingerprint": 8,
            "legacy_issues": 16,
        },
    }, indent=2))
    return topic_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic network artifacts")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    edges = build(args.out_dir)
    print(f"wrote {len(edges)} topic-network slice edges to {args.out_dir}")


if __name__ == "__main__":
    main()
