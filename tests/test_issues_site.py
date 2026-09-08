"""End-to-end contracts for the Issues Page V2 evidence model and renderers.

These tests deliberately use the checked-in corpus artifacts.  The page model is
the join point for the directory, detail pages, JSON shards, figures, and CSVs;
testing those surfaces from one module-scoped build keeps parity assertions both
realistic and reasonably fast.
"""

from __future__ import annotations

import base64
import json
import re

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import ai_labels, corpus, issues, profiles
from presidential_profiles.issues_site import (
    ISSUES_CSS,
    ISSUES_JS,
    IssuePageViewModel,
    _prune_stale_issue_outputs,
    build_issue_view_models,
    fine_topic_export_frame,
    issue_view_payload,
    president_period_export_frame,
    render_issue,
    render_issue_index,
    trend_export_frame,
)


EXPECTED_SOURCE_ISSUES = [
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
    "Discovered 5",
]

EXPECTED_LABELS = EXPECTED_SOURCE_ISSUES[:-1] + ["Security & peace"]

EXPECTED_SLUGS = [
    "economy-and-jobs",
    "taxes-and-budget",
    "war-and-military",
    "foreign-policy",
    "immigration",
    "civil-rights-and-race",
    "health-care",
    "education",
    "crime-and-justice",
    "energy-and-environment",
    "trade-and-tariffs",
    "agriculture",
    "religion-and-values",
    "money-and-banking",
    "infrastructure",
    "security-and-peace",
]

EXPECTED_SUPPORTED_STARTS = {
    "Economy & jobs": 1930,
    "Taxes & budget": 1830,
    "War & military": 1940,
    "Foreign policy": 1845,
    "Immigration": 2015,
    "Civil rights & race": 1850,
    "Health care": 2000,
    "Education": 1995,
    "Crime & justice": 1995,
    "Energy & environment": 1975,
    "Trade & tariffs": 1910,
    "Agriculture": 1930,
    "Religion & values": 1985,
    "Money & banking": 1835,
    "Infrastructure": 1805,
    "Security & peace": 1945,
}

EXPECTED_FINGERPRINT = {
    "n_speeches": 1057,
    "n_paragraphs": 36229,
    "doc_name_sha256": (
        "23eff144ab94f3fa5cedefcacf7f21bc5c068518fe3cb2946c2a2a179fdb9bcc"
    ),
}


@pytest.fixture(scope="module")
def real_issue_corpus() -> dict:
    """Load the frozen inputs once and build the production V2 route set."""
    issue_df = pd.read_parquet(issues.ISSUES_PRESIDENT_PATH)
    issue_meta = json.loads(issues.ISSUES_META_PATH.read_text())
    paragraphs = pd.read_parquet(corpus.PARAGRAPHS_PATH)
    paragraph_labels = pd.read_parquet(issues.PARA_LABELS_PATH)
    speeches = pd.read_parquet(corpus.PARQUET_PATH)
    ai_data = ai_labels.build_ai_data()

    views = build_issue_view_models(
        issue_df,
        issue_meta,
        pd.DataFrame(),
        ai_data=ai_data,
        paragraphs=paragraphs,
        labels=paragraph_labels,
        speeches=speeches,
    )
    keyed_paragraphs = paragraphs.merge(
        paragraph_labels,
        on=["doc_name", "para_idx"],
        how="inner",
        validate="one_to_one",
    ).set_index(["doc_name", "para_idx"])
    speech_lookup = speeches.set_index("doc_name")
    assert keyed_paragraphs.index.is_unique
    assert speech_lookup.index.is_unique
    return {
        "views": views,
        "keyed_paragraphs": keyed_paragraphs,
        "speech_lookup": speech_lookup,
        "issue_df": issue_df,
        "issue_meta": issue_meta,
        "paragraphs": paragraphs,
        "paragraph_labels": paragraph_labels,
        "speeches": speeches,
        "ai_data": ai_data,
    }


@pytest.fixture(scope="module")
def real_views(real_issue_corpus) -> tuple[IssuePageViewModel, ...]:
    return real_issue_corpus["views"]


@pytest.fixture(scope="module")
def real_payloads(real_views) -> dict[str, dict]:
    return {view.slug: issue_view_payload(view) for view in real_views}


def _view(real_views, label: str) -> IssuePageViewModel:
    return next(view for view in real_views if view.label == label)


def _export_matches_rows(
    frame: pd.DataFrame,
    rows: tuple[dict, ...],
    *,
    source_issue: str,
) -> None:
    """Compare evidence fields while checking export-only context columns."""
    row_frame = pd.DataFrame(rows)
    shared = [column for column in frame if column in row_frame]
    pd.testing.assert_frame_equal(
        frame[shared].reset_index(drop=True),
        row_frame.reindex(columns=shared),
        check_dtype=False,
    )
    extra = set(frame) - set(row_frame)
    assert extra <= {"source_issue", "unit"}
    assert "unit" in extra
    assert set(frame["unit"]) == {"share of eligible paragraphs"}
    if "source_issue" in extra:
        assert set(frame["source_issue"]) == {source_issue}


def _plotly_values(value) -> list:
    """Decode Plotly 6 typed-array JSON while also accepting ordinary lists."""
    if isinstance(value, dict) and {"dtype", "bdata"}.issubset(value):
        raw = base64.b64decode(value["bdata"])
        return np.frombuffer(raw, dtype=np.dtype(value["dtype"])).tolist()
    if value is None:
        return []
    return list(value)


def test_catalog_order_unique_slugs_and_source_split(real_views):
    assert len(real_views) == 16
    assert all(isinstance(view, IssuePageViewModel) for view in real_views)
    assert [view.order for view in real_views] == list(range(16))
    assert [view.source_issue for view in real_views] == EXPECTED_SOURCE_ISSUES
    assert [view.label for view in real_views] == EXPECTED_LABELS
    assert [view.slug for view in real_views] == EXPECTED_SLUGS
    assert len({view.slug for view in real_views}) == 16

    anchored = [view for view in real_views if view.source_kind == "anchored_corex"]
    surfaced = [
        view
        for view in real_views
        if view.source_kind == "surfaced_discovered_corex"
    ]
    assert len(anchored) == 15
    assert [(view.source_issue, view.label) for view in surfaced] == [
        ("Discovered 5", "Security & peace")
    ]
    assert all(view.source_badge == "Anchored CorEx axis" for view in anchored)
    assert surfaced[0].source_badge == "Surfaced CorEx theme"

    assert real_views[0].previous_issue is None
    assert real_views[-1].next_issue is None
    for previous, current in zip(real_views, real_views[1:]):
        assert previous.next_issue == {"slug": current.slug, "label": current.label}
        assert current.previous_issue == {
            "slug": previous.slug,
            "label": previous.label,
        }


def test_all_excerpts_have_unique_keyed_receipts_and_canonical_urls(
    real_issue_corpus,
):
    views = real_issue_corpus["views"]
    keyed = real_issue_corpus["keyed_paragraphs"]
    speeches = real_issue_corpus["speech_lookup"]
    all_receipts = []
    canonical_urls = []

    for view in views:
        assert [excerpt["window_role"] for excerpt in view.excerpts] == [
            "early",
            "highest_supported",
            "recent",
        ]
        receipts = [
            (excerpt["doc_name"], int(excerpt["para_idx"]))
            for excerpt in view.excerpts
        ]
        assert len(receipts) == len(set(receipts)) == 3

        for excerpt in view.excerpts:
            receipt = (excerpt["doc_name"], int(excerpt["para_idx"]))
            source = keyed.loc[receipt]
            speech = speeches.loc[excerpt["doc_name"]]
            assert bool(source[view.source_issue])
            assert excerpt["quote"] in source["text"]
            assert excerpt["president"] == speech["president"]
            assert excerpt["title"] == speech["title"]
            assert excerpt["year"] == int(speech["year"])

            expected_url = profiles.miller_speech_url(excerpt["doc_name"])
            assert excerpt["url"] == expected_url
            assert excerpt["url"].startswith(
                profiles.MILLER_ORIGIN + profiles.MILLER_SPEECH_PATH
            )
            assert excerpt["url"].count(profiles.MILLER_SPEECH_PATH) == 1
            all_receipts.append((view.source_issue, *receipt))
            canonical_urls.append(excerpt["url"])

    assert len(all_receipts) == len(set(all_receipts)) == 48
    assert len(canonical_urls) == 48


def test_highest_supported_period_is_shared_with_summary_and_excerpt(real_views):
    assert set(EXPECTED_SUPPORTED_STARTS) == {view.label for view in real_views}

    for view in real_views:
        supported = [
            row
            for row in view.trend_rows
            if row["ci_status"] == "ok" and row["point"] is not None
        ]
        expected = min(
            supported,
            key=lambda row: (-float(row["point"]), int(row["period_order"])),
        )
        actual = view.highest_supported_period
        assert actual["period_start"] == EXPECTED_SUPPORTED_STARTS[view.label]
        assert {
            key: actual[key]
            for key in ("period_start", "period_end", "point", "ci_status")
        } == {
            key: expected[key]
            for key in ("period_start", "period_end", "point", "ci_status")
        }
        supported_excerpt = next(
            excerpt
            for excerpt in view.excerpts
            if excerpt["window_role"] == "highest_supported"
        )
        assert (
            supported_excerpt["window_start"],
            supported_excerpt["window_end"],
        ) == (actual["period_start"], actual["period_end"])

    religion = _view(real_views, "Religion & values")
    control = next(
        row for row in religion.trend_rows if row["period_start"] == 1785
    )
    assert religion.highest_supported_period["period_start"] == 1985
    assert control["point"] == pytest.approx(1 / 7)
    assert control["lo"] == pytest.approx(0)
    assert control["hi"] == pytest.approx(2 / 3)
    assert control["interval_unresolvable"] is False


def test_broad_payload_exports_and_figure_share_one_1785_contract(
    real_views,
    real_payloads,
):
    unresolved_1785 = 0

    for view in real_views:
        payload = real_payloads[view.slug]
        assert payload["schema"] == "issue-page-v2"
        assert payload["source_issue"] == view.source_issue
        assert payload["label"] == view.label
        assert payload["slug"] == view.slug
        assert payload["unit"] == "share of eligible paragraphs"
        assert payload["highest_supported_period"] == view.highest_supported_period
        assert payload["trend_rows"] == list(view.trend_rows)
        assert payload["president_period_rows"] == list(view.president_period_rows)

        trend_export = trend_export_frame(view)
        president_export = president_period_export_frame(view)
        assert {"share_percent", "lo_sampling_percent", "hi_sampling_percent",
                "lo_percent", "hi_percent", "n_paired_paragraphs",
                "disagreement_band_applied", "disagreement_half_width_percent"} <= set(
                    trend_export.columns
                )
        assert not {"point", "lo_sampling", "hi_sampling", "lo", "hi"}.intersection(
            trend_export.columns
        )
        _export_matches_rows(
            trend_export,
            view.trend_rows,
            source_issue=view.source_issue,
        )
        _export_matches_rows(
            president_export,
            view.president_period_rows,
            source_issue=view.source_issue,
        )
        assert len(trend_export) == 49

        first = trend_export.iloc[0]
        assert (int(first["period_start"]), int(first["period_end"])) == (1785, 1789)
        assert int(first["n_paragraphs"]) == 14
        assert int(first["n_speeches"]) == 2
        assert first["ci_status"] == "low_cluster_caution"
        unresolved_1785 += int(bool(first["interval_unresolvable"]))

        figure = payload["trend_figure"]
        assert figure["layout"]["xaxis"]["range"] == [1785, 2029]
        expected_x = [float(row["x"]) for row in view.trend_rows]
        line_traces = [
            trace
            for trace in figure["data"]
            if trace.get("mode") == "lines"
            and trace.get("fill") is None
            and len(_plotly_values(trace.get("x"))) == len(view.trend_rows)
        ]
        assert len(line_traces) == 2
        assert all(_plotly_values(trace["x"]) == expected_x for trace in line_traces)

        for index, row in enumerate(view.trend_rows):
            plotted = []
            for trace in line_traces:
                value = _plotly_values(trace["y"])[index]
                if value is not None and np.isfinite(float(value)):
                    plotted.append(float(value))
            # Segment endpoints may appear in both the supported and caution
            # traces so the visual transition joins without shifting the point.
            assert plotted
            assert plotted == pytest.approx(
                [float(row["point"]) * 100] * len(plotted)
            )

        unresolved_rows = [
            row for row in view.trend_rows if row["interval_unresolvable"]
        ]
        ring_traces = [
            trace
            for trace in figure["data"]
            if trace.get("marker", {}).get("symbol") == "circle-open"
        ]
        if unresolved_rows:
            assert len(ring_traces) == 1
            assert _plotly_values(ring_traces[0]["x"]) == [
                float(row["x"]) for row in unresolved_rows
            ]
            assert _plotly_values(ring_traces[0]["y"]) == pytest.approx(
                [float(row["point"]) * 100 for row in unresolved_rows]
            )
        else:
            assert ring_traces == []

    assert unresolved_1785 == 11


def test_foreign_owner_threshold_ranking_and_absolute_render_width(real_views):
    foreign = _view(real_views, "Foreign policy")
    owner_names = [owner["president"] for owner in foreign.owners]
    assert len(owner_names) == 8
    assert owner_names[0] == "James K. Polk"
    assert owner_names[-1] == "Ulysses S. Grant"
    assert "Zachary Taylor" not in owner_names
    assert all(owner["n_speeches"] >= 5 for owner in foreign.owners)
    assert [owner["rank"] for owner in foreign.owners] == list(range(1, 9))

    polk = foreign.owners[0]
    assert (polk["numerator"], polk["denominator"], polk["n_speeches"]) == (
        263,
        722,
        25,
    )
    assert polk["share_percent"] == pytest.approx(263 / 722 * 100)

    page = render_issue(
        foreign,
        view_data_url="../data/issues/foreign-policy.json",
    )
    assert "Bars use an absolute 0–100% scale" in page
    for owner in foreign.owners:
        assert f'style="width:{owner["share_percent"]:.6f}%"' in page
    assert 'style="width:100.000000%"' not in page


def test_initial_topic_counts_stable_colors_and_auto_clear_contract(
    real_views,
    real_payloads,
):
    expected_counts = {
        "Health care": 1,
        "Taxes & budget": 2,
        "Education": 3,
        "Foreign policy": 8,
    }
    assert {
        label: len(_view(real_views, label).initial_topic_names)
        for label in expected_counts
    } == expected_counts

    colors_by_topic = {}
    for view in real_views:
        payload = real_payloads[view.slug]
        assert payload["initial_topic_names"] == list(view.initial_topic_names)
        assert set(payload["fine_topic_colors"]) == set(view.initial_topic_names)
        for topic in view.fine_topics:
            assert re.fullmatch(r"#[0-9a-f]{6}", topic["color"])
            assert payload["fine_topic_colors"][topic["name"]] == topic["color"]
            prior = colors_by_topic.setdefault(topic["name"], topic["color"])
            assert prior == topic["color"]

    # Auto stays readable for 1/2/3-topic pages and switches the eight-topic
    # Foreign policy page to a heatmap.  Color lookup is keyed by topic name,
    # so changing checkbox order cannot recolor a series.
    assert 'count <= 3 && count * periodCount <= 48 ? "bars" : "heatmap"' in ISSUES_JS
    assert "Auto → ${modeLabel(resolved)}" in ISSUES_JS
    assert "marker: {color: colors[topic]" in ISSUES_JS
    assert "Plotly.purge(fineTarget)" in ISSUES_JS
    assert "fineTarget.hidden = true" in ISSUES_JS
    assert "plot removed." in ISSUES_JS
    assert "IntersectionObserver" in ISSUES_JS
    assert "overflow-x:auto" in ISSUES_CSS
    assert ".chart-scroll>.chart{min-width:700px}" in ISSUES_CSS
    assert 'note.setAttribute("role", "status")' in ISSUES_JS
    assert 'target?.querySelector(".chart-loading")?.remove()' in ISSUES_JS
    assert ISSUES_JS.index("removeLoadingPlaceholder(broadTarget)") < ISSUES_JS.index(
        "Plotly.newPlot("
    )
    assert ISSUES_JS.index("removeLoadingPlaceholder(fineTarget)") < ISSUES_JS.index(
        "Plotly.react("
    )


def test_fine_exports_keep_partial_observed_edges_and_full_denominators(real_views):
    health = _view(real_views, "Health care")
    export = fine_topic_export_frame(health)
    _export_matches_rows(
        export,
        health.fine_topic_rows,
        source_issue=health.source_issue,
    )
    assert set(export["interval_status"]) == {"not_estimated"}
    assert {"support_status", "partial_period", "numerator", "denominator"} <= set(export)

    expected_edges = {
        (20, 1780): (1780, 1799, 1789, 1799, 355, True, "descriptive"),
        (20, 2020): (2020, 2039, 2020, 2026, 3096, True, "descriptive"),
        (10, 1780): (1780, 1789, 1789, 1789, 14, True, "low_support"),
        (10, 1950): (1950, 1959, 1951, 1959, 750, True, "descriptive"),
        (10, 2020): (2020, 2029, 2020, 2026, 3096, True, "descriptive"),
    }
    for (bucket, requested_start), expected in expected_edges.items():
        row = export[
            (export["bucket_years"] == bucket)
            & (export["requested_start"] == requested_start)
        ].iloc[0]
        observed = (
            int(row["requested_start"]),
            int(row["requested_end"]),
            int(row["observed_start"]),
            int(row["observed_end"]),
            int(row["denominator"]),
            bool(row["partial_period"]),
            row["support_status"],
        )
        assert observed == expected

    # Every topic uses the full eligible-paragraph denominator for its period,
    # including the eight-way Foreign-policy crosswalk.
    for view in real_views:
        rows = pd.DataFrame(view.fine_topic_rows)
        grouped = rows.groupby(["bucket_years", "requested_start"], sort=False)
        assert all(group["denominator"].nunique() == 1 for _, group in grouped)
        assert all(
            row.share_percent == pytest.approx(row.numerator / row.denominator * 100)
            for row in rows.itertuples()
        )


def test_index_and_detail_have_semantic_accessible_fallbacks(real_views):
    index = render_issue_index(list(real_views))
    assert '<html lang="en">' in index
    assert '<a class="skip-link" href="#main-content">' in index
    assert '<nav class="breadcrumbs" aria-label="Breadcrumb">' in index
    assert '<main class="directory-main" id="main-content">' in index
    assert '<ol class="issue-directory">' in index
    assert index.count('<article class="directory-card">') == 16
    assert index.count('class="source-badge anchored"') == 15
    assert index.count('class="source-badge surfaced"') == 1
    assert '<dd>1 finer AI topic</dd>' in index
    assert '<dd>8 finer AI topics</dd>' in index
    assert index.count("<h1>") == 1
    assert "<footer>" in index
    assert 'href="../feedback.html"' in index

    detail = render_issue(
        _view(real_views, "Foreign policy"),
        view_data_url="../data/issues/foreign-policy.json",
    )
    assert '<html lang="en">' in detail
    assert '<a class="skip-link" href="#main-content">' in detail
    assert '<nav class="breadcrumbs" aria-label="Breadcrumb">' in detail
    assert '<nav class="issue-local-nav" aria-label="On this issue">' in detail
    assert '<span class="issue-nav-cue" aria-hidden="true">More sections →</span>' in detail
    assert ".issue-local-nav a{min-height:44px" in ISSUES_CSS
    assert ".topic-controls label,.topic-picker summary,.exact-values summary,.download-links a{min-height:44px}" in ISSUES_CSS
    assert ".issue-fact span,.excerpt-label,.receipt,.fine-definition span,.fine-selects label," in ISSUES_CSS
    assert ".issue-adjacent span,.directory-card dt{font-size:12px!important}" in ISSUES_CSS
    assert ".receipt{overflow-wrap:anywhere}" in ISSUES_CSS
    assert '<main id="main-content">' in detail
    for section_id in (
        "broad-trend",
        "selected-excerpts",
        "president-emphasis",
        "fine-topics",
        "data-method",
    ):
        assert f'<section id="{section_id}">' in detail

    assert detail.count('<figure class="issue-figure') == 2
    assert detail.count('role="img" aria-labelledby=') == 2
    assert detail.count("<figcaption>") == 2
    assert detail.count("<noscript>") == 3
    assert detail.count('class="chart-scroll" tabindex="0" role="region"') == 2
    assert detail.count('class="table-scroll" tabindex="0" role="region"') == 3
    assert detail.count("<caption>") == 3
    assert '<fieldset class="topic-controls"' in detail
    assert '<legend>Fine topics to display</legend>' in detail
    assert '<select id="fine-topic-bucket" disabled>' in detail
    assert '<select id="fine-topic-mode" disabled>' in detail
    assert '<button type="button" id="fine-topic-all" disabled>' in detail
    assert '<button type="button" id="fine-topic-clear" disabled>' in detail
    assert "Support status" in detail
    assert "low support" in detail
    assert 'role="status" aria-live="polite" aria-atomic="true"' in detail
    assert detail.count('<blockquote class="issue-excerpt">') == 3
    assert detail.count('target="_blank" rel="noopener"') == 3
    assert '<nav class="issue-adjacent" aria-label="Adjacent issues">' in detail
    assert detail.count(" download>") == 3
    assert "<footer>" in detail
    assert 'href="../feedback.html"' in detail
    assert (
        "examples from the corpus, not claims that any speech caused the trend"
        in " ".join(detail.split())
    )


def test_every_surface_carries_the_active_corpus_fingerprint(
    real_views,
    real_payloads,
):
    for view in real_views:
        assert view.provenance["corpus_fingerprint"] == EXPECTED_FINGERPRINT
        assert (
            real_payloads[view.slug]["provenance"]["corpus_fingerprint"]
            == EXPECTED_FINGERPRINT
        )
        broad = view.provenance["broad_instrument"]
        fine = view.provenance["fine_instrument"]
        assert broad["ci_components"] == "sampling_only"
        assert broad["cluster_unit"] == "speech"
        assert fine["run_id"] == "taxonomy-v1-20260719"
        assert fine["crosswalk"]["check"]["n_pages_reconstructable"] == 16


def test_builder_rejects_foreign_assignments_and_false_active_fingerprints(
    real_issue_corpus,
):
    inputs = real_issue_corpus
    ai_data = dict(inputs["ai_data"])
    assignments = inputs["ai_data"]["topic_assignments"]
    foreign = assignments.iloc[[0]].copy()
    foreign.loc[:, "doc_name"] = "not-in-active-corpus"
    foreign.loc[:, "para_idx"] = 999999
    ai_data["topic_assignments"] = pd.concat(
        [assignments, foreign], ignore_index=True
    )
    with pytest.raises(RuntimeError, match="outside the paragraph corpus"):
        build_issue_view_models(
            inputs["issue_df"],
            inputs["issue_meta"],
            pd.DataFrame(),
            ai_data=ai_data,
            paragraphs=inputs["paragraphs"],
            labels=inputs["paragraph_labels"],
            speeches=inputs["speeches"],
        )

    incomplete_speeches = inputs["speeches"].iloc[:-1].copy()
    with pytest.raises(ValueError, match="active corpus.*bands fingerprint"):
        build_issue_view_models(
            inputs["issue_df"],
            inputs["issue_meta"],
            pd.DataFrame(),
            ai_data=inputs["ai_data"],
            paragraphs=inputs["paragraphs"],
            labels=inputs["paragraph_labels"],
            speeches=incomplete_speeches,
        )


def test_stale_route_pruning_respects_the_owned_manifest(tmp_path):
    issue_dir = tmp_path / "issues"
    data_dir = tmp_path / "data" / "issues"
    issue_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    (issue_dir / "stale.html").write_text(
        '<body data-generated="issues-v2">stale</body>'
    )
    (issue_dir / "protected.html").write_text("<body>user-owned sentinel</body>")
    (data_dir / "stale.json").write_text("{}")
    (data_dir / "stale.csv").write_text("old\n")
    (data_dir / "current.csv").write_text("legacy plot geometry\n")
    (data_dir / "protected.txt").write_text("user-owned sentinel")
    (data_dir / "manifest.json").write_text(json.dumps({
        "issue_pages": ["stale.html"],
        "data_files": ["stale.json", "stale.csv", "manifest.json"],
    }))

    _prune_stale_issue_outputs(
        issue_dir,
        data_dir,
        expected_pages={"current.html"},
        expected_data={"current.json", "manifest.json"},
        current_slugs={"current"},
    )

    assert not (issue_dir / "stale.html").exists()
    assert not (data_dir / "stale.json").exists()
    assert not (data_dir / "stale.csv").exists()
    assert not (data_dir / "current.csv").exists()
    assert (issue_dir / "protected.html").read_text() == "<body>user-owned sentinel</body>"
    assert (data_dir / "protected.txt").read_text() == "user-owned sentinel"
