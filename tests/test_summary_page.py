import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import combat, corpus, eras, indices, metrics, site, trends


DATA = Path(__file__).parents[1] / "data"


@pytest.fixture(scope="module")
def temporal_contract():
    return site.build_summary_temporal_president_contract(
        pd.read_parquet(DATA / "speech_markers.parquet"),
        corpus.load(),
    )


def _minimal_temporal_sources():
    speeches = pd.DataFrame({
        "doc_name": ["a", "b"],
        "president": ["Example President", "Example President"],
        "year": [1900, 1901],
        "date": pd.to_datetime(["1900-01-01", "1901-01-01"]),
    })
    markers = pd.DataFrame({
        "doc_name": ["a", "b"],
        "president": ["Example President", "Example President"],
        "year": [1900, 1901],
        "n_words": [100, 900],
        "future": [1, 9],
        "nostalgia": [0, 20],
    })
    return markers, speeches


def _synthetic_communication_data():
    audience = (
        ("Congress", 5, 50.0),
        ("General public", 3, 30.0),
        ("Specific organizations or groups", 1, 10.0),
        ("Other audiences", 1, 10.0),
    )
    medium = (
        ("Written message", 4, 40.0),
        ("Spoken address", 3, 30.0),
        ("Radio/TV broadcast", 2, 20.0),
        ("Press conference/debate or other performed form", 1, 10.0),
    )
    return {
        "communications": [
            {
                "short": short,
                "years": f"{1789 + index * 20}–{1808 + index * 20}",
                "total": 10,
                "audience": [
                    {"label": label, "count": count, "share": share}
                    for label, count, share in audience
                ],
                "medium": [
                    {"label": label, "count": count, "share": share}
                    for label, count, share in medium
                ],
            }
            for index, short in enumerate(site.SUMMARY_ERA_SHORT)
        ]
    }


def test_breadth_limitation_callout_is_removed():
    summary = (DATA.parent / "docs" / "summary.html").read_text()
    assert "Claim removed from the headline" not in summary
    assert "Topic breadth expands early, then largely plateaus" not in summary
    assert 'class="summary-limit"' not in summary


def test_voice_uses_two_separate_four_category_bar_charts():
    summary = (DATA.parent / "docs" / "summary.html").read_text()
    assert 'data-fig="summary_communication"' not in summary
    assert summary.count('data-fig="summary_audience"') == 1
    assert summary.count('data-fig="summary_medium"') == 1
    assert "summary-communication-who" in summary
    assert "summary-communication-how" in summary
    assert 'aria-label="WHO legend"' in summary
    assert 'aria-label="HOW legend"' in summary
    assert "the separate bars show a Congress-facing written" in summary
    assert "without treating audience or delivery as its cause" in summary
    assert "Audience, delivery, and register across the same eras" in summary
    assert "Every president remains inspectable" not in summary
    assert '<article class="communication-mosaic-card"' not in summary


def test_communication_bar_charts_keep_four_series_and_100_percent_eras():
    summary_data = _synthetic_communication_data()
    for figure in (
        site.fig_summary_audience(summary_data),
        site.fig_summary_medium(summary_data),
    ):
        assert len(figure.data) == 4
        assert all(trace.type == "bar" for trace in figure.data)
        assert all(len(trace.y) == 9 for trace in figure.data)
        totals = np.sum(
            [np.asarray(trace.y, dtype=float) for trace in figure.data], axis=0
        )
        assert np.allclose(totals, 100)
        assert figure.layout.barmode == "stack"
        assert figure.layout.showlegend is False
        assert figure.layout.hovermode == "closest"
        assert figure.layout.xaxis.showspikes is False
        assert figure.layout.yaxis.showspikes is False
        assert all(trace.customdata is None for trace in figure.data)
        assert all("customdata" not in trace.hovertemplate for trace in figure.data)


def test_post_civil_war_topic_breadth_change_is_small_beside_early_expansion():
    trends = pd.read_parquet(DATA / "register" / "trends.parquet")
    rows = site._summary_register_series(
        trends, "effective_topics", taxonomy="llm_level2"
    )
    assert tuple(rows.period.astype(int)) == site.SUMMARY_REGISTER_PERIODS
    early_gain = rows.iloc[3].value - rows.iloc[0].value
    post_civil_war_gain = rows.iloc[-1].value - rows.iloc[3].value
    assert early_gain > 4 * post_civil_war_gain


def test_hope_doom_ratio_suppresses_thin_and_zero_doom_windows():
    markers = pd.DataFrame({
        "year": [2000, 2001, 2002, 2003, 2004, 2010],
        "nrc_hope": [50, 50, 50, 50, 50, 100],
        "doom": [0, 0, 2, 2, 2, 0],
        "n_words": [5_000, 5_000, 5_000, 5_000, 5_000, 30_000],
    })
    figure = site.fig_summary_hope_doom_ratio(markers)
    ratio = np.asarray(figure.data[0].y, dtype=float)
    assert np.isnan(ratio[-1])
    assert np.isfinite(ratio).any()
    assert figure.layout.yaxis.type == "log"
    assert any(shape.type == "line" for shape in figure.layout.shapes)


def test_language_switcher_restores_exact_modals_and_pronoun_families():
    stats = pd.read_parquet(DATA / "speech_stats.parquet")
    markers = pd.read_parquet(DATA / "speech_markers.parquet")
    speeches = corpus.load()
    modals = site.fig_summary_modals(stats, markers, speeches)
    assert [trace.name for trace in modals.data] == [
        "Necessity", "Commitment / intent", "Absolute emphasis", "Conditional",
        "Advice / possibility",
    ]
    assert all("per 10,000 words" in trace.hovertemplate for trace in modals.data)
    assert all("<b>Presidents:</b>" in trace.hovertemplate for trace in modals.data)
    assert all(trace.hovertemplate.count("<br>") == 2 for trace in modals.data)
    assert all(trace.connectgaps is False for trace in modals.data)
    assert all(trace.mode == "lines+markers" for trace in modals.data)
    assert all(trace.line.dash == "solid" for trace in modals.data)
    assert len({trace.marker.symbol for trace in modals.data}) == len(modals.data)
    assert all(np.isfinite(np.asarray(trace.y, dtype=float)).all() for trace in modals.data)
    assert modals.layout.hoverlabel.font.size == 10
    assert modals.layout.yaxis.title.text == "uses per 10,000 words"
    assert "certainty index" not in modals.layout.title.text.lower()
    assert "seven-year window" in modals.layout.xaxis.title.text

    rates, context = site.summary_stance_family_rates(stats, markers, speeches)
    assert len(rates) == rates.notna().all(axis=1).sum() == 238
    assert len(context) == 238
    speech_counts = speeches.set_index("doc_name")[["transcript"]].copy()
    speech_counts["need_forms"] = speech_counts.transcript.str.count(
        site.SUMMARY_NECESSITY_NEED_RE
    )
    speech_counts["obligation_phrases"] = speech_counts.transcript.str.count(
        site.SUMMARY_NECESSITY_OBLIGATION_RE
    )
    joined = stats.set_index("doc_name").join(
        markers.set_index("doc_name")[["n_words"]], how="inner"
    ).join(speech_counts.drop(columns="transcript"), how="inner")
    first_window = joined[joined.year.between(1789, 1792)]
    expected_necessity = (
        (
            first_window.modal_must
            + first_window.need_forms
            + first_window.obligation_phrases
        ).sum()
        / first_window.n_words.sum() * 10_000
    )
    assert rates.loc[1789, "necessity"] == pytest.approx(expected_necessity)

    pronouns = site.fig_summary_pronouns(stats)
    assert [trace.name for trace in pronouns.data] == [
        "we / us / our family", "I / me / my family",
    ]
    assert "Collective language still leads" in pronouns.layout.title.text
    assert all(np.isfinite(np.asarray(trace.y, dtype=float)).all() for trace in pronouns.data)
    assert all("<b>Presidents:</b>" in trace.hovertemplate for trace in pronouns.data)
    assert all(trace.hovertemplate.count("<br>") == 2 for trace in pronouns.data)
    assert all(trace.mode == "lines+markers" for trace in pronouns.data)
    assert all(trace.line.dash == "solid" for trace in pronouns.data)
    assert len({trace.marker.symbol for trace in pronouns.data}) == len(pronouns.data)
    assert pronouns.layout.hoverlabel.font.size == 10
    assert site._decade_rate(stats, "i_count", 2020) == pytest.approx(
        233.690360, abs=1e-6
    )
    assert site._decade_rate(stats, "we_count", 2020) == pytest.approx(
        328.819710, abs=1e-6
    )

    naming_rates = indices.yearly_rates(markers)
    naming = site.fig_naming_progressive(naming_rates, speeches)
    assert [trace.name for trace in naming.data] == [
        "“United States”", "“America / American(s)”",
    ]
    assert all(trace.mode == "lines+markers" for trace in naming.data)
    assert all(trace.line.dash == "solid" for trace in naming.data)
    assert all(trace.connectgaps is False for trace in naming.data)
    assert len({trace.marker.symbol for trace in naming.data}) == len(naming.data)
    assert all("<b>Presidents:</b>" in trace.hovertemplate for trace in naming.data)
    assert all(trace.hovertemplate.count("<br>") == 2 for trace in naming.data)
    assert naming.layout.hoverlabel.font.size == 10
    assert min(naming.data[0].x) == naming_rates.index.min() == 1789
    assert max(naming.data[0].x) == naming_rates.index.max() == 2026
    assert np.isfinite(np.asarray(naming.data[0].y, dtype=float)[90])
    assert np.isfinite(np.asarray(naming.data[0].y, dtype=float)[200])
    assert naming.layout.xaxis.title.text == "center year of five-year window"


def test_summary_stance_necessity_counts_complete_need_family_once():
    metadata = {
        "doc_name": ["example"],
        "president": ["Example President"],
        "date": pd.to_datetime(["2000-01-01"]),
        "year": [2000],
    }
    stats = pd.DataFrame({
        **metadata,
        "modal_must": [1],
        "modal_will": [2],
        "modal_shall": [3],
        "modal_would": [4],
        "modal_could": [5],
        "modal_should": [6],
    })
    markers = pd.DataFrame({
        **metadata,
        "n_words": [1_000],
        "boosters": [7],
        "hedges": [8],
    })
    speeches = pd.DataFrame({
        **metadata,
        "transcript": [
            "We need action. It needs to happen. Help was needed. We are needing help. "
            "We have to act. She has to act. They had to act. We got to act. We gotta act. "
            "We pledge to act. This administration plans to act. I'm committed to act. "
            "We're determined to act."
        ],
    })

    rates, _context = site.summary_stance_family_rates(
        stats, markers, speeches, window=1, min_words=1
    )

    # One must + four non-overlapping need forms + five obligation phrases.
    assert rates.loc[2000, "necessity"] == pytest.approx(100.0)
    # Five modal uses + four explicit speaker/administration commitments.
    assert rates.loc[2000, "commitment"] == pytest.approx(90.0)
    assert rates.loc[2000, "absolute_emphasis"] == pytest.approx(70.0)


def test_divisiveness_synthesis_keeps_three_inferences_separate():
    stats = pd.read_parquet(DATA / "speech_stats.parquet")
    contract = site.summary_divisiveness_contract(
        pd.read_parquet(combat.RATIOS_PATH),
        pd.read_parquet(eras.ERA_SIMILARITY_PATH),
        stats,
    )
    assert contract["party"].ratio == pytest.approx(4.806243, abs=1e-6)
    assert (contract["party"].ci_lo, contract["party"].ci_hi) == pytest.approx(
        (2.348699, 14.315654), abs=1e-6
    )
    assert contract["enemy"].ci_lo < 1 < contract["enemy"].ci_hi
    assert contract["zero_sum"].ci_lo < 1 < contract["zero_sum"].ci_hi
    assert contract["nearest"].unit_j == "Civil War & Reconstruction"
    assert contract["nearest"].cosine == pytest.approx(0.314796, abs=1e-6)

    html = site._summary_divisiveness_html(
        pd.read_parquet(combat.RATIOS_PATH),
        pd.read_parquet(eras.ERA_SIMILARITY_PATH),
        stats,
    )
    assert "More openly partisan, yes" in html
    assert "More divisive in every broader sense, no" in html
    assert "THE RESULT I STAND BEHIND" in html
    assert "A DESCRIPTIVE SHIFT, NOT A DIVISION SCORE" in html
    assert "AN ANALOGY I WOULD CAVEAT" in html
    assert "Both intervals include one" in html
    assert "low-cluster caution" in html


def test_temporal_president_contract_reconciles_exact_receipts(temporal_contract):
    rows = temporal_contract["rows"]
    assert temporal_contract["schema_version"] == "summary-temporal-president-v2"
    assert temporal_contract["treatment"] == "all_corpus_document_owner"
    assert temporal_contract["speaker_scope_status"] == (
        "document_owned_transcripts_not_speaker_audited"
    )
    assert len(rows) == rows["president"].nunique() == 45
    assert tuple(rows["display_order"]) == tuple(range(45))
    assert tuple(rows["president"]) == site.SUMMARY_CONFLICT_DISPLAY_ORDER
    assert np.allclose(
        rows["future"],
        rows["n_future_matches"] / rows["n_rate_words"] * 10_000,
    )
    assert np.allclose(
        rows["nostalgia"],
        rows["n_nostalgia_matches"] / rows["n_rate_words"] * 10_000,
    )
    assert "self_reference" not in rows
    assert "n_i_pronouns" not in rows
    assert "n_we_pronouns" not in rows
    assert "self_reference_definition" not in temporal_contract
    assert "speech_stats.parquet" not in temporal_contract["source_label"]
    assert rows["support_status"].value_counts().to_dict() == {
        "observed": 42,
        "thin_record": 3,
    }
    assert set(rows.loc[rows["support_status"].eq("thin_record"), "president"]) == {
        "William Harrison", "James A. Garfield", "Zachary Taylor",
    }


def test_temporal_contract_pools_counts_instead_of_averaging_speech_rates():
    markers, speeches = _minimal_temporal_sources()
    contract = site.build_summary_temporal_president_contract(
        markers, speeches, expected_presidents=None
    )
    row = contract["rows"].iloc[0]
    assert row.future == pytest.approx(100)
    assert row.nostalgia == pytest.approx(200)
    assert row.n_rate_words == 1_000
    assert row.n_future_matches == 10
    assert row.n_nostalgia_matches == 20


def test_temporal_contract_rejects_key_metadata_and_receipt_drift():
    markers, speeches = _minimal_temporal_sources()

    with pytest.raises(ValueError, match="key sets differ"):
        site.build_summary_temporal_president_contract(
            markers.iloc[:1], speeches, expected_presidents=None
        )

    duplicate_markers = pd.concat([markers, markers.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="one named row per speech"):
        site.build_summary_temporal_president_contract(
            duplicate_markers, speeches, expected_presidents=None
        )

    wrong_president = markers.copy()
    wrong_president.loc[0, "president"] = "Wrong President"
    with pytest.raises(ValueError, match="president metadata differs"):
        site.build_summary_temporal_president_contract(
            wrong_president, speeches, expected_presidents=None
        )

    wrong_year = markers.copy()
    wrong_year.loc[0, "year"] = 1902
    with pytest.raises(ValueError, match="year metadata differs"):
        site.build_summary_temporal_president_contract(
            wrong_year, speeches, expected_presidents=None
        )

    fractional_count = markers.copy()
    fractional_count["future"] = fractional_count["future"].astype(float)
    fractional_count.loc[0, "future"] = 1.5
    with pytest.raises(ValueError, match="not integral"):
        site.build_summary_temporal_president_contract(
            fractional_count, speeches, expected_presidents=None
        )

    impossible_matches = markers.copy()
    impossible_matches.loc[0, ["n_words", "future"]] = [0, 1]
    with pytest.raises(ValueError, match="marker-word denominator"):
        site.build_summary_temporal_president_contract(
            impossible_matches, speeches, expected_presidents=None
        )

    null_president = speeches.copy()
    null_president.loc[0, "president"] = None
    with pytest.raises(ValueError, match="named president"):
        site.build_summary_temporal_president_contract(
            markers, null_president, expected_presidents=None
        )

    fractional_year = speeches.copy()
    fractional_year["year"] = fractional_year["year"].astype(float)
    fractional_year.loc[0, "year"] = 1900.5
    with pytest.raises(ValueError, match="finite integers"):
        site.build_summary_temporal_president_contract(
            markers, fractional_year, expected_presidents=None
        )


def test_temporal_portrait_scatter_uses_future_nostalgia_and_uniform_portraits(
    temporal_contract,
):
    rows = temporal_contract["rows"]
    faces = {
        president: "data:image/png;base64,AA=="
        for president in rows["president"].astype(str)
    }
    figure = site.fig_summary_temporal(temporal_contract, faces)
    trace = figure.data[0]
    plotted_presidents = [str(row[0]) for row in trace.customdata]
    plotted = rows.set_index("president").loc[plotted_presidents]

    assert len(trace.x) == len(figure.layout.images) == 45
    assert np.allclose(trace.x, plotted["future"])
    assert np.allclose(trace.y, plotted["nostalgia"])
    diameters = np.asarray(trace.marker.size, dtype=float)
    assert np.all(diameters == site.SUMMARY_TEMPORAL_PORTRAIT_PX)
    assert all(str(image.name).startswith("portrait::") for image in figure.layout.images)
    image_presidents = [
        str(image.name).removeprefix("portrait::")
        for image in figure.layout.images
    ]
    image_rows = rows.set_index("president").loc[image_presidents]
    assert np.allclose([image.x for image in figure.layout.images], image_rows.future)
    assert np.allclose([image.y for image in figure.layout.images], image_rows.nostalgia)
    assert figure.layout.xaxis.range[0] < 0 < figure.layout.xaxis.range[1]
    assert figure.layout.yaxis.range[0] < 0 < figure.layout.yaxis.range[1]
    assert figure.layout.xaxis.fixedrange is True
    assert figure.layout.yaxis.fixedrange is True
    assert figure.layout.xaxis.title.text == (
        "future-family matches per 10,000 marker words"
    )
    assert figure.layout.yaxis.title.text == (
        "nostalgia-family matches per 10,000 marker words"
    )
    assert "<b>Tomorrow:</b>" in trace.hovertemplate
    assert "<b>Yesterday:</b>" in trace.hovertemplate
    assert "self-reference" not in trace.hovertemplate
    assert "singular" not in trace.hovertemplate
    assert "plural" not in trace.hovertemplate
    thin = plotted["support_status"].eq("thin_record").to_numpy()
    widths = np.asarray(trace.marker.line.width, dtype=float)
    assert np.all(widths[thin] == 4)
    assert np.all(widths[~thin] == 3)
    x_span = float(figure.layout.xaxis.range[1] - figure.layout.xaxis.range[0])
    y_span = float(figure.layout.yaxis.range[1] - figure.layout.yaxis.range[0])
    visible_widths = np.asarray([
        image.sizex / x_span * site.SUMMARY_TEMPORAL_PLOT_WIDTH_PX
        for image in figure.layout.images
    ])
    visible_heights = np.asarray([
        image.sizey / y_span * site.SUMMARY_TEMPORAL_PLOT_HEIGHT_PX
        for image in figure.layout.images
    ])
    assert np.allclose(visible_widths, site.SUMMARY_TEMPORAL_PORTRAIT_PX)
    assert np.allclose(visible_heights, site.SUMMARY_TEMPORAL_PORTRAIT_PX)
    serialized = json.loads(figure.to_json())
    marker = serialized["data"][0]["marker"]
    assert isinstance(marker["size"], list)
    assert isinstance(marker["color"], list)
    assert isinstance(marker["line"]["color"], list)
    assert isinstance(marker["line"]["width"], list)


def test_temporal_portrait_fails_closed_when_a_face_is_missing(
    temporal_contract,
):
    presidents = temporal_contract["rows"]["president"].astype(str).tolist()
    faces = {
        president: "data:image/png;base64,AA=="
        for president in presidents[1:]
    }
    with pytest.raises(ValueError, match="missing portraits"):
        site.fig_summary_temporal(temporal_contract, faces)


def test_temporal_portrait_handles_unavailable_marker_denominator():
    speeches = pd.DataFrame({
        "doc_name": ["a"], "president": ["Example President"],
        "year": [1900], "date": pd.to_datetime(["1900-01-01"]),
    })
    markers = pd.DataFrame({
        "doc_name": ["a"], "president": ["Example President"],
        "year": [1900], "n_words": [0], "future": [0], "nostalgia": [0],
    })
    contract = site.build_summary_temporal_president_contract(
        markers, speeches, expected_presidents=None
    )
    assert contract["rows"].iloc[0].support_status == "not_available"
    figure = site.fig_summary_temporal(
        contract, {"Example President": "data:image/png;base64,AA=="}
    )
    assert len(figure.layout.images) == 0
    assert any("N/A" in str(annotation.text) for annotation in figure.layout.annotations)
    rendered = site._summary_temporal_portrait_html(contract)
    assert "nan%" not in rendered
    assert "data-era-select" not in rendered
    assert 'data-fig="summary_temporal"' in rendered


def test_temporal_timeline_keeps_two_distinct_supported_lines():
    years = pd.RangeIndex(1789, 2027, name="year")
    markers = pd.DataFrame({
        "year": years,
        "n_words": np.full(len(years), 6_000),
        "future": np.full(len(years), 6),
        "nostalgia": np.full(len(years), 12),
    })
    markers.loc[markers["year"].eq(1800), ["n_words", "future", "nostalgia"]] = 0
    figure = site.fig_summary_temporal_timeline(markers)
    assert len(figure.data) == 2
    assert [trace.name for trace in figure.data] == ["Tomorrow", "Yesterday"]
    assert all(trace.mode == "lines" for trace in figure.data)
    assert all(trace.connectgaps is False for trace in figure.data)
    assert figure.data[0].line.color != figure.data[1].line.color
    assert figure.data[0].line.dash != figure.data[1].line.dash
    assert figure.data[0].x[0] == 1792
    assert figure.data[0].x[-1] == 2026
    assert figure.data[0].customdata[0] == "1789–1792 window"
    assert figure.data[0].y[0] == pytest.approx(10)
    assert figure.data[1].y[0] == pytest.approx(20)
    assert np.isnan(figure.data[0].y[10])
    assert np.isnan(figure.data[1].y[10])
    assert figure.layout.hovermode == "x unified"
    assert figure.layout.title.text == "Tomorrow and yesterday over time"
    assert figure.layout.yaxis.title.text == "matches per 10,000 marker words"
    assert figure.layout.xaxis.title.text == (
        "ending year of supported four-year rolling average"
    )
    assert len(figure.layout.shapes) == 9
    assert len(figure.layout.annotations) == 9
    unequal = pd.DataFrame({
        "year": [1800, 1801, 1802, 1803],
        "n_words": [1_000, 1_000, 1_000, 17_000],
        "future": [1, 1, 1, 170],
        "nostalgia": [2, 2, 2, 34],
    })
    rolling = site._summary_temporal_timeline_frame(unequal)
    assert list(rolling.index) == [1803]
    assert rolling.iloc[0].future == pytest.approx(32.5)
    assert rolling.iloc[0].nostalgia == pytest.approx(20)
    boundary = unequal.assign(
        n_words=[2_500] * 4,
        future=[1] * 4,
        nostalgia=[2] * 4,
    )
    boundary_frame = site._summary_temporal_timeline_frame(boundary)
    assert boundary_frame.iloc[0].n_words == 10_000
    assert boundary_frame.iloc[0][["future", "nostalgia"]].notna().all()


def test_temporal_timeline_rejects_invalid_or_unsupported_marker_rows():
    with pytest.raises(ValueError, match="missing marker columns"):
        site.fig_summary_temporal_timeline(pd.DataFrame({"future": [1.0]}))
    fractional = pd.DataFrame({
        "year": [1800.5], "n_words": [20_000],
        "future": [1], "nostalgia": [1],
    })
    with pytest.raises(ValueError, match="years must be integers"):
        site.fig_summary_temporal_timeline(fractional)
    impossible = pd.DataFrame({
        "year": [1800], "n_words": [1], "future": [2], "nostalgia": [0],
    })
    with pytest.raises(ValueError, match="cannot exceed marker words"):
        site.fig_summary_temporal_timeline(impossible)
    unsupported = pd.DataFrame({
        "year": [1800, 1801, 1802, 1803], "n_words": [1_000] * 4,
        "future": [1] * 4, "nostalgia": [1] * 4,
    })
    with pytest.raises(ValueError, match="no supported windows"):
        site.fig_summary_temporal_timeline(unsupported)


def test_voice_register_uses_one_all_dim_or_single_era_selector():
    scores = pd.DataFrame({
        "first_year": [1789], "last_year": [1797], "mechanism": [40.0],
        "hype": [2.0], "n_speeches": [10],
    }, index=["George Washington"])
    rates = indices.yearly_rates(pd.read_parquet(DATA / "speech_markers.parquet"))
    rendered = site._procedural_era_html(scores, rates)
    assert rendered.count("data-era-select") == 1
    assert rendered.count('<option value="__all__" selected>') == 1
    assert rendered.count('<option value="__none__">') == 1
    assert rendered.count("<option value=") == 11
    assert rendered.count("data-era-choice") == 0
    assert rendered.count('class="era-chip"') == 0
    assert rendered.count("data-era-preset") == 0
    assert "Emphasize an era" in rendered
    assert "All nine eras emphasized" in rendered
    assert "Who + how → register" in rendered
    assert "Less procedural → procedural middle → higher hype later" in rendered
    assert "do not establish that audience or delivery caused" in rendered
    assert rendered.count("data-register-view=") == 2
    assert 'data-register-view="presidents" aria-pressed="true"' in rendered
    assert 'data-register-view="timeline" aria-pressed="false"' in rendered
    assert 'aria-describedby="procedural_eras-era-status"' in rendered
    assert 'class="chart-scroll register-chart-scroll"' in rendered
    assert (
        'data-timeline-status="Over time: the legal/procedural-per-hype ratio is shown'
        in rendered
    )
    assert "Text alternative" not in rendered
    assert "<table" not in rendered
    assert 'class="register-timeline-context"' not in rendered
    assert 'aria-label="Explanations for selected timeline movements"' not in rendered
    assert 'aria-label="Centered-window explanations"' not in rendered
    assert 'aria-label="Numbered historical event context"' not in rendered
    assert 'aria-label="Administration transition context"' not in rendered
    assert "Selected turns in the line" not in rendered
    assert 'class="register-moment-year"' not in rendered
    assert "Both exact text alternatives" not in rendered
    assert "The interactive register chart requires JavaScript" in rendered
    figure = site.fig_procedural_eras(
        scores, {"George Washington": "data:image/png;base64,AA=="}
    )
    marker = json.loads(figure.to_json())["data"][0]["marker"]
    assert isinstance(marker["opacity"], list)
    assert isinstance(marker["color"], list)
    assert isinstance(marker["line"]["width"], list)
    assert isinstance(marker["line"]["color"], list)


def test_voice_register_timeline_is_one_guarded_ratio_line_with_honest_gaps():
    years = pd.RangeIndex(1789, 2027, name="year")
    rates = pd.DataFrame({
        "mechanism": np.linspace(45, 12, len(years)),
        "hype": np.linspace(3, 22, len(years)),
    }, index=years)
    rates.loc[1800, ["mechanism", "hype"]] = np.nan
    rates.loc[1801, "hype"] = 0
    figure = site.fig_summary_register_timeline(rates)
    assert len(figure.data) == 1
    assert figure.data[0].name == "Legal/procedural ÷ hype"
    assert figure.data[0].mode == "lines"
    assert figure.data[0].connectgaps is False
    assert figure.data[0].y[0] == pytest.approx(15.0)
    assert np.isnan(figure.data[0].y[11])
    assert np.isnan(figure.data[0].y[12])
    assert figure.layout.showlegend is False
    assert figure.layout.hovermode == "x"
    assert figure.layout.title.text == "Legal/procedural ÷ hype over time"
    assert figure.layout.title.xanchor == "left"
    assert figure.layout.yaxis.type == "log"
    assert (
        figure.layout.yaxis.title.text
        == "legal/procedural matches per hype match · log scale"
    )
    assert list(figure.layout.yaxis.ticktext) == [
        "0.3", "0.5", "1", "2", "5", "10", "20", "50", "100"
    ]
    assert len(figure.layout.shapes) == 9
    assert len(figure.layout.annotations) == 16
    assert [annotation.text for annotation in figure.layout.annotations[-7:]] == [
        "1827", "1863", "1881", "1944", "1966", "2016", "2023"
    ]
    assert [annotation.x for annotation in figure.layout.annotations[-7:]] == [
        1827, 1863, 1881, 1944, 1966, 2016, 2023
    ]
    for annotation, year in zip(
        figure.layout.annotations[-7:], (1827, 1863, 1881, 1944, 1966, 2016, 2023)
    ):
        assert annotation.y == pytest.approx(
            np.log10(float(figure.data[0].y[year - 1789]))
        )
    for annotation, moment in zip(
        figure.layout.annotations[-7:], site.SUMMARY_REGISTER_MOMENTS
    ):
        summary = site.SUMMARY_REGISTER_HOVER_SUMMARIES[moment[0]]
        assert " ".join(annotation.hovertext.split("<br>")) == summary
        assert annotation.hovertext.count("<br>") >= 1
        assert moment[1] not in annotation.hovertext
        assert moment[2] not in annotation.hovertext
        assert "<b>" not in annotation.hovertext
    assert set(site.SUMMARY_REGISTER_HOVER_SUMMARIES) == {
        moment[0] for moment in site.SUMMARY_REGISTER_MOMENTS
    }
    assert figure.layout.hoverlabel.align == "left"
    assert "1798–1802 window" == figure.data[0].customdata[11][0]
    assert "legal/procedural matches per hype match" in figure.data[0].hovertemplate
    assert "Legal/procedural: %{customdata[1]:.2f}" in figure.data[0].hovertemplate
    assert "Hype: %{customdata[2]:.2f}" in figure.data[0].hovertemplate


def test_voice_register_moment_receipts_match_speech_markers():
    markers = pd.read_parquet(DATA / "speech_markers.parquet")
    for year, _, _, expected_legal, expected_hype, _ in site.SUMMARY_REGISTER_MOMENTS:
        window = markers[markers.year.between(year - 2, year + 2)]
        assert int(window.mechanism.sum()) == expected_legal
        assert int(window.hype.sum()) == expected_hype
    adams_window = markers[markers.year.between(1825, 1829)]
    adams = adams_window[adams_window.president.eq("John Quincy Adams")]
    assert (len(adams), int(adams.mechanism.sum()), int(adams.hype.sum())) == (
        7, 183, 4
    )

    mid_2010s = markers[markers.year.between(2014, 2018)]
    by_president = mid_2010s.groupby("president")[["mechanism", "hype"]].sum()
    assert tuple(by_president.loc["Barack Obama"]) == (41, 36)
    assert tuple(by_president.loc["Donald Trump"]) == (51, 200)

    recent = markers[markers.year.between(2021, 2025)]
    recent_by_president = recent.groupby("president")[["mechanism", "hype"]].sum()
    assert tuple(recent_by_president.loc["Joe Biden"]) == (148, 132)
    biden_annual = recent[
        recent.president.eq("Joe Biden")
        & recent.title.str.contains(
            r"State of (?:the )?Union|Joint Session", case=False, regex=True
        )
    ]
    assert (len(biden_annual), int(biden_annual.mechanism.sum())) == (4, 96)


def test_voice_register_timeline_fails_closed_on_invalid_rates():
    with pytest.raises(ValueError, match="missing rate columns"):
        site.fig_summary_register_timeline(pd.DataFrame({"hype": [1.0]}))
    unordered = pd.DataFrame(
        {"mechanism": [1.0, 2.0], "hype": [2.0, 3.0]},
        index=[1790, 1789],
    )
    with pytest.raises(ValueError, match="unique and ordered"):
        site.fig_summary_register_timeline(unordered)
    zero_denominator = pd.DataFrame(
        {"mechanism": [1.0, 2.0], "hype": [0.0, 0.0]},
        index=[1789, 1790],
    )
    with pytest.raises(ValueError, match="no supported windows"):
        site.fig_summary_register_timeline(zero_denominator)


def test_conflict_atlas_contract_covers_every_president_and_balances_categories():
    frame = pd.read_parquet(DATA / "combat" / "by_president.parquet")
    contract = site.build_summary_conflict_contract(frame)
    rows = contract["rows"]
    share_columns = [
        f"adv_share_{key}"
        for _, key, *_ in site.SUMMARY_CONFLICT_CATEGORY_SPECS
    ]
    assert len(rows) == 45
    assert rows.president.is_unique
    has_entities = rows["n_adversarial_entities"].gt(0)
    assert np.allclose(rows.loc[has_entities, share_columns].sum(axis=1), 1)
    assert rows.loc[~has_entities, share_columns].isna().all().all()
    maximum_rate = rows[
        [column for column, *_ in site.SUMMARY_CONFLICT_RATE_SPECS]
    ].max().max()
    assert contract["rate_axis_max"] >= maximum_rate
    assert np.isclose(contract["rate_axis_max"] * 10, round(contract["rate_axis_max"] * 10))
    assert contract["speaker_scope_status"] == "pending_speaker_audit"
    assert "speaker attribution audit pending" in contract["treatment_label"]
    ordered = site._summary_conflict_ordered_rows(contract)
    assert tuple(ordered["president"]) == site.SUMMARY_CONFLICT_DISPLAY_ORDER
    assert tuple(ordered["display_order"]) == tuple(range(45))
    expected_composition_status = np.select(
        [
            rows["n_adversarial_entities"].eq(0),
            rows["n_adversarial_entities"].lt(site.SUMMARY_CONFLICT_COMPOSITION_MIN),
        ],
        ["not_available", "thin_record"],
        default="observed",
    )
    assert tuple(rows["composition_status"]) == tuple(expected_composition_status)
    expected_support_status = np.select(
        [
            rows["n_paragraphs"].eq(0),
            (rows["n_speeches"] < 5) | (rows["n_paragraphs"] < 100),
        ],
        ["not_available", "thin_record"],
        default="observed",
    )
    assert tuple(rows["support_status"]) == tuple(expected_support_status)


def test_conflict_target_contract_covers_fixed_eras_and_reconciles_presidents():
    target_frame = pd.read_parquet(
        DATA / "combat" / "target_mix_by_era_speaker_audited_v1.parquet"
    )
    target = site.build_summary_conflict_target_contract(target_frame)
    rows = target["rows"]
    expected_axis = tuple(
        (order, label, start, end)
        for order, (label, start, end) in enumerate(trends.ERAS)
    )
    actual_axis = tuple(
        rows[["era_order", "era", "era_start", "era_end"]]
        .itertuples(index=False, name=None)
    )
    assert actual_axis == expected_axis
    share_columns = [
        f"adv_share_{key}"
        for _, key, *_ in site.SUMMARY_CONFLICT_CATEGORY_SPECS
    ]
    assert np.allclose(rows[share_columns].sum(axis=1), 1)
    assert set(rows["composition_status"]) == {"observed"}
    assert target["target_axis_max"] == 60
    assert target["era_scheme"] == "trends.ERAS"
    assert target["year_basis"] == "canonical_source_speech_year"
    assert target["corpus_end_date"].date().isoformat() == "2026-04-02"

    presidents = pd.read_parquet(
        DATA / "combat" / "by_president_speaker_audited_v2.parquet"
    )
    for count_column, *_ in site.SUMMARY_CONFLICT_CATEGORY_SPECS:
        assert int(rows[count_column].sum()) == int(presidents[count_column].sum())
    assert int(rows["n_adversarial_entities"].sum()) == 8_393
    assert int(rows["n_paragraphs"].sum()) == int(presidents["n_paragraphs"].sum())


def test_conflict_target_contract_rejects_era_boundary_drift():
    frame = pd.read_parquet(
        DATA / "combat" / "target_mix_by_era_speaker_audited_v1.parquet"
    )
    frame.loc[frame.index[0], "era_end"] += 1
    with pytest.raises(ValueError, match="era axis differs"):
        site.build_summary_conflict_target_contract(frame)


def test_conflict_contract_rejects_membership_and_count_drift():
    frame = pd.read_parquet(DATA / "combat" / "by_president.parquet")
    wrong_member = frame.copy()
    wrong_member.loc[wrong_member.index[0], "president"] = "Replacement President"
    with pytest.raises(ValueError, match="president membership changed"):
        site.build_summary_conflict_contract(wrong_member)

    invalid_count = frame.head(1).copy()
    invalid_count["n_party_attack"] = 1.5
    with pytest.raises(ValueError, match="n_party_attack.*not integral"):
        site.build_summary_conflict_contract(invalid_count, expected_presidents=None)

    orphan_count = frame.head(1).copy()
    orphan_count["enemy_naming"] = np.nan
    orphan_count["n_enemy_naming"] = 5
    with pytest.raises(ValueError, match="n_enemy_naming.*invalid"):
        site.build_summary_conflict_contract(orphan_count, expected_presidents=None)

    contradictory_status = frame.head(1).copy()
    for column, *_ in site.SUMMARY_CONFLICT_CATEGORY_SPECS:
        contradictory_status[column] = 0
    contradictory_status["composition_status"] = "observed"
    with pytest.raises(ValueError, match="composition_status contradicts"):
        site.build_summary_conflict_contract(
            contradictory_status, expected_presidents=None
        )

    contradictory_support = frame.head(1).copy()
    contradictory_support["n_speeches"] = 0
    contradictory_support["support_status"] = "observed"
    with pytest.raises(ValueError, match="support_status contradicts"):
        site.build_summary_conflict_contract(
            contradictory_support, expected_presidents=None
        )


def test_conflict_target_lines_keep_all_nine_eras_in_fixed_order():
    frame = pd.read_parquet(
        DATA / "combat" / "target_mix_by_era_speaker_audited_v1.parquet"
    )
    contract = site.build_summary_conflict_target_contract(frame)
    figure = site.fig_summary_conflict_targets(contract)
    lines = [
        trace for trace in figure.data
        if trace.type == "scatter" and trace.mode == "lines+markers"
    ]
    assert len(lines) == 5
    for trace in lines:
        assert tuple(int(value) for value in trace.x) == tuple(range(9))
        assert tuple(str(row[0]) for row in trace.customdata) == (
            tuple(label for label, _start, _end in trends.ERAS)
        )
    assert tuple(figure.layout.xaxis.tickvals) == tuple(range(9))
    assert len(set(figure.layout.xaxis.ticktext)) == 9
    assert "Founding<br>1789–1815" == figure.layout.xaxis.ticktext[0]
    assert "Present<br>2017–2026" == figure.layout.xaxis.ticktext[-1]
    assert "xaxis2" not in figure.layout


def test_conflict_target_graph_is_five_lines_on_one_shared_axis():
    frame = pd.read_parquet(
        DATA / "combat" / "target_mix_by_era_speaker_audited_v1.parquet"
    )
    contract = site.build_summary_conflict_target_contract(frame)
    ordered = contract["rows"]
    figure = site.fig_summary_conflict_targets(contract)
    lines = [
        trace for trace in figure.data
        if trace.type == "scatter" and trace.mode == "lines+markers"
    ]

    assert [trace.name for trace in lines] == [
        label for _column, _key, _emoji, label, _color
        in site.SUMMARY_CONFLICT_CATEGORY_SPECS
    ]
    expected_colors = [
        color for _column, _key, _emoji, _label, color
        in site.SUMMARY_CONFLICT_CATEGORY_SPECS
    ]
    assert [trace.line.color for trace in lines] == expected_colors
    assert len({trace.line.dash for trace in lines}) == 5
    assert all(trace.connectgaps is False for trace in lines)
    assert not any(trace.type == "bar" for trace in figure.data)
    assert [trace.xaxis for trace in lines] == [None] * 5
    assert [trace.yaxis for trace in lines] == [None] * 5

    for trace, (_column, key, _emoji, _label, _color) in zip(
        lines, site.SUMMARY_CONFLICT_CATEGORY_SPECS
    ):
        assert np.allclose(
            np.asarray(trace.y, dtype=float),
            ordered[f"adv_share_{key}"].mul(100),
            equal_nan=True,
        )
    supported = ordered["n_adversarial_entities"].gt(0)
    plotted_shares = np.asarray([trace.y for trace in lines], dtype=float)
    assert np.allclose(plotted_shares[:, supported].sum(axis=0), 100)
    assert tuple(figure.layout.yaxis.range) == (0, 60)
    assert figure.layout.yaxis.ticksuffix == "%"
    assert "yaxis2" not in figure.layout
    assert figure.layout.xaxis.title.text == "fixed reporting eras · categorical order"
    assert "share of adversarial" in figure.layout.yaxis.title.text
    assert figure.layout.hovermode == "closest"
    assert figure.layout.hoversubplots is None
    assert figure.layout.showlegend is False
    assert [trace.meta for trace in lines] == [
        key for _column, key, _emoji, _label, _color
        in site.SUMMARY_CONFLICT_CATEGORY_SPECS
    ]
    assert all(trace.text is None for trace in lines)
    callouts = {
        str(annotation.text): annotation
        for annotation in figure.layout.annotations
        if str(annotation.text) != "N/A"
    }
    assert set(callouts) == {
        "Institution spike", "Domestic turn", "Group peak",
        "Wartime nations", "Group share", "Broader mix",
    }
    assert all(annotation.hovertext for annotation in callouts.values())
    assert "14 to 235" in callouts["Institution spike"].hovertext.replace("<br>", " ")
    assert "Bank of the United States" in callouts["Institution spike"].hovertext.replace("<br>", " ")
    assert "Southern people" in callouts["Domestic turn"].hovertext.replace("<br>", " ")
    assert "Democratic Party" in callouts["Group peak"].hovertext.replace("<br>", " ")
    assert "570 to 572" in callouts["Wartime nations"].hovertext.replace("<br>", " ")
    assert "Germany and Japan" in callouts["Wartime nations"].hovertext.replace("<br>", " ")
    assert "al Qaeda" in callouts["Group share"].hovertext.replace("<br>", " ")
    assert "despite more group mentions" in callouts["Broader mix"].hovertext.replace("<br>", " ")
    assert "Putin" in callouts["Broader mix"].hovertext.replace("<br>", " ")


def test_conflict_turning_point_named_drivers_match_frozen_entity_evidence():
    entities = pd.read_parquet(
        DATA / "llm_annotations" / "paragraph_entities.parquet"
    )
    paragraph_view = pd.read_parquet(
        DATA / "speaker_views" / "paragraph_view_v1.parquet"
    )
    eligible = paragraph_view.loc[
        paragraph_view["analysis_eligible"],
        ["doc_name", "para_idx", "year"],
    ]
    adversarial = entities.loc[entities["stance"].eq("adversarial")].merge(
        eligible,
        on=["doc_name", "para_idx"],
        how="inner",
        validate="many_to_one",
    )
    adversarial["era"] = adversarial["year"].map(combat.era_of)
    expected = {
        ("Expansion", "Bank of the United States"): 122,
        ("Expansion", "Senate"): 37,
        ("Civil War & Reconstruction", "Southern people"): 39,
        ("Civil War & Reconstruction", "Edwin M. Stanton"): 26,
        ("Civil War & Reconstruction", "Senator Douglas"): 16,
        ("Progressives & Depression", "Democratic Party"): 24,
        ("Progressives & Depression", "opponents"): 11,
        ("Progressives & Depression", "alien enemies"): 9,
        ("War & New Deal", "Germany"): 70,
        ("War & New Deal", "Japan"): 70,
        ("War & New Deal", "Italy"): 19,
        ("Post-Cold War", "al Qaeda"): 73,
        ("Post-Cold War", "terrorists"): 52,
        ("Post-Cold War", "Taliban"): 37,
        ("The present era", "Putin"): 53,
        ("The present era", "Biden"): 43,
        ("The present era", "Joe Biden"): 34,
        ("The present era", "Donald Trump"): 32,
    }
    actual = adversarial.groupby(["era", "entity"], observed=True).size()
    assert {
        key: int(actual.get(key, 0))
        for key in expected
    } == expected


def test_conflict_portrait_scatter_uses_requested_axes_and_enemy_size():
    frame = pd.read_parquet(DATA / "combat" / "by_president.parquet")
    contract = site.build_summary_conflict_contract(frame)
    ordered = site._summary_conflict_ordered_rows(contract)
    faces = {
        president: "data:image/png;base64,AA=="
        for president in ordered["president"].astype(str)
    }
    figure = site.fig_summary_conflict_portraits(contract, faces)
    trace = figure.data[0]
    plotted_presidents = [str(row[0]) for row in trace.customdata]
    plotted = ordered.set_index("president").loc[plotted_presidents]

    assert len(trace.x) == 45
    assert len(figure.layout.images) == 45
    assert np.allclose(trace.x, plotted["zero_sum"] * 100)
    assert np.allclose(trace.y, plotted["party_attack"] * 100)
    image_presidents = [
        image.name.removeprefix("conflict-portrait::")
        for image in figure.layout.images
    ]
    image_rows = ordered.set_index("president").loc[image_presidents]
    assert np.allclose(
        [image.x for image in figure.layout.images], image_rows["zero_sum"] * 100
    )
    assert np.allclose(
        [image.y for image in figure.layout.images],
        image_rows["party_attack"] * 100,
    )
    diameter = np.asarray(trace.marker.size, dtype=float)
    encoded_area = diameter ** 2 / site.SUMMARY_CONFLICT_PORTRAIT_MAX_PX ** 2
    expected_area = plotted["enemy_naming"].div(
        plotted["enemy_naming"].max()
    )
    assert np.allclose(encoded_area, expected_area)
    assert np.all(np.diff(diameter) <= 0)
    assert figure.layout.xaxis.range[0] < 0
    assert figure.layout.xaxis.range[1] > max(trace.x)
    assert figure.layout.yaxis.range[0] < 0
    assert figure.layout.yaxis.range[1] > max(trace.y)
    assert figure.layout.xaxis.title.text == "paragraphs with zero-sum framing"
    assert figure.layout.yaxis.title.text == "paragraphs with partisan attack"
    assert "<b>Enemy naming:</b> %{customdata[3]:.1f}% of paragraphs" in (
        trace.hovertemplate
    )
    expected_border_colors = [
        site._summary_conflict_dominant_adversary(row)[2]
        for row in plotted.itertuples(index=False)
    ]
    assert list(trace.marker.line.color) == expected_border_colors
    assert float(trace.marker.line.width) == (
        site.SUMMARY_CONFLICT_PORTRAIT_BORDER_PX
    )
    assert "<b>Most-named adversary type:</b> %{customdata[6]}" in (
        trace.hovertemplate
    )
    assert "<b>Speeches analyzed:</b> %{customdata[4]}" in trace.hovertemplate
    assert "<br>counts:" not in trace.hovertemplate
    assert "<b>Support:</b> %{customdata[5]}" in trace.hovertemplate
    assert all(
        image.name.startswith("conflict-portrait::")
        for image in figure.layout.images
    )


def test_conflict_portrait_common_genre_uses_governed_annual_messages():
    treatments = pd.read_parquet(
        DATA / "combat" / "by_president_treatments_v2.parquet"
    )
    frame = treatments.loc[
        treatments["treatment"].eq(site.SUMMARY_CONFLICT_PORTRAIT_TREATMENT)
    ]
    contract = site.build_summary_conflict_contract(frame)
    ordered = site._summary_conflict_ordered_rows(contract)
    faces = {
        president: "data:image/png;base64,AA=="
        for president in ordered["president"].astype(str)
    }
    figure = site.fig_summary_conflict_portraits(contract, faces)
    plotted_presidents = [str(row[0]) for row in figure.data[0].customdata]

    assert contract["treatment"] == "annual_message_strict"
    assert contract["treatment_label"] == "Annual messages, speaker-audited"
    assert len(ordered) == 45
    assert len(plotted_presidents) == 42
    assert len(figure.layout.images) == 42
    assert set(ordered.loc[ordered["support_status"].eq("not_available"), "president"]) == {
        "William Harrison", "James A. Garfield", "Harry S. Truman",
    }
    assert not {
        "William Harrison", "James A. Garfield", "Harry S. Truman",
    } & set(plotted_presidents)
    carter = ordered.set_index("president").loc["Jimmy Carter"]
    assert carter.n_speeches == 3
    assert carter.n_paragraphs == 131
    assert carter.n_party_attack == 0
    assert carter.party_attack == 0
    assert carter.n_enemy_naming == 13
    assert carter.enemy_naming == pytest.approx(13 / 131)
    assert carter.n_zero_sum == 6
    assert carter.zero_sum == pytest.approx(6 / 131)
    assert site._summary_conflict_dominant_adversary(carter) == (
        "nation", "Nation", "#315f78",
    )
    monroe = ordered.set_index("president").loc["James Monroe"]
    assert site._summary_conflict_dominant_adversary(monroe) == (
        "tie", "Tie · Nation + Group",
        site.SUMMARY_CONFLICT_PORTRAIT_TIE_COLOR,
    )


def test_conflict_graphs_update_from_replaceable_contract_data():
    target_frame = pd.read_parquet(
        DATA / "combat" / "target_mix_by_era_speaker_audited_v1.parquet"
    )
    before_target_contract = site.build_summary_conflict_target_contract(
        target_frame
    )
    before_targets = site.fig_summary_conflict_targets(before_target_contract)
    changed_targets = target_frame.copy()
    for column, *_ in site.SUMMARY_CONFLICT_CATEGORY_SPECS:
        changed_targets.loc[changed_targets.index[0], column] = 0
    changed_targets.loc[changed_targets.index[0], "adv_n_group"] = 100
    after_target_contract = site.build_summary_conflict_target_contract(
        changed_targets
    )
    after_targets = site.fig_summary_conflict_targets(after_target_contract)

    frame = pd.read_parquet(DATA / "combat" / "by_president.parquet").head(1)
    before_president_contract = site.build_summary_conflict_contract(
        frame, expected_presidents=None
    )
    president = str(frame.iloc[0]["president"])
    faces = {president: "data:image/png;base64,AA=="}
    before_portrait = site.fig_summary_conflict_portraits(
        before_president_contract, faces
    )
    changed = frame.copy()
    paragraph_count = int(changed.iloc[0]["n_paragraphs"])
    partisan_count = max(1, paragraph_count // 8)
    changed.loc[:, "party_attack"] = partisan_count / paragraph_count
    if "n_party_attack" in changed:
        changed.loc[:, "n_party_attack"] = partisan_count
    after_president_contract = site.build_summary_conflict_contract(
        changed, expected_presidents=None
    )
    after_portrait = site.fig_summary_conflict_portraits(
        after_president_contract, faces
    )

    assert before_targets.to_json() != after_targets.to_json()
    assert before_portrait.to_json() != after_portrait.to_json()
    group_trace = next(
        trace for trace in after_targets.data if trace.name == "Group"
    )
    assert float(group_trace.y[0]) == 100
    assert float(after_portrait.data[0].y[0]) == pytest.approx(
        partisan_count / paragraph_count * 100
    )


def test_conflict_graphs_preserve_missing_speaker_audited_rows_as_na():
    target_frame = pd.read_parquet(
        DATA / "combat" / "target_mix_by_era_speaker_audited_v1.parquet"
    ).copy()
    for column, *_ in site.SUMMARY_CONFLICT_CATEGORY_SPECS:
        target_frame.loc[target_frame.index[0], column] = 0
    target_frame.loc[target_frame.index[0], "n_adversarial_speeches"] = 0
    target_frame.loc[target_frame.index[0], "n_active_years"] = 0
    target_contract = site.build_summary_conflict_target_contract(target_frame)
    targets = site.fig_summary_conflict_targets(target_contract)
    assert all(np.isnan(float(trace.y[0])) for trace in targets.data[:5])
    assert all(trace.connectgaps is False for trace in targets.data[:5])
    assert sum(
        "N/A" in str(annotation.text)
        for annotation in targets.layout.annotations
    ) == 1
    assert tuple(targets.layout.xaxis.tickvals) == tuple(range(9))

    frame = pd.read_parquet(DATA / "combat" / "by_president.parquet").head(1).copy()
    frame.loc[:, "n_speeches"] = 0
    frame.loc[:, "n_paragraphs"] = 0
    for column, *_ in site.SUMMARY_CONFLICT_CATEGORY_SPECS:
        frame.loc[:, column] = 0
    for column, *_ in site.SUMMARY_CONFLICT_RATE_SPECS:
        frame.loc[:, column] = np.nan
    contract = site.build_summary_conflict_contract(frame, expected_presidents=None)

    president = str(frame.iloc[0]["president"])
    portrait = site.fig_summary_conflict_portraits(
        contract, {president: "data:image/png;base64,AA=="}
    )
    assert len(portrait.layout.images) == 0
    assert any(
        "N/A" in str(annotation.text)
        for annotation in portrait.layout.annotations
    )

    panel = site._summary_conflict_graphs_html(target_contract, contract)
    assert "Enemy naming N/A" not in panel
    assert "nan%" not in panel


def test_generated_summary_uses_target_timeline_and_portrait_conflict_graphs():
    summary = (DATA.parent / "docs" / "summary.html").read_text()
    expected_keys = {
        "summary_conflict_targets",
        "summary_conflict_frame_portraits",
    }
    assert set(site.SUMMARY_CONFLICT_FIGURE_KEYS) == expected_keys
    assert 'data-conflict-president="' not in summary
    assert 'data-conflict-schema="president-conflict-v2"' in summary
    assert 'data-target-schema="conflict-target-mix-v1"' in summary
    assert 'data-conflict-target-treatment="speaker_audited_all"' in summary
    assert 'data-conflict-president-treatment="annual_message_strict"' in summary
    for key in expected_keys:
        assert summary.count(f'data-fig="{key}"') == 1
    assert "9 ERAS · 45 PRESIDENTS · 2 COMPARISONS" in summary
    assert "Conflict across eras and presidents" in summary
    assert "How the target mix changes" in summary
    assert "Five target types share one scale" in summary
    assert "inspect six labeled changes in the mix" in summary
    assert "Five aligned panels share one vertical scale" not in summary
    assert "What each target category includes" in summary
    assert '<details class="conflict-category-guide">' in summary
    assert (
        '<summary id="conflict-category-guide-title">'
        "What each target category includes</summary>"
    ) in summary
    assert '<details class="conflict-category-guide" open' not in summary
    assert "Text alternative · target mix by era" not in summary
    assert 'data-conflict-line-picker' in summary
    assert summary.count('data-conflict-line=') == 6
    assert 'data-conflict-line="__all__" aria-pressed="true"' in summary
    assert "All five target lines emphasized" in summary
    assert 'chart.on("plotly_hover"' in summary
    assert 'chart.on("plotly_click"' in summary
    assert "setupContainedPortraitHover" in summary
    assert 'chart.on("plotly_click", event =>' in summary
    assert "Plotly.Fx.hover" in summary
    assert "rightOverflow" in summary
    assert 'event.key !== "Escape"' in summary
    assert "heterogeneous residual" in summary
    target_chart_index = summary.index('data-fig="summary_conflict_targets"')
    category_guide_index = summary.index("What each target category includes")
    portrait_chart_index = summary.index(
        'data-fig="summary_conflict_frame_portraits"'
    )
    assert target_chart_index < category_guide_index < portrait_chart_index
    assert "Zero-sum × partisan × enemy naming" in summary
    assert "BY PRESIDENT · COMPARABLE SPEECHES" in summary
    assert "How presidents frame conflict" in summary
    assert (
        "Compared within State of the Union and annual-message speeches."
    ) in summary
    assert "One common-genre portrait view" not in summary
    assert "HOW DO THE THREE FRAMES COMBINE IN A SHARED SPEECH GENRE?" not in summary
    assert "ANNUAL MESSAGES · 42 OF 45 PRESIDENTS WITH COVERAGE" not in summary
    assert "Enemy naming 1.6%–41.4%" not in summary
    assert (
        "Position carries two annual-message paragraph shares; portrait area carries "
        "the third."
    ) not in summary
    assert "x = zero-sum framing · y = partisan attack" not in summary
    assert "Zero-sum framing runs left to right" not in summary
    assert "Portrait area = enemy naming.<br>" in summary
    assert (
        "All values are shares of eligible State of the Union and annual-message "
        "paragraphs."
    ) in summary
    assert "Annual messages, speaker-audited" in summary
    assert "State of the Union and annual-message paragraphs" in summary
    assert "Portrait border · most named in these messages" in summary
    assert "Ties use a neutral border" not in summary
    assert "Portrait border</th>" not in summary
    assert "Text alternative · all president points and portrait sizes" not in summary
    assert 'class="conflict-treatment"' in summary
    assert "Speaker-audited all eligible paragraphs" in summary
    assert "speaker attribution audit pending" not in summary
    assert "A later audit can replace" not in summary
    assert "fixed reporting era" in summary
    assert "ends in April 2026" in summary
    assert "president-level shares in fixed presidential succession" not in summary
    for old_key in (
        "summary_conflict_enemy", "summary_conflict_zero_sum",
        "summary_conflict_partisan",
        "summary_conflict_early", "summary_conflict_middle",
        "summary_conflict_recent", "summary_enemy_categories",
        "summary_enemy_presidents", "summary_combat",
    ):
        assert f'data-fig="{old_key}"' not in summary
    assert "Three measures, separated" not in summary
    assert "summary_conflict_enemy" not in metrics.CHART_METRICS
    assert "summary_conflict_zero_sum" not in metrics.CHART_METRICS
    assert "summary_conflict_partisan" not in metrics.CHART_METRICS


def test_generated_summary_uses_all_president_temporal_portraits():
    summary = (DATA.parent / "docs" / "summary.html").read_text()
    assert summary.count('data-fig="summary_temporal"') == 1
    assert 'data-temporal-schema="summary-temporal-president-v2"' in summary
    assert 'data-temporal-treatment="all_corpus_document_owner"' in summary
    assert "Presidents can sell tomorrow and yesterday at the same time" in summary
    assert "supported four-year rolling averages" in summary
    assert "10,000-word window floor" in summary
    assert "Two temporal appeals, viewed together" in summary
    assert "portraits use a uniform size" in summary
    assert "self-reference" not in summary
    assert "I/me/my/mine/myself" not in summary
    assert "we/us/our/ours/ourselves" not in summary
    assert "per 10,000 marker words" in summary
    assert "Text alternative · all president temporal points and portrait sizes" not in summary
    assert summary.count('data-register-view="presidents"') >= 2
    assert summary.count('data-register-view="timeline"') >= 2
    assert 'data-register-timeline-figure="summary_temporal_timeline"' in summary
    assert 'aria-label="Tomorrow and yesterday comparison view"' in summary
    assert 'aria-label="Tomorrow and yesterday chart; scroll horizontally on narrow screens"' in summary
    assert summary.count('data-fig="summary_temporal_timeline"') == 0
    assert "Over time: tomorrow and yesterday are separate lines" in summary
    assert "supported four-year rolling average" in summary
    assert "The interactive temporal chart requires JavaScript" in summary
    assert "Separate dictionary audit · founding annual messages only" not in summary
    assert "not the all-speech population" not in summary
    assert "summary-term-audit" not in summary
    assert "Future rises first; nostalgia catches up" not in summary
    assert "fixed 30-year annual-message block" not in summary
    assert "Annual-message/SOTU-only levels" not in summary
    assert "const baseOutlines = Object.freeze" in summary
    assert "const baseWidths = Object.freeze" in summary
    assert "sourceTrace?.customdata || []" in summary
    assert summary.index("const baseWidths = Object.freeze") < summary.index(
        "function applyEraSelection"
    )
    assert (
        f"width:{site.SUMMARY_TEMPORAL_CHART_WIDTH_PX:.0f}px" in summary
    )
    assert "temporal-size-key" not in summary


def test_every_summary_chart_has_a_metric_contract():
    expected = {
        "summary_audience", "summary_medium",
        *site.SUMMARY_CONFLICT_FIGURE_KEYS,
        "summary_temporal", "summary_temporal_timeline",
        "summary_hope_doom_ratio",
    }
    metrics.validate_charts(expected)
    assert metrics.CHART_METRICS["summary_temporal"] == {
        "temporal_portrait", "rate_10k",
    }
    assert metrics.CHART_METRICS["summary_temporal_timeline"] == {
        "temporal_portrait", "rate_10k",
    }
