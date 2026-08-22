import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import corpus, metrics, site, trends


DATA = Path(__file__).parents[1] / "data"


@pytest.fixture(scope="module")
def temporal_contract():
    return site.build_summary_temporal_president_contract(
        pd.read_parquet(DATA / "speech_markers.parquet"),
        pd.read_parquet(DATA / "speech_stats.parquet"),
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
    stats = pd.DataFrame({
        "doc_name": ["a", "b"],
        "president": ["Example President", "Example President"],
        "year": [1900, 1901],
        "i_count": [3, 7],
        "we_count": [9, 1],
    })
    return markers, stats, speeches


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
    assert "independent 100% stacked bars, palettes, and legends" in summary
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


def test_temporal_president_contract_reconciles_exact_receipts(temporal_contract):
    rows = temporal_contract["rows"]
    assert temporal_contract["schema_version"] == "summary-temporal-president-v1"
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
    assert np.allclose(
        rows["self_reference"],
        rows["n_i_pronouns"] / (
            rows["n_i_pronouns"] + rows["n_we_pronouns"]
        ),
    )
    assert rows["support_status"].value_counts().to_dict() == {
        "observed": 42,
        "thin_record": 3,
    }
    assert set(rows.loc[rows["support_status"].eq("thin_record"), "president"]) == {
        "William Harrison", "James A. Garfield", "Zachary Taylor",
    }


def test_temporal_contract_pools_counts_instead_of_averaging_speech_rates():
    markers, stats, speeches = _minimal_temporal_sources()
    contract = site.build_summary_temporal_president_contract(
        markers, stats, speeches, expected_presidents=None
    )
    row = contract["rows"].iloc[0]
    assert row.future == pytest.approx(100)
    assert row.nostalgia == pytest.approx(200)
    assert row.self_reference == pytest.approx(.5)
    assert row.n_rate_words == 1_000
    assert row.n_future_matches == 10
    assert row.n_nostalgia_matches == 20


def test_temporal_contract_rejects_key_metadata_and_receipt_drift():
    markers, stats, speeches = _minimal_temporal_sources()

    with pytest.raises(ValueError, match="key sets differ"):
        site.build_summary_temporal_president_contract(
            markers, stats.iloc[:1], speeches, expected_presidents=None
        )

    duplicate_markers = pd.concat([markers, markers.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="one named row per speech"):
        site.build_summary_temporal_president_contract(
            duplicate_markers, stats, speeches, expected_presidents=None
        )

    wrong_president = stats.copy()
    wrong_president.loc[0, "president"] = "Wrong President"
    with pytest.raises(ValueError, match="president metadata differs"):
        site.build_summary_temporal_president_contract(
            markers, wrong_president, speeches, expected_presidents=None
        )

    wrong_year = stats.copy()
    wrong_year.loc[0, "year"] = 1902
    with pytest.raises(ValueError, match="year metadata differs"):
        site.build_summary_temporal_president_contract(
            markers, wrong_year, speeches, expected_presidents=None
        )

    fractional_count = markers.copy()
    fractional_count["future"] = fractional_count["future"].astype(float)
    fractional_count.loc[0, "future"] = 1.5
    with pytest.raises(ValueError, match="not integral"):
        site.build_summary_temporal_president_contract(
            fractional_count, stats, speeches, expected_presidents=None
        )

    impossible_matches = markers.copy()
    impossible_matches.loc[0, ["n_words", "future"]] = [0, 1]
    with pytest.raises(ValueError, match="marker-word denominator"):
        site.build_summary_temporal_president_contract(
            impossible_matches, stats, speeches, expected_presidents=None
        )

    null_president = speeches.copy()
    null_president.loc[0, "president"] = None
    with pytest.raises(ValueError, match="named president"):
        site.build_summary_temporal_president_contract(
            markers, stats, null_president, expected_presidents=None
        )

    fractional_year = speeches.copy()
    fractional_year["year"] = fractional_year["year"].astype(float)
    fractional_year.loc[0, "year"] = 1900.5
    with pytest.raises(ValueError, match="finite integers"):
        site.build_summary_temporal_president_contract(
            markers, stats, fractional_year, expected_presidents=None
        )


def test_temporal_portrait_scatter_uses_future_nostalgia_and_area(
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
    assert np.allclose(
        diameters ** 2 / site.SUMMARY_TEMPORAL_PORTRAIT_REFERENCE_PX ** 2,
        plotted["self_reference"],
    )
    assert np.all(np.diff(diameters) <= 0)
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
    assert "portrait area" in trace.hovertemplate
    assert "singular-family forms" in trace.hovertemplate
    assert "plural-family forms" in trace.hovertemplate
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
    image_shares = image_rows["self_reference"].to_numpy(dtype=float)
    expected_pixels = (
        site.SUMMARY_TEMPORAL_PORTRAIT_REFERENCE_PX * np.sqrt(image_shares)
    )
    assert np.allclose(visible_widths, expected_pixels)
    assert np.allclose(visible_heights, expected_pixels)
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


def test_temporal_portrait_handles_unavailable_pronoun_denominator():
    speeches = pd.DataFrame({
        "doc_name": ["a"], "president": ["Example President"],
        "year": [1900], "date": pd.to_datetime(["1900-01-01"]),
    })
    markers = pd.DataFrame({
        "doc_name": ["a"], "president": ["Example President"],
        "year": [1900], "n_words": [100], "future": [1], "nostalgia": [1],
    })
    stats = pd.DataFrame({
        "doc_name": ["a"], "president": ["Example President"],
        "year": [1900], "i_count": [0], "we_count": [0],
    })
    contract = site.build_summary_temporal_president_contract(
        markers, stats, speeches, expected_presidents=None
    )
    assert contract["rows"].iloc[0].support_status == "not_available"
    figure = site.fig_summary_temporal(
        contract, {"Example President": "data:image/png;base64,AA=="}
    )
    assert len(figure.layout.images) == 0
    assert any("N/A" in str(annotation.text) for annotation in figure.layout.annotations)
    rendered = site._summary_temporal_portrait_html(contract)
    assert "N/A" in rendered
    assert "nan%" not in rendered
    assert "data-era-choice" not in rendered


def test_temporal_portrait_uses_a_hollow_locator_for_true_zero_share():
    speeches = pd.DataFrame({
        "doc_name": ["a"], "president": ["Example President"],
        "year": [1900], "date": pd.to_datetime(["1900-01-01"]),
    })
    markers = pd.DataFrame({
        "doc_name": ["a"], "president": ["Example President"],
        "year": [1900], "n_words": [100], "future": [1], "nostalgia": [1],
    })
    stats = pd.DataFrame({
        "doc_name": ["a"], "president": ["Example President"],
        "year": [1900], "i_count": [0], "we_count": [4],
    })
    contract = site.build_summary_temporal_president_contract(
        markers, stats, speeches, expected_presidents=None
    )
    figure = site.fig_summary_temporal(
        contract, {"Example President": "data:image/png;base64,AA=="}
    )
    trace = figure.data[0]
    assert float(contract["rows"].iloc[0].self_reference) == 0
    assert float(trace.marker.size[0]) == site.SUMMARY_TEMPORAL_PORTRAIT_ZERO_PX
    assert str(trace.marker.symbol[0]) == "circle-open"
    assert len(figure.layout.images) == 0
    assert "hollow locator, not portrait area" in str(trace.customdata[0][11])
    rendered = site._summary_temporal_portrait_html(contract)
    assert "true 0% uses a 16px hollow locator" in rendered


def test_all_president_era_controls_start_with_every_era_selected():
    scores = pd.DataFrame({
        "first_year": [1789], "last_year": [1797], "mechanism": [40.0],
        "hype": [2.0], "n_speeches": [10],
    }, index=["George Washington"])
    rendered = site._procedural_era_html(scores)
    assert rendered.count("data-era-choice") == 9
    assert rendered.count(" checked") == 9
    assert rendered.count('class="era-chip"') == 9
    assert rendered.count("data-era-preset") == 6
    assert "data-era-all" in rendered
    assert "data-era-clear" in rendered
    assert "All nine eras highlighted; all plotted presidents visible" in rendered
    figure = site.fig_procedural_eras(
        scores, {"George Washington": "data:image/png;base64,AA=="}
    )
    marker = json.loads(figure.to_json())["data"][0]["marker"]
    assert isinstance(marker["opacity"], list)
    assert isinstance(marker["color"], list)
    assert isinstance(marker["line"]["width"], list)
    assert isinstance(marker["line"]["color"], list)


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
        if trace.type == "scatter" and trace.mode == "lines+markers+text"
    ]
    assert len(lines) == 5
    for trace in lines:
        assert tuple(int(value) for value in trace.x) == tuple(range(9))
        assert tuple(str(row[0]) for row in trace.customdata) == (
            tuple(label for label, _start, _end in trends.ERAS)
        )
    assert tuple(figure.layout.xaxis5.tickvals) == tuple(range(9))
    assert len(set(figure.layout.xaxis5.ticktext)) == 9
    assert "Founding<br>1789–1815" == figure.layout.xaxis5.ticktext[0]
    assert "Present<br>2017–2026" == figure.layout.xaxis5.ticktext[-1]
    assert not figure.layout.xaxis.showticklabels
    assert figure.layout.xaxis5.showticklabels


def test_conflict_target_graph_is_five_aligned_shared_scale_era_panels():
    frame = pd.read_parquet(
        DATA / "combat" / "target_mix_by_era_speaker_audited_v1.parquet"
    )
    contract = site.build_summary_conflict_target_contract(frame)
    ordered = contract["rows"]
    figure = site.fig_summary_conflict_targets(contract)
    lines = [
        trace for trace in figure.data
        if trace.type == "scatter" and trace.mode == "lines+markers+text"
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
    assert [trace.xaxis for trace in lines] == ["x", "x2", "x3", "x4", "x5"]
    assert [trace.yaxis for trace in lines] == ["y", "y2", "y3", "y4", "y5"]

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
    for axis_name in ("yaxis", "yaxis2", "yaxis3", "yaxis4", "yaxis5"):
        axis = getattr(figure.layout, axis_name)
        assert tuple(axis.range) == (0, 60)
        assert axis.ticksuffix == "%"
    assert figure.layout.xaxis5.title.text == "fixed reporting eras · categorical order"
    assert "share of adversarial" in figure.layout.yaxis3.title.text
    assert figure.layout.hovermode == "x unified"
    assert figure.layout.hoversubplots == "axis"
    assert figure.layout.showlegend is False
    assert all(len(trace.text) == 9 for trace in lines)


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
    assert "enemy naming %{customdata[3]:.1f}% · portrait area" in (
        trace.hovertemplate
    )
    assert all(
        image.name.startswith("conflict-portrait::")
        for image in figure.layout.images
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
    ) == 5
    assert tuple(targets.layout.xaxis5.tickvals) == tuple(range(9))

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
    assert "Enemy naming N/A" in panel
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
    for key in expected_keys:
        assert summary.count(f'data-fig="{key}"') == 1
    assert "9 ERAS · 45 PRESIDENTS · 2 COMPARISONS" in summary
    assert "Conflict across eras and presidents" in summary
    assert "How the target mix changes" in summary
    assert "Five aligned panels share one vertical scale" in summary
    assert "What each target category includes" in summary
    assert "Text alternative · target mix by era" in summary
    assert "heterogeneous residual" in summary
    target_chart_index = summary.index('data-fig="summary_conflict_targets"')
    category_guide_index = summary.index("What each target category includes")
    target_table_index = summary.index(
        "Text alternative · target mix by era"
    )
    portrait_chart_index = summary.index(
        'data-fig="summary_conflict_frame_portraits"'
    )
    assert (
        target_chart_index
        < category_guide_index
        < target_table_index
        < portrait_chart_index
    )
    assert "Zero-sum × partisan × enemy naming" in summary
    assert "portrait area = enemy naming" in summary
    assert "Text alternative · all president points and portrait sizes" in summary
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
    assert 'data-temporal-schema="summary-temporal-president-v1"' in summary
    assert 'data-temporal-treatment="all_corpus_document_owner"' in summary
    assert "Presidents can sell tomorrow and yesterday at the same time" in summary
    assert "Tomorrow × yesterday × self-reference" in summary
    assert "portrait area = singular share" in summary
    assert "area—not diameter—is proportional" in summary
    assert "true 0% uses a 16px hollow locator" in summary
    assert "I/me/my/mine/myself" in summary
    assert "we/us/our/ours/ourselves" in summary
    assert "per 10,000 marker words" in summary
    assert "Text alternative · all president temporal points and portrait sizes" in summary
    assert "ALL 45 PRESIDENTS · ALL AVAILABLE CORPUS SPEECHES" in summary
    assert "Separate dictionary audit · founding annual messages only" in summary
    assert "not the all-speech population" in summary
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
    assert "grid-template-columns:repeat(3,minmax(0,1fr))" in summary


def test_every_summary_chart_has_a_metric_contract():
    expected = {
        "summary_audience", "summary_medium",
        *site.SUMMARY_CONFLICT_FIGURE_KEYS, "summary_temporal",
        "summary_hope_doom_ratio",
    }
    metrics.validate_charts(expected)
    assert metrics.CHART_METRICS["summary_temporal"] == {
        "temporal_portrait", "rate_10k",
    }
