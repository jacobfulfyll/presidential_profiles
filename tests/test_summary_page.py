from pathlib import Path

import numpy as np
import pandas as pd

from presidential_profiles import metrics, site


DATA = Path(__file__).parents[1] / "data"


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
    assert "All nine eras highlighted" in rendered


def test_enemy_category_shares_cover_every_era_and_sum_to_100_percent():
    frame = pd.read_parquet(DATA / "combat" / "adversary_mix.parquet")
    frame = frame[frame.grain.eq("era")].copy()
    figure = site.fig_summary_enemy_categories({"adversary_categories": frame})
    assert len(figure.data) == 5
    assert all(len(trace.y) == 9 for trace in figure.data)
    totals = np.sum([np.asarray(trace.y, dtype=float) for trace in figure.data], axis=0)
    assert np.allclose(totals, 100)


def test_summary_combat_view_has_no_confidence_whiskers():
    frame = pd.read_parquet(DATA / "combat" / "combativeness.parquet")
    figure = site.fig_summary_combat({
        "combat": frame[frame.treatment.eq("sotu_only")].copy()
    })
    assert all(trace.error_y.visible is not True for trace in figure.data)


def test_every_summary_chart_has_a_metric_contract():
    expected = {
        "summary_communication", "summary_enemy_categories",
        "summary_enemy_presidents", "summary_combat", "summary_temporal",
        "summary_hope_doom_ratio",
    }
    metrics.validate_charts(expected)
