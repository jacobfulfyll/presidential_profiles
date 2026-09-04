"""Focused contracts for the generated Explore v2 page and renderer."""
from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from presidential_profiles import explore_assets, explore_projection, explorer


def test_information_architecture_and_default_state(explore_projection_bundle):
    page = explorer.render_page(explore_projection_bundle)

    assert page.index("Explore language and topics") < page.index("Choose series")
    assert page.index("Choose series") < page.index("Guided comparisons")
    assert page.index("Guided comparisons") < page.index('<h2 id="trends-heading">Trends</h2>')
    assert page.index('<h2 id="trends-heading">Trends</h2>') < page.index("Exact values and support")
    assert page.index("Exact values and support") < page.index("Data and methods")
    assert "3 of 6 selected" in page
    assert "tariff, freedom, border" in page
    assert "plotly" not in page.lower()


def test_page_refuses_broad_issue_display_name_registry_drift(
    explore_projection_bundle,
):
    index = copy.deepcopy(explore_projection_bundle.index)
    index["catalog"]["broad_issues"][-1]["label"] = "Unregistered label"
    drifted = replace(explore_projection_bundle, index=index)

    with pytest.raises(explore_projection.ExploreProjectionError, match="registry"):
        explorer.render_page(drifted)


def test_word_family_choice_is_primary_and_chart_options_are_retired(
    explore_projection_bundle,
):
    page = explorer.render_page(explore_projection_bundle)

    word_start = page.index("Add a word or phrase")
    assert word_start < page.index("Exact form") < page.index("Group word forms")
    assert "Chart options" not in page
    assert "Combine all selected words" not in page
    assert "Clear era bands" not in page
    assert "Historical context" in page


def test_no_javascript_fallback_keeps_chart_tables_and_downloads(
    explore_projection_bundle,
):
    page = explorer.render_page(explore_projection_bundle)

    assert "Interactive selection requires JavaScript" in page
    assert "fallback-chart-title" in page
    assert page.count('<table class="exact-table">') == 3
    assert "Published uncertainty" in page
    assert f'explorer/{explore_projection.DEFAULT_VALUES_FILE}' in page
    assert f'explorer/{explore_projection.TOPIC_VALUES_FILE}' in page
    assert f'explorer/{explore_projection.SERIES_CATALOG_FILE}' in page
    assert page.count("data-hydration-control disabled") >= 9


def test_measure_scope_and_uncertainty_language_are_permanent(
    explore_projection_bundle,
):
    page = explorer.render_page(explore_projection_bundle)

    assert "complete source-document transcripts" in page
    assert "Broad issues use deterministic-model paragraph shares" in page
    assert "Detailed topics use exploratory AI-labeled paragraph shares" in page
    assert "Explore is not filtered to actual-president paragraph speakers" in page
    assert "Missing bounds are unknown, never zero" in page
    assert "†" in page and "‡" in page and "◇" in page


def test_external_assets_are_written_with_the_page(
    tmp_path, explore_projection_bundle
):
    output = explorer.write_page(explore_projection_bundle, tmp_path)

    assert output == tmp_path / "explorer.html"
    assert (tmp_path / "assets" / explore_assets.CSS_FILE).read_text().strip()
    assert (tmp_path / "assets" / explore_assets.JS_FILE).read_text().strip()
    assert f'assets/{explore_assets.CSS_FILE}' in output.read_text()
    assert f'assets/{explore_assets.JS_FILE}' in output.read_text()


def test_renderer_uses_native_search_and_collapsible_governed_groups():
    script = explore_assets.EXPLORE_JS

    assert 'input.type = "checkbox"' in script
    assert 'document.createElement("details")' in script
    assert '"Broad issues · deterministic model"' in script
    assert '"Detailed topics · AI labels (50)"' in script
    assert '"Case-sensitive named acronyms"' in script
    assert "catalogMatches" in script
    assert "word.startsWith(token)" in script
    assert "savedDisclosureState" in script
    assert "recent" not in script.lower()


def test_renderer_has_per_series_grouping_and_strict_term_parsing():
    script = explore_assets.EXPLORE_JS

    assert "function parseTerm(" in script
    assert ".replace(/[’‘]/g, \"'\")" in script
    assert "Use no more than two indexed words" in script
    assert "hasUnsupportedLetter" in script
    assert "normalizeGrouped" in script
    assert '"Use exact form" : "Group word forms"' in script
    assert "familyFormCount" in script
    assert "Position ${position + 1}" in script


def test_renderer_url_v2_history_and_legacy_migration_are_explicit():
    script = explore_assets.EXPLORE_JS

    assert 'url.searchParams.set("state", "2")' in script
    assert 'url.searchParams.append("s", encodedSelection(item))' in script
    assert 'url.searchParams.set("uncertainty", "off")' in script
    assert 'url.searchParams.set("context", contextKey)' in script
    assert 'history[mode === "replace" ? "replaceState" : "pushState"]' in script
    assert 'addEventListener("popstate"' in script
    assert 'type==="word"' in script
    assert 'type==="topic"' in script
    assert 'type==="entity"' in script
    assert "combined-word state retired; individual series restored" in script
    assert "multiple era highlights reduced to the first recognized historical context" in script


def test_pointer_keyboard_pin_and_live_announcement_contracts():
    script = explore_assets.EXPLORE_JS

    for key in ("ArrowLeft", "ArrowRight", "Home", "End", "Enter", "Escape"):
        assert f'event.key === "{key}"' in script
    assert 'event.key === " "' in script
    assert 'button.setAttribute("aria-pressed"' in script
    assert 'hit.addEventListener("pointermove"' in script
    assert 'hit.addEventListener("click"' in script
    assert 'button.addEventListener("click"' in script
    assert "focusId || hoverId || pinnedId" in script
    assert '.catalog-row[hidden]' in explore_assets.EXPLORE_CSS
    assert 'byId("chart-live").textContent' in script


def test_renderer_never_derives_measures_or_support_states():
    script = explore_assets.EXPLORE_JS

    assert "rate_per_10k_x10000" not in script
    assert "function ciStatus" not in script
    assert "function supportStatus" not in script
    assert "n_paragraphs /" not in script
    assert "numerator_count /" not in script
    assert "own_peak" not in script
    assert "rolling(" not in script
    assert "combineWords" not in script
    assert "INDEX.fixed_point_scale" in script
    assert "payload.series[id]" in script


def test_uncertainty_toggle_preserves_status_and_exact_paths():
    script = explore_assets.EXPLORE_JS

    assert "if(uncertainty &&" in script
    assert "row.interval_unresolvable" in script
    assert 'row.ci_status!=="ok"' in script
    assert "Not published for this measure" in script
    assert "renderExactValues" in script
    assert 'link.download="presidential-profiles-explore.csv"' in script


def test_accessibility_and_responsive_media_contracts():
    css = explore_assets.EXPLORE_CSS
    page = explorer.render_page.__doc__

    assert "min-height:44px" in css
    assert "outline:3px" in css
    assert "@media(max-width:899px)" in css
    assert "@media(max-width:639px)" in css
    assert "@media(max-width:480px)" in css
    assert "prefers-reduced-motion:reduce" in css
    assert "forced-colors:active" in css
    assert "max-height:min(40vh,18rem)" in css
    assert "complete default no-JavaScript path" in page


def test_failure_paths_leave_fallback_and_existing_selection_intact():
    script = explore_assets.EXPLORE_JS

    assert "fetchCache.delete(cacheKey)" in script
    assert "Current selections were left unchanged" in script
    assert "the prior selection was retained" in script
    assert "Exact-form search still works" in script
    assert "restoreSerial" in script
    assert "if(serial!==restoreSerial||epoch!==selectionEpoch)return" in script
    assert "The default fallback remains available" in script
