"""Behavioral and security contracts for the narrative Compare V3 page."""
from __future__ import annotations

from copy import deepcopy
import html
import json
from pathlib import Path
import re

import pytest

from presidential_profiles import ai_labels
from presidential_profiles import compare_projection
from presidential_profiles import compare_site
from presidential_profiles import profiles
from presidential_profiles import profiles_site
from presidential_profiles import speaker_topic_network
from presidential_profiles import topic_quality
from presidential_profiles.compare_assets import (
    AGENDA_COMPARISON_CSS,
    AGENDA_COMPARISON_JS,
    COMPARE_V3_CSS,
    COMPARE_V3_JS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = REPO_ROOT / "docs" / "data" / "presidents"
HOSTILE = '</script><img src=x onerror="alert(1)">'


@pytest.fixture(scope="module")
def public_profiles() -> dict[str, dict]:
    return {
        president: json.loads(
            (PROFILE_DIR / f"{profiles.slug(president)}.json").read_text(encoding="utf-8")
        )
        for president in profiles.PARTY
    }


@pytest.fixture(scope="module")
def compare_bundle():
    data = profiles.build_profile_data()
    data["ai"] = ai_labels.build_ai_data()
    data["feature_neighbors"] = profiles.feature_neighbors(data, data["ai"])
    display = topic_quality.display_issues(data["issue_meta"]["issues"])
    views = profiles_site.build_profile_view_models(data, display, require_complete=True)
    network = speaker_topic_network.load_network_bundle()
    return compare_projection.build_projection(network, data, views, display)


@pytest.fixture(scope="module")
def compare_payload(public_profiles, compare_bundle) -> dict:
    return compare_site.comparison_payload(
        public_profiles, require_complete=True, projection=compare_bundle
    )


@pytest.fixture(scope="module")
def compare_page(compare_payload, compare_bundle) -> str:
    return compare_site.render_compare_page(compare_payload, compare_bundle)


def test_payload_is_v3_compact_and_uses_canonical_ids(compare_payload):
    assert compare_payload["schema_version"] == "president-comparison-page-v3"
    assert compare_payload["profile_schema_version"] == "president-profile-v3"
    assert compare_payload["order"] == [profiles.slug(name) for name in profiles.PARTY]
    assert compare_payload["defaults"] == ["abraham-lincoln", "franklin-d-roosevelt"]
    assert len(compare_payload["presidents"]) == 45
    assert compare_payload["presidents"]["william-harrison"]["display_name"] == "William Henry Harrison"
    assert compare_payload["presidents"]["william-taft"]["display_name"] == "William Howard Taft"
    assert compare_payload["presidents"]["abraham-lincoln"]["short_name"] == "Lincoln"
    assert compare_payload["presidents"]["franklin-d-roosevelt"]["short_name"] == "F.D. Roosevelt"
    assert compare_payload["presidents"]["theodore-roosevelt"]["short_name"] == "T. Roosevelt"
    encoded = json.dumps(compare_payload, separators=(",", ":"), allow_nan=False).encode()
    assert len(encoded) <= 150_000
    assert "topic_ids" not in compare_payload and "default_topic_id" not in compare_payload
    assert "legacy_issue_csv" not in compare_payload["urls"]
    for record in compare_payload["presidents"].values():
        for removed in (
            "legacy_issue_attention", "issue_evidence", "neighbors", "agenda",
            "proposal_values", "speech_types", "adversaries", "signature_speeches",
        ):
            assert removed not in record


def test_payload_separates_source_document_and_actual_speaker_support(compare_payload):
    lincoln = compare_payload["presidents"]["abraham-lincoln"]
    garfield = compare_payload["presidents"]["james-a-garfield"]
    assert lincoln["source_document_speech_count"] == 15
    assert lincoln["actual_speaker_appearance_count"] == 15
    assert lincoln["source_document_support_state"] == "supported"
    assert lincoln["actual_speaker_support_state"] == "supported"
    assert garfield["source_document_support_state"] == "thin"
    assert garfield["actual_speaker_support_state"] == "thin"


def test_measure_catalogs_and_values_match_profile_v3(
    public_profiles, compare_payload
):
    expected = {
        "corpus": profiles_site.LEGACY_MEASURE_SPECS,
        "ai": profiles_site.AI_MEASURE_SPECS,
    }
    for layer, specs in expected.items():
        assert [row["key"] for row in compare_payload["measure_catalogs"][layer]] == [
            row["key"] for row in specs
        ]
        assert [row["label"] for row in compare_payload["measure_catalogs"][layer]] == [
            row["label"] for row in specs
        ]
    for name in ("Abraham Lincoln", "Franklin D. Roosevelt", "William Harrison"):
        president_id = profiles.slug(name)
        for output, source_layer in (("corpus", "rhetorical_radar"), ("ai", "ai_radar")):
            source = profiles_site.profile_measure_rows(public_profiles[name], source_layer)
            projected = compare_payload["presidents"][president_id]["measures"][output]
            assert len(projected) == len(source)
            for expected_row, actual in zip(source, projected, strict=True):
                assert actual["absolute"] == expected_row["absolute"]
                assert actual["percentile"] == expected_row["percentile"]
    assert all(
        row["absolute"] is not None and row["percentile"] is None
        for layer in ("corpus", "ai")
        for row in compare_payload["presidents"]["william-harrison"]["measures"][layer]
    )


def test_missing_ai_and_supported_null_rank_remain_unavailable(public_profiles):
    missing = deepcopy(public_profiles["Abraham Lincoln"])
    missing["ai"] = None
    payload = compare_site.comparison_payload({"Abraham Lincoln": missing})
    assert all(
        row == {
            "absolute": None, "absolute_display": "N/A", "percentile": None,
            "rank_display": "N/A · AI labels unavailable",
        }
        for row in payload["presidents"]["abraham-lincoln"]["measures"]["ai"]
    )
    null_rank = deepcopy(public_profiles["Abraham Lincoln"])
    null_rank["rhetorical_radar"][0]["percentile"] = None
    payload = compare_site.comparison_payload({"Abraham Lincoln": null_rank})
    assert payload["presidents"]["abraham-lincoln"]["measures"]["corpus"][0]["rank_display"] == "N/A · Not ranked"


def test_url_normalization_never_manufactures_invalid_third_president():
    order = list(profiles.PARTY)
    by_slug = {profiles.slug(name): name for name in order}
    assert compare_site.normalize_selection(order, by_slug, {}) == (
        ["Abraham Lincoln", "Franklin D. Roosevelt"], False
    )
    assert compare_site.normalize_selection(order, by_slug, {"a": "abraham-lincoln"}) == (
        ["Abraham Lincoln", "Andrew Johnson"], False
    )
    assert compare_site.normalize_selection(order, by_slug, {"a": profiles.slug(order[-1])}) == (
        [order[-1], order[-2]], False
    )
    selected, corrected = compare_site.normalize_selection(
        order, by_slug,
        {"a": "not-real", "b": "abraham-lincoln", "c": "abraham-lincoln"},
    )
    assert selected == ["Abraham Lincoln", "Franklin D. Roosevelt"]
    assert corrected is True
    selected, corrected = compare_site.normalize_selection(
        order, by_slug,
        {"a": "abraham-lincoln", "b": "franklin-d-roosevelt", "c": "barack-obama"},
    )
    assert selected == ["Abraham Lincoln", "Franklin D. Roosevelt", "Barack Obama"]
    assert corrected is False


def test_page_has_only_the_three_narrative_sections_and_simple_controls(compare_page):
    assert compare_page.count("<h1>") == 1
    assert re.findall(r"<h2>(.*?)</h2>", compare_page) == [
        "How do the records differ in rhetorical form?",
        "Which topics are most prominent in the selected records?",
        "What supports this comparison?",
    ]
    assert '<legend>Compare presidents</legend>' in compare_page
    assert compare_page.count("<select") == 3
    assert '<option value="" selected>None</option>' in compare_page
    for removed in (
        "Swap A/B", "Add third president", "Remove third president", "Copy link",
        "Reset", "Nearest-neighbor", "Proposal-versus-values", "Speech-type composition",
    ):
        assert removed not in compare_page
    assert '<nav class="compare-nav"' in compare_page
    assert ">Rhetoric</a>" in compare_page
    assert ">Agenda</a>" in compare_page
    assert ">Evidence</a>" in compare_page


def test_focused_topic_union_gives_every_selected_record_three_entries(compare_bundle):
    index = compare_bundle.agenda_index
    focused_topic_limit = compare_site.FOCUSED_TOPIC_LIMIT
    assert focused_topic_limit == 3
    governed_order = {
        topic["topic_id"]: order
        for order, topic in enumerate(index["level1_topics"])
    }
    for selected in (
        ["abraham-lincoln", "franklin-d-roosevelt"],
        ["abraham-lincoln", "franklin-d-roosevelt", "george-washington"],
    ):
        focused = compare_site._focused_broad_topics(index, selected)
        assert 3 <= len(focused) <= focused_topic_limit * len(selected)
        assert [governed_order[topic["topic_id"]] for topic, _ in focused] == sorted(
            governed_order[topic["topic_id"]] for topic, _ in focused
        )
        assert [sum(slot in slots for _, slots in focused) for slot in range(len(selected))] == [
            focused_topic_limit
        ] * len(selected)


def test_server_fallback_uses_only_the_focused_broad_agenda(compare_page):
    assert 'data-agenda-tab="' not in compare_page
    assert compare_page.count('class="agenda-row broad-row"') == 5
    assert compare_page.count('class="agenda-row"') == 0
    assert compare_page.count('class="agenda-reason"') == 5
    assert "War &amp; Military Affairs" in compare_page
    assert 'class="agenda-row fine-row"' not in compare_page
    assert 'id="agenda-broad"' in compare_page
    assert 'id="agenda-topic-focus"' not in compare_page
    assert "rank among the top three" in compare_page
    assert "bring each topic into view" in compare_page
    assert "A · Lincoln" in compare_page
    assert "B · F.D. Roosevelt" in compare_page
    assert "select a value for its exact counts and one example" in compare_page
    assert "CorEx" not in compare_page
    assert "fine topic" not in compare_page.lower()
    assert "multi-label shares are non-additive" not in compare_page
    assert "Method bridge" not in compare_page
    assert "method bridge" not in AGENDA_COMPARISON_JS.lower()
    assert '.agenda-panel[hidden]' not in compare_page


def test_rhetoric_exact_path_and_lazy_plotly_contract(compare_page):
    assert compare_page.count('data-rhetoric-tab="') == 2
    assert compare_page.count('class="exact-table"') == 2
    assert "Effective topic breadth" in compare_page
    assert "Zero-sum framing" in compare_page
    assert "Hope" in compare_page and "Partisan attack" in compare_page
    assert '<script src="assets/plotly-3.0.1.min.js">' not in compare_page
    assert '"plotly": "assets/plotly-3.0.1.min.js"' in compare_page
    assert 'assets/compare-v3.css?v=5' in compare_page
    assert 'assets/compare-v3.js?v=5' in compare_page
    assert "loadPlotly" in COMPARE_V3_JS
    assert 'rootMargin: "600px 0px"' in COMPARE_V3_JS
    assert "Math.min(112, Math.max(86, target.clientWidth * .29))" in COMPARE_V3_JS
    assert "window.Plotly.Plots.resize(target)" in COMPARE_V3_JS


def test_phone_radar_preserves_a_220px_plot_and_redraws_safely():
    js = COMPARE_V3_JS

    assert 'matchMedia("(max-width: 760px)")' in js
    assert "Math.max(220, Math.min(430, target.clientWidth - 64))" in js
    assert "Math.ceil(compactDiameter + topMargin + bottomMargin)" in js
    assert "height: radarHeight" in js
    assert 'target.dataset.radarRevision = String(revision)' in js
    assert 'target.closest("[hidden]")' in js
    assert '!compactRadarMedia.matches && (!event || event.type !== "change")' in js
    assert 'window.addEventListener("resize", scheduleRadarRedraw' in js
    assert 'window.addEventListener("orientationchange", scheduleRadarRedraw' in js
    assert 'compactRadarMedia.addEventListener("change", scheduleRadarRedraw)' in js
    mobile_css = COMPARE_V3_CSS.split("@media (max-width: 760px)", 1)[1]
    assert ".radar-chart" in mobile_css
    assert "min-height: 328px" in mobile_css


def test_evidence_is_comparison_wide_and_population_labeled(compare_page):
    assert compare_page.count('class="footprint-card') == 2
    assert compare_page.count('class="evidence-family"') == 4
    for label in (
        "Adversarial entities", "Presidential invocations", "Distinctive vocabulary",
        "Signature speeches", "source-document paragraphs", "cosine score",
    ):
        assert label in compare_page
    assert "z-score" not in compare_page
    assert "Evidence receipts" not in COMPARE_V3_JS
    assert "Legacy invocation" not in compare_page
    assert "Full actual-speaker topic CSV" in compare_page
    assert "Full original CorEx topic CSV" not in compare_page


def test_controller_owns_canonical_url_history_and_download_state():
    js = COMPARE_V3_JS
    for state_name in ("selectedPresidentIds", "activeRhetoricLayer"):
        assert state_name in js
    assert "fineParentTopicId" not in js
    assert "history.pushState" in js and "history.replaceState" in js
    assert 'window.addEventListener("popstate"' in js
    assert "url.hash" in js
    assert 'params.set("c"' in js
    assert 'params.set("topic"' not in js
    assert 'params.set("agenda"' not in js
    assert 'params.get("agenda")' not in js
    assert "if (!slots.includes(key)) corrected = true" in js
    assert "selected comparison" not in js.lower()
    assert "president-comparison-v3" in js
    assert "Promise.all(selected().map" in js
    assert "profile_data_url" in js
    assert "full speaker-topic-network" not in js


def test_agenda_renderer_is_safe_semantic_and_keyboard_operable():
    js = AGENDA_COMPARISON_JS
    assert "document.createElement" in js and "textContent" in js and "replaceChildren" in js
    assert ".innerHTML" not in js and "innerHTML" not in js
    assert "insertAdjacentHTML" not in js
    assert '"Escape"' in js
    assert "focus" in js
    assert "topic-focus-button" not in js
    assert "renderFocusedTopic" not in js
    assert "detail.hidden = true" in js and "detail.hidden = false" in js
    assert "agenda-plot-mark" in js
    assert "FOCUSED_TOPIC_LIMIT = 3" in js
    assert "focusedReasons" in js
    assert "Show all 17 topics" not in js
    assert "Show focused topics" not in js
    assert 'return "Top 3 for " + names.join(" + ")' in js
    assert "original CorEx" not in js
    assert "renderCorex" not in js
    assert "method bridge" not in js.lower()
    assert "fine-heatmap" not in js
    assert "speaker_paragraph_share" in js
    assert "topic_paragraph_count" in js
    assert "eligible_president_paragraph_count" in js
    assert "topic-bearing appearances" in js
    assert "One topic example" in js
    assert 'row.selection_role === "first"' in js
    assert "receipts.forEach" not in js
    for forbidden in ("composite score", "similarity", "importance rank"):
        assert forbidden not in js.lower()


def test_css_encodes_slot_identity_support_and_accessibility():
    css = (COMPARE_V3_CSS + AGENDA_COMPARISON_CSS).lower()
    assert "#2a78d6" in css and "#9b6200" in css and "#008300" in css
    assert "min-height: 44px" in css
    assert "border-bottom-style: dashed" in css
    assert "border-bottom-style: dotted" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert "@media (forced-colors: active)" in css
    assert "@media (max-width: 760px)" in css and "@media (max-width: 430px)" in css
    assert "overflow-wrap: anywhere" in css
    assert "ellipsis" not in css
    assert ".agenda-row-plot" in AGENDA_COMPARISON_CSS
    assert ".agenda-row-values" in AGENDA_COMPARISON_CSS
    assert ".agenda-view-controls" not in AGENDA_COMPARISON_CSS
    assert ".agenda-view-toggle" not in AGENDA_COMPARISON_CSS
    assert "border: 1px solid transparent" in AGENDA_COMPARISON_CSS
    assert ".mark-president" in AGENDA_COMPARISON_CSS
    assert ".rhetoric-tabs" in COMPARE_V3_CSS
    assert "width: max-content" in COMPARE_V3_CSS
    assert ".method-lanes" not in AGENDA_COMPARISON_CSS
    assert "@media (forced-colors: active)" in AGENDA_COMPARISON_CSS


def test_page_and_assets_meet_raw_budgets(compare_page, compare_payload):
    assert len(compare_page.encode()) <= 230_000
    assert len(json.dumps(compare_payload, separators=(",", ":")).encode()) <= 150_000
    assert len(COMPARE_V3_JS.encode()) <= 45_000
    assert len(AGENDA_COMPARISON_JS.encode()) <= 45_000
    assert len((COMPARE_V3_CSS + AGENDA_COMPARISON_CSS).encode()) <= 60_000


def test_hostile_profile_text_is_html_and_script_safe(public_profiles):
    hostile = deepcopy(public_profiles["Abraham Lincoln"])
    hostile["party"] = HOSTILE
    payload = compare_site.comparison_payload({"Abraham Lincoln": hostile})
    page = compare_site.render_compare_page(payload)
    assert HOSTILE not in page
    assert html.escape(HOSTILE) in page
    assert r"\u003c/script\u003e" in page
