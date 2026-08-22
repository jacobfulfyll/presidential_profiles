"""Regression coverage for ``topic_display_names.json`` HTML/DOM sinks.

Sink inventory:

* issue page title, heading, prose, and issue-index card: escaped HTML text;
* profile issue card and president-index badge: escaped HTML text;
* dashboard/profile Plotly bundles and comparison payload: script-safe JSON;
* issue-page Plotly bundle: Plotly's direct ``to_json`` escaping;
* Explorer chip labels and comparison issue labels: DOM ``textContent`` /
  ``createTextNode`` construction;
* profile-download JSON and Explorer ``meta.json``: data-only JSON whose
  browser consumers use the safe DOM paths above.
"""

import html
import json

import pandas as pd
import plotly.graph_objects as go

from presidential_profiles import (
    compare_site,
    explorer,
    issues_site,
    profiles,
    profiles_site,
)
from presidential_profiles.html_safety import json_for_script


HOSTILE = '<img src=x onerror=alert(1)>'
SCRIPT_BREAKOUT = '</script><img src=x onerror=alert(1)>'
PUNCTUATED = 'Peace & workers\' "rights"'


def _profile_card_data(issue: str) -> dict:
    return {
        "issue_cards": {
            "President A": {
                "cards": [{
                    "issue": issue,
                    "words": [],
                    "quote": "",
                    "cite": "",
                    "stance": "",
                    "share": 0.25,
                    "rel": 2.0,
                    "topic_of_day": False,
                }],
                "voice": [],
                "low_confidence": False,
            }
        }
    }


def _compare_data() -> dict:
    row = {
        "party": "Test",
        "first_year": 1900,
        "last_year": 1901,
        "n_speeches": 6,
        "certainty": 1,
        "hype": 2,
        "mechanism": 3,
        "nrc_hope": 4,
        "nrc_fear": 5,
        "fk_grade": 6,
    }
    for key, _ in profiles.RADAR_AXES:
        row[f"pct_{key}"] = 50
    return {
        "scores": pd.DataFrame([row], index=["President A"]),
        "voice_neighbors": {},
        "agenda_neighbors": {},
    }


def _shared_profile_payload(label: str) -> dict:
    return {
        "schema_version": "president-profile-v3",
        "president": "President A",
        "slug": "president-a",
        "party": "Test",
        "years": {"first": 1900, "last": 1901},
        "sample": {
            "n_speeches": 6,
            "n_words": 1000,
            "thin_record": False,
            "warning": None,
        },
        "rhetorical_radar": [
            {
                "key": key,
                "label": label,
                "absolute": 1.0,
                "percentile": 50.0,
            }
            for key, label in profiles.RADAR_AXES
        ],
        "raw_stats": {"x": 1},
        "legacy_issue_attention": [{
            "key": "Discovered 5",
            "label": label,
            "share": 0.1,
            "era_relative_difference": 2.0,
        }],
        "issue_evidence": {"cards": []},
        "ai": None,
        "distinctive_vocabulary": [{"term": "word"}],
        "signature_speeches": [],
        "legacy_invocations": {"invokes": [], "invoked_by": None},
        "classified_invocations": [],
        "voice_neighbors": [],
        "agenda_neighbors": [],
        "feature_neighbors": {},
        "context_specific": {"profile_only": [], "compare_only": []},
    }


def test_issue_page_title_heading_prose_and_index_escape_display_names():
    escaped = html.escape(HOSTILE)
    page = issues_site.render_issue(
        HOSTILE,
        pd.DataFrame({"share": []}),
        go.Figure(),
        [],
    )
    index = issues_site.render_issue_index([{
        "slug": "hostile",
        "label": HOSTILE,
        "peak": 1900,
        "top": "President A",
    }])

    assert HOSTILE not in page
    assert f"<title>{escaped} - Presidential Profiles</title>" in page
    assert f"<h1>{escaped}</h1>" in page
    assert f"speech about {html.escape(HOSTILE.lower())}." in page
    assert HOSTILE not in index
    assert f'<div class="name">{escaped}</div>' in index


def test_profile_cards_and_index_escape_every_issue_label(monkeypatch):
    monkeypatch.setattr(
        profiles_site,
        "DISCOVERED_LABELS",
        {"Discovered 5": HOSTILE},
    )
    cards = profiles_site._issue_cards_html(
        "President A",
        _profile_card_data("Discovered 5"),
    )
    scores = pd.DataFrame([{
        "party": "Test",
        "first_year": 1900,
        "last_year": 1901,
        "n_speeches": 6,
    }], index=["President A"])
    issue_frame = pd.DataFrame([{
        "n_paragraphs": 100,
        "share_Discovered 5": 0.25,
        "rel_Discovered 5": 2.0,
    }], index=["President A"])
    index = profiles_site.render_index(
        {"scores": scores, "issues": issue_frame},
        ["Discovered 5"],
    )

    assert HOSTILE not in cards
    assert html.escape(HOSTILE) in cards
    assert HOSTILE not in index
    assert html.escape(HOSTILE) in index


def test_ordinary_punctuation_is_escaped_once_and_renders_as_text():
    escaped = html.escape(PUNCTUATED)
    page = issues_site.render_issue(
        PUNCTUATED,
        pd.DataFrame({"share": []}),
        go.Figure(),
        [],
    )

    assert f"<h1>{escaped}</h1>" in page
    assert "&amp;amp;" not in page
    assert html.unescape(escaped) == PUNCTUATED


def test_script_json_blocks_breakout_without_changing_decoded_values():
    value = {"hostile": SCRIPT_BREAKOUT, "ordinary": PUNCTUATED}
    encoded = json_for_script(value)

    assert SCRIPT_BREAKOUT not in encoded
    assert "</script>" not in encoded.lower()
    assert r"\u003c/script\u003e" in encoded
    assert json.loads(encoded) == value


def test_direct_plotly_serialization_keeps_display_name_out_of_markup():
    figure = go.Figure(go.Bar(y=[SCRIPT_BREAKOUT], x=[1]))
    page = issues_site.render_issue(
        "Ordinary",
        pd.DataFrame({"share": []}),
        figure,
        [],
    )

    assert SCRIPT_BREAKOUT not in page
    assert r"\u003c\u002fscript\u003e" in page


def test_explorer_display_name_chips_use_safe_dom_construction(
    monkeypatch, tmp_path
):
    (tmp_path / "docs").mkdir()
    monkeypatch.setattr(explorer, "REPO_ROOT", tmp_path)
    explorer.write_page()
    page = (tmp_path / "docs" / "explorer.html").read_text()
    chips = page[
        page.index("function renderChips()"):
        page.index("function msg(", page.index("function renderChips()"))
    ]
    tooltip = page[
        page.index("function showTip("):
        page.index("function moveTip(", page.index("function showTip("))
    ]

    assert "innerHTML" not in chips
    assert "createTextNode(` ${s.label} `)" in chips
    assert "innerHTML" not in tooltip
    assert "heading.textContent" in tooltip
    assert "o.textContent =" in page


def test_comparison_payload_and_issue_labels_use_safe_script_and_dom(
    monkeypatch, tmp_path
):
    (tmp_path / "docs").mkdir()
    monkeypatch.setattr(compare_site, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        compare_site.profiles_site,
        "public_profile_payload",
        lambda president, data, issues: _shared_profile_payload(SCRIPT_BREAKOUT),
    )
    compare_site.write_compare(_compare_data(), ["Discovered 5"])
    page = (tmp_path / "docs" / "compare.html").read_text()

    assert SCRIPT_BREAKOUT not in page
    assert r"\u003c/script\u003e" in page
    assert html.escape(SCRIPT_BREAKOUT) in page
    assert "const escapeHTML = value =>" in page
    assert "option.textContent = P[name].display_name" in page
    assert "escapeHTML(row.name)" in page


def test_all_embedded_display_name_payloads_use_script_safe_json():
    profile_source = profiles_site.render_profile.__code__.co_consts
    assert any(
        isinstance(value, str) and "const FIGS = " in value
        for value in profile_source
    )
    assert "json_for_script(fig_json)" in open(
        profiles_site.__file__, encoding="utf-8"
    ).read()
    from presidential_profiles import site

    assert "json_for_script(fig_json)" in open(
        site.__file__, encoding="utf-8"
    ).read()
    assert "json_for_script(payload)" in open(
        compare_site.__file__, encoding="utf-8"
    ).read()
