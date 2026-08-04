from __future__ import annotations

import re

import pytest

from presidential_profiles.expansion_site import (
    NAV,
    NAV_JS,
    _prune_retired_routes,
    add_global_navigation,
    nav,
)


def _top_level_labels(markup: str) -> list[str]:
    navigation_markup = markup.split("<script>", 1)[0]
    return re.findall(r'data-nav-label="([^"]+)"', navigation_markup)


def _submenu(markup: str, key: str) -> str:
    match = re.search(
        rf'<div class="nav-submenu" id="nav-submenu-{key}" hidden>(.*?)</div>',
        markup,
        re.S,
    )
    assert match is not None
    return match.group(1)


def _submenu_links(markup: str, key: str) -> list[tuple[str, str]]:
    return re.findall(
        r'<a href="([^"]+)"(?: aria-current="page")?>([^<]+)</a>',
        _submenu(markup, key),
    )


def test_information_architecture_has_exact_order_and_submenu_membership():
    assert [label for label, _ in NAV] == [
        "Story",
        "Summary",
        "Compare",
        "Explore",
        "Profiles",
        "Data",
    ]
    markup = nav(current_href="index.html")
    assert _top_level_labels(markup) == [
        "Story",
        "Summary",
        "Compare",
        "Explore",
        "Profiles",
        "Data",
    ]
    assert _submenu_links(markup, "profiles") == [
        ("presidents/index.html", "Presidents"),
        ("issues/index.html", "Issues"),
    ]
    assert _submenu_links(markup, "data") == [
        ("data-quality.html", "Data Quality"),
        ("methodology.html", "Methods"),
        ("era-boundaries.html", "Era Choices"),
    ]
    assert "Explorer" not in _top_level_labels(markup)
    assert "Feedback" not in _top_level_labels(markup)


@pytest.mark.parametrize(
    ("prefix", "summary", "presidents", "feedback"),
    [
        ("", "summary.html", "presidents/index.html", "feedback.html"),
        ("../", "../summary.html", "../presidents/index.html", "../feedback.html"),
    ],
)
def test_root_and_nested_navigation_urls_are_depth_correct(
    tmp_path, prefix, summary, presidents, feedback
):
    markup = nav(prefix=prefix, current_href="presidents/index.html")
    assert f'href="{summary}" data-nav-label="Summary"' in markup
    assert f'href="{presidents}" aria-current="page">Presidents</a>' in markup

    page_dir = tmp_path if not prefix else tmp_path / "presidents"
    page_dir.mkdir(exist_ok=True)
    page = page_dir / ("index.html" if not prefix else "washington.html")
    page.write_text(
        '<body><section id="synthesis"></section>'
        '<section id="records_appendix"></section><footer></footer></body>'
    )
    add_global_navigation(tmp_path)
    generated = page.read_text()
    assert f'href="{feedback}">Feedback and accuracy →</a>' in generated


@pytest.mark.parametrize(
    ("current", "group", "child"),
    [
        ("presidents/index.html", "Profiles", "Presidents"),
        ("issues/index.html", "Profiles", "Issues"),
        ("data-quality.html", "Data", "Data Quality"),
        ("methodology.html", "Data", "Methods"),
        ("era-boundaries.html", "Data", "Era Choices"),
    ],
)
def test_group_and_exact_child_current_state(current, group, child):
    markup = nav(current_href=current)
    assert (
        f'class="nav-item nav-group nav-group-current" data-nav-label="{group}"'
        in markup
    )
    group_key = group.lower()
    child_markup = _submenu(markup, group_key)
    assert f'aria-current="page">{child}</a>' in child_markup
    assert child_markup.count('aria-current="page"') == 1


@pytest.mark.parametrize("descendant", ["metrics.html", "label-models.html"])
def test_methods_descendants_mark_data_and_methods(tmp_path, descendant):
    page = tmp_path / descendant
    page.write_text("<body><footer></footer></body>")
    add_global_navigation(tmp_path)
    markup = page.read_text()
    assert (
        'class="nav-item nav-group nav-group-current" data-nav-label="Data"'
        in markup
    )
    assert 'href="methodology.html" aria-current="page">Methods</a>' in markup


def test_disclosures_use_native_controls_and_cover_required_interactions():
    markup = nav()
    assert markup.count('class="nav-trigger" type="button"') == 2
    assert markup.count('aria-expanded="false"') == 2
    assert 'aria-controls="nav-submenu-profiles"' in markup
    assert 'aria-controls="nav-submenu-data"' in markup
    assert 'role="menu"' not in markup
    assert 'role="menuitem"' not in markup
    for behavior in (
        "pointerenter",
        "pointerleave",
        'addEventListener("click"',
        'event.key !== "Enter" && event.key !== " "',
        'event.key !== "Escape"',
        "focusout",
        "pointerdown",
        "finePointer.matches",
    ):
        assert behavior in NAV_JS


def test_story_and_summary_have_separate_destinations_and_current_states():
    story = nav(current_href="index.html")
    summary = nav(current_href="summary.html")
    assert 'href="summary.html" data-nav-label="Summary"' in story
    assert 'href="index.html" data-nav-label="Story" aria-current="page"' in story
    assert 'href="summary.html" data-nav-label="Summary" aria-current="page"' in summary
    assert 'href="index.html" data-nav-label="Story"' in summary
    assert "#synthesis" not in story
    assert "#synthesis" not in summary
    assert 'data-story-document="true"' not in story
    assert "--global-nav-height" in NAV_JS


def test_retired_route_is_pruned_without_touching_network_artifacts(tmp_path):
    legacy = tmp_path / "networks.html"
    legacy.write_text("stale public page")
    network_data = tmp_path / "data" / "networks"
    network_data.mkdir(parents=True)
    artifact = network_data / "invocation_evidence.csv"
    artifact.write_text("evidence")

    _prune_retired_routes(tmp_path)

    assert not legacy.exists()
    assert artifact.read_text() == "evidence"
