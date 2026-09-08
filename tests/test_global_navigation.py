from __future__ import annotations

import re

import pytest

from presidential_profiles.expansion_site import (
    NAV,
    NAV_CSS,
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
        rf'<div class="nav-submenu" id="nav-submenu-{key}" hidden>'
        rf'<div class="nav-submenu-panel">(.*?)</div></div>',
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


def test_selfcontained_story_shell_retains_story_current_state(tmp_path):
    page = tmp_path / "index_selfcontained.html"
    page.write_text("<body><footer></footer></body>")

    add_global_navigation(tmp_path)

    markup = page.read_text()
    assert 'href="index.html" data-nav-label="Story" aria-current="page"' in markup


def test_disclosures_use_native_controls_and_cover_required_interactions():
    markup = nav()
    assert markup.count('class="nav-trigger" type="button"') == 2
    assert markup.count('aria-expanded="false"') == 2
    assert 'aria-controls="nav-submenu-profiles"' in markup
    assert 'aria-controls="nav-submenu-data"' in markup
    assert markup.count('class="nav-submenu-panel"') == 2
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
        'event.pointerType === "touch"',
        'event.pointerType === "mouse"',
        'event.pointerType === "pen"',
        'open(group, "hover")',
        'open(group, "pinned")',
    ):
        assert behavior in NAV_JS


def test_phone_navigation_is_native_complete_and_current():
    markup = nav(prefix="../", current_href="presidents/index.html")
    assert '<div class="mobile-nav">' in markup
    assert '<span class="mobile-current">Presidents</span>' in markup
    assert (
        '<details class="mobile-menu"><summary aria-controls="primary-nav-links">'
        "Menu</summary>"
    ) in markup
    shared_links = markup.split('<div class="nav-links" id="primary-nav-links">', 1)[1].split(
        '<div class="mobile-nav">', 1
    )[0]
    for label, href in [
        ("Story", "../index.html"),
        ("Summary", "../summary.html"),
        ("Compare", "../compare.html"),
        ("Explore", "../explorer.html"),
        ("Presidents", "../presidents/index.html"),
        ("Issues", "../issues/index.html"),
        ("Data Quality", "../data-quality.html"),
        ("Methods", "../methodology.html"),
        ("Era Choices", "../era-boundaries.html"),
    ]:
        assert f'href="{href}"' in shared_links
        assert f">{label}</a>" in shared_links
    assert (
        'href="../presidents/index.html" aria-current="page">Presidents</a>'
        in shared_links
    )
    assert "role=\"menu\"" not in shared_links
    assert markup.count('href="../presidents/index.html"') == 1


def test_phone_navigation_is_56px_and_desktop_shell_is_not_reflowed():
    compact_css = re.sub(r"\s+", "", NAV_CSS)
    assert ".mobile-nav{display:none}" in compact_css
    assert (
        "@media(max-width:760px),(max-height:500px)and(max-width:932px)and(pointer:coarse)"
        in compact_css
    )
    assert "min-height:56px" in compact_css
    assert ".nav-brand-label,.nav-links{display:none}" in compact_css
    assert ".mobile-nav{display:flex" in compact_css
    assert "max-height:calc(100dvh-var(--global-nav-height,56px)-16px)" in compact_css
    assert ".nav-shell:has(.mobile-menu[open])>.nav-links{position:fixed" in compact_css
    assert ".nav-submenu[hidden]," in compact_css
    assert "min-height:44px" in compact_css
    assert "position:static!important;top:auto!important" in compact_css


def test_phone_menu_escape_outside_activation_and_link_close_are_enhanced():
    assert 'const mobileMenu = nav.querySelector(".mobile-menu")' in NAV_JS
    assert 'event.key !== "Escape" || !mobileMenu.open' in NAV_JS
    assert 'mobileMenu.removeAttribute("open")' in NAV_JS
    assert "mobileSummary.focus()" in NAV_JS
    assert "!nav.contains(event.target)" in NAV_JS
    assert 'nav.querySelectorAll(".nav-links a")' in NAV_JS


def test_submenu_wrapper_bridges_trigger_and_panel_without_a_dead_zone():
    compact_css = re.sub(r"\s+", "", NAV_CSS)
    assert (
        ".nav-submenu{position:absolute;top:100%;right:0;z-index:120;"
        "padding-top:7px}" in compact_css
    )
    panel_rule = re.search(r"\.nav-submenu-panel\{([^}]+)\}", compact_css)
    assert panel_rule is not None
    assert "background:#fff" in panel_rule.group(1)
    assert "border:1pxsolidrgba(11,11,11,.14)" in panel_rule.group(1)
    assert "box-shadow:" in panel_rule.group(1)
    outer_rule = re.search(r"\.nav-submenu\{([^}]+)\}", compact_css)
    assert outer_rule is not None
    assert "background:" not in outer_rule.group(1)
    assert "box-shadow:" not in outer_rule.group(1)


def test_nav_state_machine_distinguishes_hover_and_pinned_activation():
    assert "let openGroup = null" in NAV_JS
    assert "let openMode = null" in NAV_JS
    assert 'if (openMode === "hover")' in NAV_JS
    assert 'openMode = "pinned"' in NAV_JS
    assert 'openGroup === group && openMode === "hover"' in NAV_JS
    assert (
        'trigger.setAttribute("aria-expanded", isOpen ? "true" : "false")'
        in NAV_JS
    )
    assert "submenu.hidden = !isOpen" in NAV_JS
    assert 'group.classList.toggle("is-open", isOpen)' in NAV_JS


def test_progressive_and_high_contrast_navigation_styles_are_shared():
    compact_css = re.sub(r"\s+", "", NAV_CSS)
    assert "html:not(.nav-enhanced).nav-group:hover>.nav-submenu," in compact_css
    assert (
        "html:not(.nav-enhanced).nav-group:focus-within>.nav-submenu{display:block!important}"
        in compact_css
    )
    assert "min-width:44px" in compact_css
    assert "min-height:44px" in compact_css
    assert "@media(forced-colors:active)" in compact_css
    assert "outline-color:Highlight" in compact_css
    assert "text-decoration-thickness:3px" in compact_css
    assert "border:2pxsolidCanvasText" in compact_css


def test_enhancement_class_is_added_only_after_listeners_are_installed():
    enhancement = NAV_JS.index('classList.add("nav-enhanced")')
    assert NAV_JS.rindex("addEventListener") < enhancement


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
