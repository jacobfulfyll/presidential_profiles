"""Focused contracts for the generated Explorer interaction."""

from presidential_profiles import explorer


def _page(monkeypatch, tmp_path):
    (tmp_path / "docs").mkdir()
    monkeypatch.setattr(explorer, "REPO_ROOT", tmp_path)
    explorer.write_page()
    return (tmp_path / "docs" / "explorer.html").read_text()


def test_complete_typed_ordered_url_state_and_empty_state(monkeypatch, tmp_path):
    page = _page(monkeypatch, tmp_path)
    assert 'url.searchParams.append("s", `${s.type}:${s.type === "word" ? s.q : s.label}`)' in page
    assert 'url.searchParams.set("state", "1")' in page
    assert 'params.getAll("s")' in page
    assert 'type === "word"' in page
    assert 'type === "topic"' in page
    assert 'type === "entity"' in page
    assert 'url.searchParams.set("grouped", "1")' in page
    assert 'url.searchParams.set("combine", "1")' in page
    assert 'url.searchParams.set("periods", [...activePeriods].join(","))' in page


def test_preset_customization_and_url_validation(monkeypatch, tmp_path):
    page = _page(monkeypatch, tmp_path)
    assert "function reconcilePreset()" in page
    assert "Customized from ${PRESETS[customizedFrom].label}." in page
    assert 'url.searchParams.set("from", customizedFrom)' in page
    assert "Unknown preset skipped" in page
    assert "Unknown topic skipped" in page
    assert "Unknown named entity skipped" in page
    assert "unknown era" in page
    assert "Skipped ${skipped.join" in page


def test_fetch_errors_are_not_under_threshold_results(monkeypatch, tmp_path):
    page = _page(monkeypatch, tmp_path)
    get_shard = page[page.index("async function getShard"):page.index("function rolling")]
    add_term = page[page.index("async function addTerm"):page.index("function addTopic")]
    assert "catch" not in get_shard
    assert "Could not load corpus data" in add_term
    assert "appears fewer than" in add_term
    assert "pendingTerms.has" in add_term
    assert "return false" in add_term
    assert 'if (await addTerm(input.value)) input.value = ""' in page


def test_csv_and_accessibility_contracts(monkeypatch, tmp_path):
    page = _page(monkeypatch, tmp_path)
    assert "series,series_type,year,value,unit,scale,grouped" in page
    assert "own_peak_index" in page and "own_peak_100" in page
    assert "Different source units — each line is indexed to its own peak (=100)." in page
    assert 'role="status" aria-live="polite"' in page
    assert 'aria-label`, `Remove' not in page  # dynamic labels are not interpolated into HTML
    assert 'remove.setAttribute("aria-label", `Remove ${s.label}.`)' in page
    assert 'aria-labelledby", "chart-heading chart-summary"' in page
    assert "centered five-year smoothing" in page
    assert 'id="evidence"' in page and "buildExactTable()" in page
    assert 'if (e.target.open) buildExactTable()' in page
    assert "membership.textContent" in page
    assert "td.textContent = value" in page
    assert ":focus-visible" in page and "outline:3px" in page


def test_information_hierarchy_and_mobile_overview(monkeypatch, tmp_path):
    page = _page(monkeypatch, tmp_path)
    assert page.index("Add a word or phrase") < page.index('id="chips"') < page.index('id="unitnote"')
    assert page.index('id="unitnote"') < page.index('id="chart"') < page.index('id="guided"')
    assert page.index('id="guided"') < page.index('id="options"') < page.index('id="evidence"')
    assert "Clear series" in page and "Clear era bands" in page and "Reset chart" in page
    assert "@media (max-width:600px)" in page
    assert ".chart-scroll { overflow:hidden; min-width:0; }" in page
    assert "#chart { width:100%!important; min-width:0!important; }" in page
