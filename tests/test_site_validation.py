import pytest

from presidential_profiles.expansion_site import _feedback_page
from presidential_profiles.site import bundle_plotly_runtime, write_github_pages_marker
from presidential_profiles.site_validation import validate_site


def test_link_and_anchor_validation(tmp_path):
    (tmp_path / "index.html").write_text(
        '<body><nav aria-label="Primary"></nav><a href="other.html#ok">go</a></body>')
    (tmp_path / "other.html").write_text(
        '<body><nav aria-label="Primary"></nav><h1 id="ok">yes</h1></body>')
    assert validate_site(tmp_path)["ok"]
    (tmp_path / "index.html").write_text(
        '<body><nav aria-label="Primary"></nav><a href="missing.html">bad</a></body>')
    with pytest.raises(ValueError, match="missing target"):
        validate_site(tmp_path)


def test_feedback_embed_has_real_giscus_ids_and_fallback():
    page = _feedback_page()
    assert 'data-repo-id="R_kgDOCrXyjA"' in page
    assert 'data-category-id="DIC_kwDOCrXyjM4DByqZ"' in page
    assert 'data-mapping="pathname"' in page
    assert "/discussions/1" in page


@pytest.mark.parametrize("legacy_shape", ["output", "link"])
def test_retired_network_route_is_rejected(legacy_shape, tmp_path):
    (tmp_path / "index.html").write_text(
        '<body><nav aria-label="Primary"></nav></body>'
    )
    if legacy_shape == "output":
        (tmp_path / "networks.html").write_text(
            '<body><nav aria-label="Primary"></nav></body>'
        )
    else:
        (tmp_path / "index.html").write_text(
            '<body><nav aria-label="Primary"></nav>'
            '<a href="networks.html">legacy</a></body>'
        )
    with pytest.raises(ValueError, match="retired"):
        validate_site(tmp_path)


def test_substantive_chart_requires_lesson_and_evidence_controls(tmp_path):
    (tmp_path / "index.html").write_text(
        '<body><nav aria-label="Primary"></nav><div class="chart"></div></body>'
    )
    with pytest.raises(ValueError, match="metric lessons"):
        validate_site(tmp_path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_validation_rejects_nonfinite_constants(tmp_path, constant):
    (tmp_path / "index.html").write_text(
        '<body><nav aria-label="Primary"></nav></body>'
    )
    (tmp_path / "payload.json").write_text(f'{{"value": {constant}}}')
    with pytest.raises(ValueError, match="non-finite JSON value"):
        validate_site(tmp_path)


def test_plotly_runtime_is_bundled_with_depth_correct_paths(tmp_path):
    nested = tmp_path / "issues"
    nested.mkdir()
    cdn = '<script src="https://cdn.plot.ly/plotly-3.0.1.min.js" charset="utf-8"></script>'
    (tmp_path / "index.html").write_text(cdn)
    (nested / "one.html").write_text(cdn)
    bundle_plotly_runtime(tmp_path)
    assert (tmp_path / "assets" / "plotly-3.0.1.min.js").stat().st_size > 1_000_000
    assert 'src="assets/plotly-3.0.1.min.js"' in (tmp_path / "index.html").read_text()
    assert 'src="../assets/plotly-3.0.1.min.js"' in (nested / "one.html").read_text()


def test_github_pages_marker_is_empty_and_idempotent(tmp_path):
    marker = write_github_pages_marker(tmp_path)
    assert marker == tmp_path / ".nojekyll"
    assert marker.read_bytes() == b""
    assert write_github_pages_marker(tmp_path).read_bytes() == b""
