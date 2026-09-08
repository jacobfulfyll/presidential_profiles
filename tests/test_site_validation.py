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


def test_css_chart_class_does_not_define_a_substantive_figure(tmp_path):
    (tmp_path / "index.html").write_text(
        '<body><nav aria-label="Primary"></nav><div class="chart"></div></body>'
    )
    assert validate_site(tmp_path)["ok"]


def test_governed_route_rejects_mixed_marked_and_unmarked_charts(tmp_path):
    (tmp_path / "metrics.html").write_text(
        '<body><nav aria-label="Primary"></nav><h1 id="kappa">Kappa</h1></body>'
    )
    (tmp_path / "data-quality.html").write_text(
        '<body><nav aria-label="Primary"></nav>'
        '<figure data-substantive-chart data-metric="kappa" '
        'data-evidence="audit.csv" aria-label="Marked">'
        '<a href="metrics.html#kappa">Definition</a></figure>'
        '<div class="chart" id="unmarked"></div></body>'
    )
    with pytest.raises(ValueError, match="outside a marked substantive figure"):
        validate_site(tmp_path)


def test_governed_route_rejects_unmarked_figure(tmp_path):
    (tmp_path / "metrics.html").write_text(
        '<body><nav aria-label="Primary"></nav></body>'
    )
    (tmp_path / "methodology.html").write_text(
        '<body><nav aria-label="Primary"></nav><figure>Evidence</figure></body>'
    )
    with pytest.raises(ValueError, match="lack data-substantive-chart"):
        validate_site(tmp_path)


@pytest.mark.parametrize(
    ("markup", "message"),
    [
        (
            '<figure data-substantive-chart="quality" data-evidence="audit.csv" '
            'aria-label="Quality audit"></figure>',
            "missing data-metric",
        ),
        (
            '<figure data-substantive-chart="quality" data-metric="kappa" '
            'aria-label="Quality audit"></figure>',
            "missing data-evidence",
        ),
        (
            '<figure data-substantive-chart="quality" data-metric="kappa" '
            'data-evidence="audit.csv"></figure>',
            "missing an accessible name",
        ),
    ],
)
def test_semantic_chart_contract_requires_metric_evidence_and_name(
    tmp_path, markup, message
):
    (tmp_path / "index.html").write_text(
        f'<body><nav aria-label="Primary"></nav>{markup}</body>'
    )
    with pytest.raises(ValueError, match=message):
        validate_site(tmp_path)


def test_semantic_chart_contract_closes_quality_chart_class_loophole(tmp_path):
    (tmp_path / "index.html").write_text(
        '<body><nav aria-label="Primary"></nav>'
        '<figure class="quality-chart" data-substantive-chart="quality" '
        'data-metric="kappa" data-evidence="data/quality/audit.csv" '
        'aria-labelledby="chart-title"><h2 id="chart-title">Audit</h2></figure>'
        '</body>'
    )
    assert validate_site(tmp_path)["schema_version"] == "site-validation-v2"


def test_semantic_chart_contract_rejects_unregistered_measures(tmp_path):
    (tmp_path / "index.html").write_text(
        '<body><nav aria-label="Primary"></nav>'
        '<figure data-substantive-chart data-metric="invented_score" '
        'data-evidence="audit.csv" aria-label="Invented chart"></figure>'
        '</body>'
    )
    with pytest.raises(ValueError, match="unregistered data-metric"):
        validate_site(tmp_path)


def test_semantic_chart_contract_accepts_multiple_registered_measures(tmp_path):
    (tmp_path / "index.html").write_text(
        '<body><nav aria-label="Primary"></nav>'
        '<figure data-substantive-chart data-metric="jaccard kappa" '
        'data-evidence="audit.csv" aria-label="Two agreement measures"></figure>'
        '</body>'
    )
    assert validate_site(tmp_path)["ok"] is True


def test_substantive_marker_requires_figure_markup(tmp_path):
    (tmp_path / "index.html").write_text(
        '<body><nav aria-label="Primary"></nav>'
        '<div data-substantive-chart data-metric="kappa" '
        'data-evidence="audit.csv" aria-label="Improper chart"></div></body>'
    )
    with pytest.raises(ValueError, match="semantic figure markup"):
        validate_site(tmp_path)


def test_governed_figure_must_link_each_metric_definition(tmp_path):
    (tmp_path / "metrics.html").write_text(
        '<body><nav aria-label="Primary"></nav><h1 id="jaccard">Jaccard</h1>'
        '<h1 id="kappa">Kappa</h1></body>'
    )
    (tmp_path / "label-models.html").write_text(
        '<body><nav aria-label="Primary"></nav>'
        '<figure data-substantive-chart data-metric="jaccard kappa" '
        'data-evidence="audit.csv" aria-label="Agreement">'
        '<a href="metrics.html#jaccard">Jaccard definition</a></figure></body>'
    )
    with pytest.raises(ValueError, match="metric definition kappa"):
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


def test_story_download_links_are_validated_like_other_internal_assets(tmp_path):
    story_dir = tmp_path / "data" / "story"
    story_dir.mkdir(parents=True)
    (story_dir / "era_distinctive_v1.csv").write_text("rank\n1\n")
    (tmp_path / "index.html").write_text(
        '<body><nav aria-label="Primary"></nav>'
        '<a href="data/story/era_distinctive_v1.csv">download</a></body>'
    )
    assert validate_site(tmp_path)["ok"]
