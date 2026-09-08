import math

import pytest

from presidential_profiles import metrics, trust_appendices


def test_methods_local_navigation_has_the_shared_seven_destinations():
    markup = trust_appendices.methods_local_nav("metric-dictionary")
    assert markup.count("<li><a") == 7
    assert "More sections →" in markup
    for label in (
        "Overview",
        "Labeling",
        "Evaluation",
        "Populations",
        "Model Comparison",
        "Metric Dictionary",
        "Downloads",
    ):
        assert label in markup
    assert 'href="metrics.html" aria-current="page"' in markup
    assert "@media(max-width:760px)" in trust_appendices.APPENDIX_CSS
    assert ".methods-local-nav a,.metric-filters button,.appendix-section summary,.download-links a{min-height:44px" in trust_appendices.APPENDIX_CSS
    assert ".metric-card .deep-link{display:grid;place-items:center;min-width:44px;min-height:44px" in trust_appendices.APPENDIX_CSS
    assert ".instrument-card .eyebrow,.agreement-values span,.receipt-card span{font-size:12px}" in trust_appendices.APPENDIX_CSS


def test_metric_dictionary_is_grouped_searchable_and_key_complete():
    body, script = trust_appendices.render_metric_dictionary()
    assert body.count('class="metric-card"') == len(metrics.METRICS)
    for name in metrics.METRICS:
        assert f'id="{name}"' in body
    for group_key, group_label, _ in metrics.METRIC_GROUPS:
        assert f'id="group-{group_key}"' in body
        assert group_label in body
    assert 'id="metric-search"' in body
    assert "aria-live=\"polite\"" in body
    assert "card.hidden" in script
    assert "current.offsetLeft" in script
    assert "Two-minute visual narrative" in body
    for label in (
        "Claim", "Population", "Method", "Uncertainty", "Artifact", "Status",
        "Limitation", "Downstream use",
    ):
        assert f"<dt>{label}</dt>" in body


def test_model_comparison_recomputes_the_failed_prediction_from_artifacts():
    evidence = trust_appendices.load_model_comparison()
    prediction = evidence["prediction"]
    assert len(evidence["overall"]) == 16
    assert prediction["n_issues"] == 15
    assert prediction["counterexample"] == "Immigration"
    assert math.isclose(prediction["rho"], -0.05357142857142857)
    assert math.isclose(prediction["p_value"], 0.8496099367776072)
    assert evidence["population"] == {
        "source_paragraphs": 36_229,
        "topic_bearing_paragraphs": 35_825,
        "no_topic_paragraphs_excluded": 404,
        "exclusion_rule": "drop_unlabeled=True in the preregistered comparison estimand",
    }
    assert evidence["public_schema"] == {
        "schema_version": "method-comparison-public-v1",
        "method-agreement.csv": {
            "semantic_key": ["issue", "era"],
            "columns": [
                "issue", "era", "n", "n_llm", "n_corex", "n_both", "jaccard",
                "kappa", "min_support", "low_support", "empty_arm",
            ],
        },
        "method-compositions.csv": {
            "semantic_key": ["doc_name", "method", "topic"],
            "columns": ["doc_name", "method", "topic", "share", "n_paragraphs"],
        },
    }


def test_model_comparison_is_server_rendered_and_links_each_measure():
    body, script = trust_appendices.render_model_comparison()
    assert "current.offsetLeft" in script
    assert "Plotly" not in body
    assert body.count("data-substantive-chart") == 2
    assert 'data-metric="paragraph_share"' in body
    assert 'data-metric="jaccard kappa"' in body
    assert 'href="metrics.html#paragraph_share"' in body
    assert 'href="metrics.html#jaccard"' in body
    assert 'href="metrics.html#kappa"' in body
    assert "Preregistered result · failed prediction" in body
    assert "Spearman ρ = -0.054" in body
    assert "two-sided p = 0.850" in body
    assert "false positive" not in body.casefold()
    assert "false negative" not in body.casefold()
    assert "data/method-agreement.csv" in body
    assert "data/method_agreement.parquet" not in body
    assert "method-comparison-public-v1" in body
    assert "semantic key <code>(issue, era)</code>" in body
    assert "semantic key <code>(doc_name, method, topic)</code>" in body
    assert "method_compositions.parquet" in body
    assert "35,825 topic-bearing paragraphs" in body
    assert "404 primary-LLM no-topic paragraphs" in body
    assert "36,229 source paragraphs" in body


def test_model_comparison_rejects_population_and_margin_drift(monkeypatch):
    source = trust_appendices.pd.read_parquet(
        trust_appendices.corpus.DATA_DIR / "method_agreement.parquet"
    )
    overall_indices = source.index[source["era"].isna()].tolist()

    drifted = source.copy()
    drifted.loc[overall_indices[0], "n"] += 1
    monkeypatch.setattr(
        trust_appendices.pd, "read_parquet", lambda _path: drifted.copy()
    )
    with pytest.raises(ValueError, match="share one comparison population"):
        trust_appendices.load_model_comparison()

    inconsistent = source.copy()
    inconsistent.loc[overall_indices[0], "n_both"] = (
        inconsistent.loc[overall_indices[0], "n_llm"] + 1
    )
    monkeypatch.setattr(
        trust_appendices.pd, "read_parquet", lambda _path: inconsistent.copy()
    )
    with pytest.raises(ValueError, match="count margins"):
        trust_appendices.load_model_comparison()


def test_trust_receipt_exposes_all_eight_contract_fields():
    markup = trust_appendices.trust_receipt(
        claim="c",
        population="p",
        method="m",
        uncertainty="u",
        artifact="a",
        status="s",
        limitation="l",
        downstream_use="d",
    )
    for label in (
        "Claim",
        "Population",
        "Method",
        "Uncertainty",
        "Artifact",
        "Status",
        "Limitation",
        "Downstream use",
    ):
        assert f"<dt>{label}</dt>" in markup
