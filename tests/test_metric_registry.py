from presidential_profiles import metrics


def test_every_metric_has_complete_public_definition():
    required = {"label", "question", "definition", "example", "steps", "formula",
                "unit", "limitations", "source", "status"}
    assert len(metrics.METRICS) >= 15
    for value in metrics.METRICS.values():
        assert required <= set(value)
        assert all(value[key] for key in required)


def test_every_registered_chart_resolves_to_metrics():
    metrics.validate_charts(set(metrics.CHART_METRICS))

