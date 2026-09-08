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


def test_metric_dictionary_groups_cover_each_measure_once():
    metrics.validate_metric_groups()
    grouped = [
        name
        for _group_key, _group_label, names in metrics.METRIC_GROUPS
        for name in names
    ]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(metrics.METRICS)


def test_distinctive_reference_metric_names_unit_and_limits():
    metric = metrics.METRICS["era_distinctive_reference"]
    assert metric["status"] == "exploratory descriptive"
    assert metric["source"] == (
        "reference_entities/era_distinctive_v1.parquet + "
        "speaker_views/paragraph_view_v1.parquet"
    )
    assert "Jeffreys" in metric["definition"]
    assert "source agreement" in metric["limitations"]
    assert "representative" in metric["limitations"]


def test_reference_landscape_metric_uses_complete_actual_speaker_denominator():
    metric = metrics.METRICS["reference_type_presence"]
    assert metric["status"] == "exploratory descriptive"
    assert "every eligible actual-president paragraph" in metric["definition"]
    assert "non-additive" in metric["limitations"]
    assert metrics.CHART_METRICS["era_reference_landscape"] == {
        "reference_type_presence"
    }


def test_explore_metrics_are_registered_at_their_governed_grains():
    lexical = metrics.METRICS["explore_centered_lexical_rate"]

    assert "center year minus two" in lexical["definition"]
    assert "exceeds 20,000 indexed words" in lexical["definition"]
    assert "unknown rather than zero" in lexical["limitations"]
    assert metrics.CHART_METRICS["explore_lexical_trends"] == {
        "explore_centered_lexical_rate"
    }
    assert metrics.CHART_METRICS["explore_corex_trends"] == {
        "paragraph_share",
        "confidence_interval",
    }
    assert metrics.CHART_METRICS["explore_ai_topic_trends"] == {
        "paragraph_share",
        "uncertainty_envelope",
    }


def test_profile_connection_metrics_keep_their_governed_grains():
    invocation = metrics.METRICS["actual_speaker_invocation_paragraph_count"]

    assert invocation["unit"] == "eligible reference paragraphs"
    assert "distinct (doc_name, para_idx)" in invocation["formula"]
    assert "not historical contact" in invocation["limitations"]
    assert metrics.CHART_METRICS["profile_topic_ego"] == {
        "actual_speaker_topic_presence"
    }
    assert "profile_invocation_ego" not in metrics.CHART_METRICS
    assert metrics.CHART_METRICS["profile_invocation_bars"] == {
        "actual_speaker_invocation_paragraph_count"
    }
    assert "displayed bar value" in invocation["example"]


def test_data_trust_units_and_pipeline_metric_match_their_public_figures():
    uncertainty = metrics.METRICS["uncertainty_envelope"]
    assert "percentage-point interval width" in uncertainty["unit"]
    assert "× 100 percentage points" in uncertainty["formula"]
    assert uncertainty["source"].startswith(
        "data/quality/uncertainty_decomposition.csv"
    )
    pipeline = metrics.METRICS["annotation_pipeline_lineage"]
    assert pipeline["unit"] == "ordered governed processing stages"
    assert "human validity" in pipeline["limitations"]
