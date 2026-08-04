"""All-era derivation, rendering, and publication contracts for context."""

from __future__ import annotations

import json
from math import hypot
from pathlib import Path

import pandas as pd
import pytest

from presidential_profiles import (
    attention,
    corpus,
    era_contextualizations,
    era_profiles,
    era_visualizations,
    site,
)


DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def all_contextualizations() -> dict[str, dict]:
    return era_contextualizations.build_era_contextualizations(
        corpus.load(),
        pd.read_parquet(DATA / "paragraphs.parquet"),
        pd.read_parquet(
            DATA / "llm_annotations" / "paragraph_annotations.parquet"
        ),
        attention.load_taxonomy(),
        era_profiles.load_all_era_profiles(),
        pd.read_parquet(
            DATA / "networks" / "invocation_evidence.parquet"
        ),
    )


def test_all_nine_eras_share_one_complete_contextualization_contract(
    all_contextualizations,
):
    profiles_by_era = era_profiles.load_all_era_profiles()
    assert list(all_contextualizations) == [
        spec.key for spec in era_profiles.ERA_PROFILE_SPECS
    ]
    for spec in era_profiles.ERA_PROFILE_SPECS:
        record = all_contextualizations[spec.key]
        assert record["years"] == f"{spec.start_year}–{spec.end_year}"
        assert len(record["era_axis"]) == 9
        assert len(record["topic_life"]["rows"]) == 6
        assert all(
            len(row["values"]) == 9
            for row in record["topic_life"]["rows"]
        )
        assert "topics" in record["era_echoes"]["incoming"]
        assert "topics" in record["era_echoes"]["outgoing"]
        assert "network" in record["era_echoes"]
        assert "paths" in record["era_echoes"]["network"]
        gravity = record["era_echoes"]["network"]["gravity"]
        assert gravity["layout"] == "connection_weighted_gravity"
        assert [
            node["label"] for node in gravity["anchor_nodes"]
        ] == [
            president["name"]
            for president in profiles_by_era[spec.key]["presidents"]
        ]
        assert "receipts" in record["era_echoes"]
        assert record["support"]["paragraphs"] > 0
        assert record["support"]["speeches"] > 0
        assert record["support"]["invocation_rows"] == 1_447
    assert all(
        record["era_defined"]["status"] == "authored"
        and record["era_defined"]["visual_kind"] == "combined_trajectory"
        and record["era_defined"]["focal_index"] == index
        and len(record["era_defined"]["rows"]) == 4
        for index, record in enumerate(all_contextualizations.values())
    )
    for record in all_contextualizations.values():
        echoes = record["era_echoes"]
        assert set(echoes["directional_views"]) == {"incoming", "outgoing"}
        assert echoes["default_direction"] in {"incoming", "outgoing"}
        assert any(
            view["available"]
            for view in echoes["directional_views"].values()
        )


def test_founding_definition_uses_equal_era_historical_average(
    all_contextualizations,
):
    chart = all_contextualizations["founding"]["era_defined"]
    assert chart["visual_kind"] == "combined_trajectory"
    assert chart["scale_max"] == 35
    assert chart["scale_step"] == 5
    assert chart["observed_max_share"] == pytest.approx(31.5203643)
    rows = {
        row["key"]: row
        for row in chart["rows"]
    }
    assert list(rows) == ["native", "abroad", "union", "finance"]
    assert rows["abroad"]["aggregate_share"] == pytest.approx(26.3929619)
    assert rows["abroad"]["historical_average_share"] == pytest.approx(
        10.9909367
    )
    assert rows["abroad"]["president_average_share"] == pytest.approx(
        30.3794334
    )
    assert rows["native"]["aggregate_share"] == pytest.approx(18.3284457)
    assert rows["native"]["historical_average_share"] == pytest.approx(
        1.4839746
    )
    assert rows["union"]["aggregate_share"] == pytest.approx(20.8211144)
    assert rows["union"]["historical_average_share"] == pytest.approx(
        12.5158730
    )
    assert rows["finance"]["aggregate_share"] == pytest.approx(12.3167155)
    assert rows["finance"]["historical_average_share"] == pytest.approx(
        15.8298654
    )
    assert rows["finance"]["difference_pp"] < 0
    coverage = all_contextualizations["founding"]["era_defined"]["coverage"]
    assert coverage["founding_summed_share"] == pytest.approx(
        sum(row["aggregate_share"] for row in rows.values())
    )
    assert coverage["founding_count"] == 485
    assert coverage["founding_denominator"] == 682
    assert coverage["founding_share"] == pytest.approx(71.1143695)
    assert coverage["historical_average_share"] == pytest.approx(
        38.8770245
    )
    assert coverage["historical_summed_share"] == pytest.approx(
        40.8206497
    )
    assert coverage["historical_summed_share"] == pytest.approx(
        sum(row["historical_average_share"] for row in rows.values())
    )
    assert coverage["founding_count"] < sum(
        row["aggregate_count"] for row in rows.values()
    )
    assert len(coverage["historical_era_values"]) == 8
    assert all(
        len(row["historical_era_values"]) == 8
        and len(row["era_values"]) == 9
        and len(row["president_values"]) == 3
        for row in rows.values()
    )
    assert all(
        [value["era_key"] for value in row["era_values"]]
        == [spec.key for spec in era_profiles.ERA_PROFILE_SPECS]
        for row in rows.values()
    )


def test_expansion_definition_uses_the_shared_four_line_trajectory(
    all_contextualizations,
):
    chart = all_contextualizations["expansion"]["era_defined"]
    assert chart["visual_kind"] == "combined_trajectory"
    assert chart["focal_index"] == 1
    assert chart["headline"] == (
        "Diplomacy, institutions, war, and continental power"
    )
    values = {
        row["key"]: row["aggregate_share"]
        for row in chart["rows"]
    }
    assert values["institutions"] == pytest.approx(23.5011384)
    assert values["territory"] == pytest.approx(23.2987604)
    assert values["diplomacy"] == pytest.approx(17.1768277)
    assert values["war"] == pytest.approx(15.4566152)
    assert all(len(row["era_values"]) == 9 for row in chart["rows"])
    assert "historically declared topic families" in chart["measure_note"]
    assert "shares are independent" in chart["measure_note"]
    body = site._era_defined_html(all_contextualizations["expansion"])
    assert body.count('class="era-defined-combined-line"') == 4
    assert body.count("era-defined-combined-node") == 36
    assert 'class="era-defined-focal-band"' in body
    assert "Continental republic paragraphs" in body
    assert "diplomatic trunk" not in body


def test_topic_life_reuses_each_profiles_six_major_topics(
    all_contextualizations,
):
    profiles = era_profiles.load_all_era_profiles()
    for key, record in all_contextualizations.items():
        assert [
            row["topic"] for row in record["topic_life"]["rows"]
        ] == [
            row["topic"] for row in profiles[key]["major_topics"]
        ]
        for row in record["topic_life"]["rows"]:
            assert "canonical_annotation_id" in row
            assert "annotation_history" in row
            for value in row["values"]:
                assert value["share"] == pytest.approx(
                    value["count"] / value["denominator"] * 100
                )
    founding = {
        row["label"]: row
        for row in all_contextualizations["founding"]["topic_life"]["rows"]
    }
    assert (
        founding["Indian & Native affairs"]["attention_pattern"]
        == "Fades"
    )
    assert (
        founding["Barbary & War of 1812"]["attention_pattern"]
        == "Fades"
    )
    assert (
        founding["Federal law enforcement"]["attention_pattern"]
        == "Persists + returns"
    )
    assert (
        founding["Executive power & courts"]["attention_pattern"]
        == "Persists"
    )
    finance = founding["Public finance"]
    assert (
        all_contextualizations["founding"]["topic_life"]["scale_mode"]
        == "shared_percent"
    )
    assert (
        all_contextualizations["founding"]["topic_life"]["scale_max"]
        == 20
    )
    assert (
        finance["source_topic"]
        == "Public Debt, Revenue & Treasury Finance"
    )
    assert finance["values"][0]["share"] == pytest.approx(8.9442815)
    assert finance["successor_values"][0]["share"] == pytest.approx(
        0.4398827
    )
    assert finance["successor_values"][6]["share"] == pytest.approx(
        7.8106682
    )
    naval = founding["Barbary & War of 1812"]
    assert naval["attention_pattern"] == "Fades"
    assert naval["successor_values"] is None
    assert naval["successor_topics"] == []
    assert naval["interpretation"] is None
    assert naval["reframe_gate"]["passes"] is False
    assert naval["reframe_gate"]["spearman_correlation"] == pytest.approx(
        -0.3
    )
    assert naval["reframe_gate"]["successor_growth_pp"] == pytest.approx(
        -1.3747415
    )
    assert finance["reframe_gate"]["passes"] is True
    assert finance["reframe_gate"]["spearman_correlation"] == pytest.approx(
        -0.7666667
    )
    assert finance["reframe_switch_index"] == 5
    assert finance["reframe_switch_era_key"] == "war-new-deal"
    assert finance["scale_max"] == pytest.approx(10.6875618)
    assert finance["interpretation"] == {
        "annotation_id": (
            "topic-life-founding-public-finance-20260726t161338z"
        ),
        "recorded_at": "2026-07-26T16:13:38Z",
        "annotator": {
            "type": "ai",
            "model": "gpt-5",
            "reasoning_effort": "high",
        },
        "label": "Reframes",
        "successor_kind": "level2",
        "successor_topic": "Taxes, Budget Deficits & Federal Spending",
        "successor_display": "taxes, deficits & spending",
        "successor_topics": [
            "Taxes, Budget Deficits & Federal Spending",
        ],
        "rationale": (
            "Debt, revenue, and Treasury language declines while taxes, "
            "deficits, and spending become the larger public-finance "
            "vocabulary."
        ),
        "validation_status": "unvalidated",
    }


def test_era_echoes_are_topic_networks_with_actor_receipts(
    all_contextualizations,
):
    founding = all_contextualizations["founding"]["era_echoes"]
    assert founding["direction_mode"] == "incoming"
    assert founding["outgoing"]["reference_paragraphs"] == 0
    assert founding["incoming"]["reference_paragraphs"] == 218
    assert founding["incoming"]["raw_mentions"] == 275
    assert founding["incoming"]["speeches"] == 126
    first_topic = founding["incoming"]["topics"][0]
    assert first_topic["topic"] == "Providence, Faith & American Ideals"
    assert first_topic["paragraphs"] == 55
    assert first_topic["share"] == pytest.approx(55 / 218 * 100)
    assert [receipt["title"] for receipt in founding["receipts"]] == [
        "Founding figures invoked",
        "Presidents looking back",
    ]
    assert founding["receipts"][0]["items"][0] == {
        "name": "Thomas Jefferson",
        "raw_mentions": 148,
        "paragraphs": 126,
        "speeches": 82,
    }
    assert (
        founding["receipts"][1]["items"][0]["name"]
        == "Ronald Reagan"
    )
    network = founding["network"]
    assert network["reference_paragraphs"] == 218
    assert network["raw_mentions"] == 275
    assert network["speeches"] == 126
    assert [node["label"] for node in network["source_nodes"][:3]] == [
        "Ronald Reagan",
        "Abraham Lincoln",
        "Calvin Coolidge",
    ]
    assert network["topic_nodes"][0]["label"] == "Faith & national ideals"
    assert [node["label"] for node in network["target_nodes"]] == [
        "Thomas Jefferson",
        "George Washington",
        "John Adams",
    ]
    assert any(
        path["speaker"] == "Ronald Reagan"
        and path["target"] == "Thomas Jefferson"
        for path in network["paths"]
    )
    assert network["source_target_links"][0]["source"] == (
        "Other invoking presidents"
    )
    assert network["source_target_links"][0]["target"] == "Thomas Jefferson"
    gravity = network["gravity"]
    assert gravity["layout"] == "connection_weighted_gravity"
    assert [
        node["label"] for node in gravity["anchor_nodes"]
    ] == [
        "George Washington",
        "John Adams",
        "Thomas Jefferson",
    ]
    assert [
        node["reference_paragraphs"] for node in gravity["anchor_nodes"]
    ] == [100, 18, 126]
    assert len(gravity["president_satellites"]) == 32
    assert len(gravity["topic_satellites"]) == 40
    assert all(
        node["key"] != "Other invoking presidents"
        for node in gravity["president_satellites"]
    )
    reagan = next(
        node
        for node in gravity["president_satellites"]
        if node["key"] == "Ronald Reagan"
    )
    assert sum(
        connection["weight_share"]
        for connection in reagan["connections"]
    ) == pytest.approx(100)
    assert {
        connection["anchor"]: connection["reference_paragraphs"]
        for connection in reagan["connections"]
    } == {
        "George Washington": 9,
        "John Adams": 2,
        "Thomas Jefferson": 23,
    }
    faith = next(
        node
        for node in gravity["topic_satellites"]
        if node["key"] == "Providence, Faith & American Ideals"
    )
    assert faith["president_connections"][0] == {
        "label": "Ronald Reagan",
        "reference_paragraphs": 11,
        "raw_mentions": 15,
        "speeches": 6,
    }
    assert {
        row["label"] for row in faith["president_connections"]
    } == {
        path["speaker"]
        for path in network["paths"]
        if path["topic"] == "Providence, Faith & American Ideals"
    }
    present = all_contextualizations["present"]["era_echoes"]
    assert present["incoming"]["reference_paragraphs"] == 0
    assert present["outgoing"]["reference_paragraphs"] > 0
    assert all(
        row["source_eras"] == ["present"]
        for row in present["outgoing"]["topics"]
    )
    for record in all_contextualizations.values():
        echoes = record["era_echoes"]
        for direction in ("incoming", "outgoing"):
            view = echoes["directional_views"][direction]
            assert view["network"]["reference_paragraphs"] == (
                echoes[direction]["reference_paragraphs"]
            )
            assert view["available"] is (
                echoes[direction]["reference_paragraphs"] > 0
            )
    cold_war_body = site._era_echoes_html(
        all_contextualizations["cold-war"]
    )
    assert cold_war_body.count('class="era-echo-gravity-chart"') == 2
    assert 'data-echo-direction="incoming"' in cold_war_body
    assert 'data-echo-direction="outgoing"' in cold_war_body
    assert "Invoked by" in cold_war_body
    assert "Invoking" in cold_war_body


def test_era_echo_gravity_layout_preserves_weighted_meaning_and_spacing(
    all_contextualizations,
):
    gravity = all_contextualizations["founding"]["era_echoes"]["network"][
        "gravity"
    ]
    layout = site._era_echo_gravity_layout(gravity)
    anchors = {node["key"]: node for node in layout["anchors"]}
    assert (
        anchors["Thomas Jefferson"]["radius"]
        > anchors["George Washington"]["radius"]
        > anchors["John Adams"]["radius"]
    )
    reagan = next(
        node
        for node in layout["satellites"]
        if node["key"] == "Ronald Reagan"
    )
    total = sum(
        connection["reference_paragraphs"]
        for connection in reagan["connections"]
    )
    expected_x = sum(
        anchors[connection["anchor"]]["x"]
        * connection["reference_paragraphs"]
        for connection in reagan["connections"]
    ) / total
    expected_y = sum(
        anchors[connection["anchor"]]["y"]
        * connection["reference_paragraphs"]
        for connection in reagan["connections"]
    ) / total
    assert reagan["semantic_x"] == pytest.approx(expected_x)
    assert reagan["semantic_y"] == pytest.approx(expected_y)
    for index, left in enumerate(layout["satellites"]):
        for right in layout["satellites"][index + 1:]:
            assert hypot(
                left["x"] - right["x"],
                left["y"] - right["y"],
            ) + .1 >= left["radius"] + right["radius"]
        for anchor in layout["anchors"]:
            assert hypot(
                left["x"] - anchor["x"],
                left["y"] - anchor["y"],
            ) + .1 >= left["radius"] + anchor["radius"]


def test_one_renderer_handles_every_era_with_unique_context_screen_ids(
    all_contextualizations,
):
    rendered = []
    for key, record in all_contextualizations.items():
        body = site._era_contextualize_html(
            record,
            panel_id=f"test-{key}-contextualize",
        )
        assert f'id="test-{key}-contextualize"' in body
        assert f'id="{key}-screen-defined"' in body
        assert f'id="{key}-screen-echoes"' in body
        assert f'id="{key}-screen-topic-life"' not in body
        assert body.count("data-founding-screen-tab=") == 2
        assert "Era Defined" in body
        assert "Topic Life" not in body
        assert "Era Echoes" in body
        assert "Next era" not in body
        rendered.append(body)
    joined = "".join(rendered)
    for spec in era_profiles.ERA_PROFILE_SPECS:
        assert joined.count(f'id="{spec.key}-screen-defined"') == 1
    founding = site._era_contextualize_html(
        all_contextualizations["founding"],
        panel_id="founding-contextualize-contract",
    )
    retained_topic_life = site._topic_life_html(
        all_contextualizations["founding"]
    )
    assert "topic-life-family-ring" in retained_topic_life
    assert "AI · unvalidated" in retained_topic_life
    assert "topic-life-family-ring" not in founding
    assert "later-era average" not in founding
    assert "Founding 26.4%" not in founding
    assert "Defend treaty rights, neutral commerce" not in founding
    assert 'class="era-defined-combined-chart"' in founding
    assert founding.count('class="era-defined-combined-chart"') == 1
    assert "era-defined-trajectory-row" not in founding
    assert founding.count('class="era-defined-combined-line"') == 4
    assert founding.count('class="era-defined-combined-hit"') == 4
    assert founding.count("era-defined-combined-node") == 36
    assert founding.count('class="era-defined-point-value"') == 36
    assert ">Native power</text>" in founding
    assert ">Native power 0.0%</text>" not in founding
    assert ">Union 10.6%</text>" not in founding
    assert "Shared scale: 0–35%" in founding
    assert "largest observed value is\n  31.5%" in founding
    assert "Text alternative" not in founding
    assert '<details class="era-defined-combined-method">' in founding
    assert "<summary>How the measure works</summary>" in founding
    assert "era-defined-common" not in founding
    assert "Pale difference" not in founding
    assert "Who invoked the Founding—and around what?" in founding
    assert "Reframes →\n      broader war" not in retained_topic_life
    assert "Fades" in retained_topic_life
    assert "Spearman ≤ -0.50" in retained_topic_life
    assert "“Reframes” is AI · unvalidated" in retained_topic_life
    assert "Every row uses the same 0–20%" in retained_topic_life
    assert "era-defined-coverage-line" in founding
    assert "overlapping family shares sum to" in founding
    assert founding.count('class="era-echo-gravity-chart"') == 1
    assert founding.count("era-echo-gravity-satellite is-president") == 32
    assert founding.count("era-echo-gravity-satellite is-topic") == 40
    expected_topic_president_pairs = {
        (path["topic"], path["speaker"])
        for path in all_contextualizations["founding"]["era_echoes"][
            "network"
        ]["paths"]
    }
    assert founding.count(
        'class="era-echo-topic-president-link"'
    ) == len(expected_topic_president_pairs)
    assert 'data-echo-gravity' in founding
    assert 'data-topic-president-link' in founding
    assert 'data-president-keys=' in founding
    assert "Invoking presidents:" in founding
    assert "Distance = share of connected paragraphs" in founding
    assert "shared pull" in founding
    assert 'portraits/thomas-jefferson.png' in founding
    assert "data-echo-cluster-tab" not in founding
    assert "Ronald Reagan" in founding
    assert "Thomas Jefferson" in founding


def test_one_shared_workspace_exposes_the_same_four_views_for_every_era(
    all_contextualizations,
):
    profiles_by_era = era_profiles.load_all_era_profiles()
    visualizations_by_era = era_visualizations.load_all_era_visualizations()
    rendered = []
    for spec in era_profiles.ERA_PROFILE_SPECS:
        body = site._era_workspace_html(
            profiles_by_era[spec.key],
            visualizations_by_era[spec.key],
            all_contextualizations[spec.key],
        )
        assert f'id="{spec.key}-workspace"' in body
        assert body.count("data-era-workspace-tab=") == 4
        assert body.count("founding-story-card founding-panel") == 4
        assert body.count("data-founding-screen-group") == 0
        assert f'id="{spec.key}-profile"' in body
        assert f'id="{spec.key}-screen-defined"' in body
        assert f'id="{spec.key}-screen-adversaries"' in body
        assert f'id="{spec.key}-screen-echoes"' in body
        assert "Era profile" in body
        assert "Era defined" in body
        assert "Adversaries" in body
        assert "Era echoes" in body
        assert "Presidential agendas" not in body
        assert 'data-echo-direction="incoming"' in body
        assert 'data-echo-direction="outgoing"' in body
        assert body.count('class="era-defined-combined-line"') == 4
        rendered.append(body)
    joined = "".join(rendered)
    assert joined.count('class="founding-workspace era-workspace"') == 9


def test_shared_workspace_is_the_only_story_body_for_an_era(
    all_contextualizations,
):
    profiles_by_era = era_profiles.load_all_era_profiles()
    visualizations_by_era = era_visualizations.load_all_era_visualizations()
    workspace = site._era_workspace_html(
        profiles_by_era["expansion"],
        visualizations_by_era["expansion"],
        all_contextualizations["expansion"],
    )
    bodies = {key: "" for key, *_ in site.SECTIONS[:9]}
    page = site.build_html(
        {},
        {
            "speeches": 1_057,
            "words": 4_179_266,
            "presidents": 45,
            "start": 1789,
            "end": 2026,
        },
        bodies,
        inline=False,
        era_profile_sections={"expansion_conflict": workspace},
        page_kind="story",
    )
    assert ".era-defined-combined-series:not(:hover)" in page
    assert ".era-defined-combined-hit" in page
    assert "stroke-width:4.4" in page
    assert (
        "grid-template-columns:repeat(4,minmax(0,1fr));gap:0;"
        in page
    )
    assert 'data-era-panel="defined"' in page
    assert 'data-era-panel="adversaries"' in page
    assert 'data-era-panel="echoes"' in page
    assert ".era-defined-point-value" in page
    expansion_start = page.index('<section id="expansion_conflict"')
    expansion_end = page.index('<section id="union_crisis"', expansion_start)
    section = page[expansion_start:expansion_end]
    assert section.count('id="expansion-workspace"') == 1
    assert "data-existing-story" not in section
    assert "<blockquote>" not in section


def test_topic_life_annotation_history_is_timestamped_and_fail_closed(
    tmp_path,
):
    taxonomy = attention.load_taxonomy()
    profiles_by_era = era_profiles.load_all_era_profiles()
    records = era_contextualizations.load_topic_life_annotations(
        taxonomy,
        profiles_by_era,
    )
    assert set(records) == {
        (
            "founding",
            "Early Naval Wars: Barbary & the War of 1812",
        ),
        ("founding", "Public finance"),
    }
    assert records[(
        "founding",
        "Early Naval Wars: Barbary & the War of 1812",
    )]["validation_status"] == "rejected"
    assert records[("founding", "Public finance")][
        "validation_status"
    ] == "unvalidated"
    assert records[("founding", "Public finance")]["annotation_id"] == (
        "topic-life-founding-public-finance-20260726t161338z"
    )
    assert len(
        records[("founding", "Public finance")]["annotation_history"]
    ) == 1
    assert records[("founding", "Public finance")]["annotator"] == {
        "type": "ai",
        "model": "gpt-5",
        "reasoning_effort": "high",
    }

    payload = json.loads(
        era_contextualizations.TOPIC_LIFE_ANNOTATIONS_PATH.read_text()
    )
    latest = dict(payload["records"][1])
    latest["annotation_id"] = "topic-life-founding-public-finance-review"
    latest["recorded_at"] = "2026-07-27T12:00:00Z"
    latest["validation_status"] = "validated"
    latest["annotator"] = {
        "type": "human",
        "name": "Review editor",
    }
    payload["records"].append(latest)
    history_path = tmp_path / "topic_life_annotations_v1.json"
    history_path.write_text(json.dumps(payload))
    current = era_contextualizations.load_topic_life_annotations(
        taxonomy,
        profiles_by_era,
        history_path,
        verify_fingerprints=False,
    )[("founding", "Public finance")]
    assert current["annotation_id"] == latest["annotation_id"]
    assert current["validation_status"] == "validated"
    assert len(current["annotation_history"]) == 2

    payload["records"][0]["validation_status"] = "pretend-validated"
    bad = tmp_path / "bad_topic_life_annotations_v1.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="validation_status"):
        era_contextualizations.load_topic_life_annotations(
            taxonomy,
            profiles_by_era,
            bad,
            verify_fingerprints=False,
        )
    missing_model = json.loads(
        era_contextualizations.TOPIC_LIFE_ANNOTATIONS_PATH.read_text()
    )
    del missing_model["records"][0]["annotator"]["model"]
    bad_ai = tmp_path / "bad_ai_topic_life_annotations_v1.json"
    bad_ai.write_text(json.dumps(missing_model))
    with pytest.raises(ValueError, match="require model"):
        era_contextualizations.load_topic_life_annotations(
            taxonomy,
            profiles_by_era,
            bad_ai,
            verify_fingerprints=False,
        )


def test_contextualizations_publish_as_one_switchable_json_file(
    all_contextualizations,
    tmp_path,
):
    path = era_contextualizations.write_era_contextualizations(
        all_contextualizations,
        tmp_path,
    )
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == "era-contextualizations-v10"
    assert payload["era_order"] == list(all_contextualizations)
    assert (
        payload["contextualizations"]["founding"]["topic_life"][
            "focal_index"
        ]
        == 0
    )


def test_one_argument_loader_only_selects_the_requested_era(
    all_contextualizations,
    monkeypatch,
):
    monkeypatch.setattr(
        era_contextualizations,
        "load_all_era_contextualizations",
        lambda: all_contextualizations,
    )
    assert era_contextualizations.load_era_contextualization(
        "founding"
    ) is all_contextualizations["founding"]
    assert era_contextualizations.load_era_contextualization(
        "The Cold War"
    ) is all_contextualizations["cold-war"]
