"""All-era derivation, rendering, and publication contracts for shared graphs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from presidential_profiles import (
    attention,
    corpus,
    era_profiles,
    era_visualizations,
    site,
)


DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def all_visualizations() -> dict[str, dict]:
    return era_visualizations.build_era_visualizations(
        corpus.load(),
        pd.read_parquet(DATA / "paragraphs.parquet"),
        pd.read_parquet(
            DATA / "llm_annotations" / "paragraph_annotations.parquet"
        ),
        pd.read_parquet(
            DATA / "llm_annotations" / "speech_annotations.parquet"
        ),
        pd.read_parquet(
            DATA / "llm_annotations" / "paragraph_entities.parquet"
        ),
        attention.load_taxonomy(),
    )


def test_all_nine_eras_share_one_complete_visualization_contract(
    all_visualizations,
):
    assert list(all_visualizations) == [
        spec.key for spec in era_profiles.ERA_PROFILE_SPECS
    ]
    for spec in era_profiles.ERA_PROFILE_SPECS:
        visualization = all_visualizations[spec.key]
        assert visualization["years"] == (
            f"{spec.start_year}–{spec.end_year}"
        )
        assert visualization["presidents"]
        agenda = visualization["agenda"]
        assert len(agenda["categories"]) == 12
        assert len(agenda["compositions"]) == len(
            visualization["presidents"]
        )
        assert all(category["definition"] for category in agenda["categories"])
        assert all(
            [segment["key"] for segment in row["segments"]]
            == row["top_policy_domains"]
            and 1 <= len(row["segments"]) <= 5
            and len(row["domain_presence"]) == 12
            and [segment["president_rank"] for segment in row["segments"]]
            == list(range(1, len(row["segments"]) + 1))
            and all(segment["kind"] == "policy" for segment in row["segments"])
            and all(segment["definition"] for segment in row["segments"])
            and row["displayed_share_sum"]
            == pytest.approx(
                sum(segment["share"] for segment in row["segments"])
            )
            and all(0 <= segment["share"] <= 100 for segment in row["segments"])
            for row in agenda["compositions"]
        )
        assert {
            "comparison_domains",
            "comparison_categories",
            "remaining_policy_domains",
            "scale_max",
            "thin_scale_max",
        }.isdisjoint(agenda)
        assert all(
            {"remaining_policy", "non_policy"}.isdisjoint(row)
            for row in agenda["compositions"]
        )
        assert visualization["governance"]["presidents"]
        assert len(visualization["governance"]["audience_options"]) == 4
        assert len(visualization["governance"]["medium_options"]) == 4
        assert visualization["adversaries"]["president_summary"]
        assert visualization["adversaries"]["edges"]
        assert visualization["support"]["speeches"] > 0
        assert visualization["support"]["paragraphs"] > 0


def test_each_graph_uses_era_local_denominators(all_visualizations):
    assert sum(
        record["support"]["speeches"]
        for record in all_visualizations.values()
    ) == 1_057
    assert sum(
        record["support"]["paragraphs"]
        for record in all_visualizations.values()
    ) == 36_229
    for visualization in all_visualizations.values():
        paragraph_total = visualization["support"]["paragraphs"]
        speech_total = visualization["support"]["speeches"]
        agenda = visualization["agenda"]
        assert sum(
            row["paragraphs"] for row in agenda["president_support"]
        ) == paragraph_total
        for composition in agenda["compositions"]:
            support = next(
                row
                for row in agenda["president_support"]
                if row["president"] == composition["president"]
            )
            assert all(
                segment["denominator_paragraphs"] == support["paragraphs"]
                and segment["count"] <= support["paragraphs"]
                for segment in composition["segments"]
            )
            assert composition["unassigned"]["count"] <= support["paragraphs"]
        governance = visualization["governance"]
        assert sum(
            row["count"] for row in governance["audience_options"]
        ) == speech_total
        assert sum(
            row["count"] for row in governance["medium_options"]
        ) == speech_total
        assert sum(
            row["speeches"] for row in governance["presidents"]
        ) == speech_total


def test_founding_graphs_use_the_individual_agenda_display(
    all_visualizations,
):
    founding = all_visualizations["founding"]
    assert [row["name"] for row in founding["presidents"]] == [
        "George Washington",
        "John Adams",
        "Thomas Jefferson",
    ]
    assert [
        row["label"]
        for row in founding["agenda"]["categories"]
    ] == [
        "Diplomacy",
        "War & military",
        "Money & banking",
        "Trade & tariffs",
        "Public finance",
        "Economy & welfare",
        "Land & agriculture",
        "Infrastructure",
        "Civil rights",
        "Native affairs",
        "Law & justice",
        "Constitutional order",
    ]
    washington = next(
        row
        for row in founding["agenda"]["compositions"]
        if row["president"] == "George Washington"
    )
    assert [
        (segment["label"], segment["count"])
        for segment in washington["segments"]
    ] == [
        ("Constitutional order", 59),
        ("Native affairs", 57),
        ("Diplomacy", 49),
        ("Law & justice", 36),
        ("War & military", 30),
    ]
    assert washington["displayed_share_sum"] == pytest.approx(91.3043478)
    assert washington["unassigned"]["count"] == 0
    washington = founding["governance"]["presidents"][0]
    assert washington["routes"][0] == {
        "audience": "Congress",
        "medium": "Written message",
        "count": 10,
        "share": pytest.approx(47.6190476),
    }
    jefferson_edge = next(
        edge
        for edge in founding["adversaries"]["edges"]
        if edge["president"] == "Thomas Jefferson"
        and edge["adversary"] == "Great Britain"
    )
    assert jefferson_edge["paragraphs"] == 14


def test_agenda_rows_separate_thin_presidential_records(
    all_visualizations,
):
    expansion = all_visualizations["expansion"]
    agenda = expansion["agenda"]
    assert agenda["supported_presidents"] == [
        "James Madison",
        "James Monroe",
        "John Quincy Adams",
        "Andrew Jackson",
        "Martin Van Buren",
        "John Tyler",
        "James K. Polk",
    ]
    assert agenda["thin_presidents"] == [
        "William Harrison",
        "Zachary Taylor",
    ]
    assert agenda["minimum_supported_speeches"] == 5
    assert "may sum above 100%" in agenda["measure"]
    assert all(
        row["status"] == (
            "thin"
            if row["president"] in {"William Harrison", "Zachary Taylor"}
            else "supported"
        )
        for row in agenda["president_support"]
    )
    body = site._era_presidential_agendas_html(expansion)
    assert "Thin support · William Harrison, Zachary Taylor" in body
    assert "1 speech · 57 paragraphs" in body
    assert "2 speeches · 62 paragraphs" in body
    assert "remain separate supporting evidence" in body
    assert body.count("data-agenda-cards") == 1
    assert body.count("data-agenda-thin-cards") == 1


def test_agenda_counts_multilabel_paragraphs_in_each_applicable_domain():
    taxonomy = attention.load_taxonomy()
    frame = pd.DataFrame({
        "president": ["Test President"] * 3,
        "word_count": [100, 50, 50],
        "topic_set": [
            frozenset({
                "Treaties, Diplomacy & International Arbitration",
                "Military Preparedness, Armed Forces & Veterans",
            }),
            frozenset({
                "Treaties, Diplomacy & International Arbitration",
                "Monroe Doctrine & Latin American Policy",
            }),
            frozenset(),
        ],
    })
    agenda = era_visualizations._agenda(
        frame,
        [{"name": "Test President", "speeches": 5}],
        taxonomy,
    )
    composition = agenda["compositions"][0]
    shares = {
        segment["key"]: segment["share"]
        for segment in composition["segments"]
    }
    assert shares["Foreign Relations & Diplomacy"] == pytest.approx(200 / 3)
    assert shares["War & Military Affairs"] == pytest.approx(100 / 3)
    assert sum(shares.values()) == pytest.approx(100)
    assert composition["unassigned"]["share"] == pytest.approx(100 / 3)
    diplomacy = next(
        row
        for row in composition["domain_presence"]
        if row["key"] == "Foreign Relations & Diplomacy"
    )
    assert diplomacy["word_presence_share"] == pytest.approx(75)
    assert diplomacy["count"] == 2


def test_monroe_agenda_uses_paragraph_presence_with_word_sensitivity(
    all_visualizations,
):
    agenda = all_visualizations["expansion"]["agenda"]
    monroe = next(
        row
        for row in agenda["compositions"]
        if row["president"] == "James Monroe"
    )
    shares = {
        segment["key"]: segment["share"]
        for segment in monroe["segments"]
    }
    assert shares["Foreign Relations & Diplomacy"] == pytest.approx(
        40.5405405
    )
    assert shares["War & Military Affairs"] == pytest.approx(20.8845209)
    assert shares["Public Finance & Taxation"] == pytest.approx(12.5307125)
    assert shares["Trade, Commerce & Tariffs"] == pytest.approx(11.5479115)
    assert shares["Constitutional Order & Governance"] == pytest.approx(
        10.3194103
    )
    assert list(shares) == [
        "Foreign Relations & Diplomacy",
        "War & Military Affairs",
        "Public Finance & Taxation",
        "Trade, Commerce & Tariffs",
        "Constitutional Order & Governance",
    ]
    assert monroe["unassigned"]["share"] == pytest.approx(0.2457002)
    assert monroe["weighting_sensitivity"] == {
        "leading_domain_same": True,
        "top_five_set_same": True,
    }


def test_one_renderer_handles_every_era_with_unique_interaction_ids(
    all_visualizations,
):
    rendered = []
    for key, visualization in all_visualizations.items():
        body = site._era_visualize_html(
            visualization,
            panel_id=f"test-{key}-visualize",
        )
        assert f'id="test-{key}-visualize"' in body
        assert f'id="{key}-screen-agenda"' in body
        assert f'id="{key}-screen-adversaries"' in body
        assert f'id="{key}-screen-governance"' not in body
        assert body.count("data-founding-screen-tab=") == 2
        assert body.count("data-agenda-cards") == 1
        assert "data-agenda-presence" not in body
        assert "data-governance-filter" not in body
        assert "Governing channels" not in body
        assert "founding-adversary-network" in body
        assert "Text alternative · paragraph and raw-word presence shares" not in body
        assert '<details class="agenda-method-note">' in body
        assert "<summary>What the percentages mean</summary>" in body
        assert body.count('class="agenda-card-description"') == (
            body.count('class="agenda-card-priority"')
        )
        supported_count = len(
            visualization["agenda"]["supported_presidents"]
        )
        scroll_marker = (
            'class="agenda-card-grid agenda-card-grid-scroll" '
            'data-agenda-cards role="region"'
        )
        assert (scroll_marker in body) is (supported_count > 3)
        rendered.append(body)
    joined = "".join(rendered)
    for spec in era_profiles.ERA_PROFILE_SPECS:
        assert joined.count(f'id="{spec.key}-screen-agenda"') == 1


def test_adversary_network_exposes_connected_hover_and_focus_targets(
    all_visualizations,
):
    visualization = all_visualizations["civil-war-reconstruction"]
    body = site._era_adversary_network_html(visualization)
    edges = visualization["adversaries"]["edges"]
    adversaries = {row["adversary"] for row in edges}
    assert body.count("data-adversary-edge") == len(edges)
    assert body.count("data-adversary-node") == (
        len(visualization["presidents"]) + len(adversaries)
    )
    assert body.count('tabindex="0"') == (
        len(edges) + len(visualization["presidents"]) + len(adversaries)
    )
    assert "data-president-key=" in body
    assert "data-adversary-key=" in body

    page = site.build_html(
        {},
        {
            "speeches": 1_057,
            "words": 4_179_266,
            "presidents": 45,
            "start": 1789,
            "end": 2026,
        },
        {key: "" for key, *_ in site.SECTIONS[:9]},
        inline=False,
        page_kind="story",
    )
    assert 'document.querySelectorAll("[data-adversary-network]")' in page
    assert '"pointerenter", () => activateAdversaryItem(item)' in page
    assert 'item.addEventListener("focus", () => activateAdversaryItem(item))' in page
    assert ".founding-adversary-network.has-active" in page
    assert 'stroke-width:calc(var(--adversary-edge-width) * 1.35)' in page


def test_visualizations_publish_as_one_switchable_json_file(
    all_visualizations,
    tmp_path,
):
    path = era_visualizations.write_era_visualizations(
        all_visualizations, tmp_path
    )
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == "era-visualizations-v8"
    assert payload["era_order"] == list(all_visualizations)
    assert (
        payload["visualizations"]["cold-war"]["support"]["speeches"]
        == 192
    )


def test_one_argument_loader_only_selects_the_requested_era(
    all_visualizations,
    monkeypatch,
):
    monkeypatch.setattr(
        era_visualizations,
        "load_all_era_visualizations",
        lambda: all_visualizations,
    )
    assert era_visualizations.load_era_visualization(
        "founding"
    ) is all_visualizations["founding"]
    assert era_visualizations.load_era_visualization(
        "The Cold War"
    ) is all_visualizations["cold-war"]
