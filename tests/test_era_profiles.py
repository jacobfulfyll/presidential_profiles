"""All-era profile derivation, switching, and rendering contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from presidential_profiles import (
    attention,
    corpus,
    era_profiles,
    site,
    story_foundation,
)


DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def all_profiles() -> dict[str, dict]:
    return era_profiles.build_era_profiles(
        corpus.load(),
        pd.read_parquet(DATA / "paragraphs.parquet"),
        pd.read_parquet(
            DATA / "llm_annotations" / "paragraph_annotations.parquet"
        ),
        pd.read_parquet(
            DATA / "llm_annotations" / "speech_annotations.parquet"
        ),
        attention.load_taxonomy(),
        story_foundation.load_story_foundation(),
    )


def test_specs_are_a_one_to_one_route_over_the_canonical_eras():
    assert [
        (spec.label, spec.start_year, spec.end_year)
        for spec in era_profiles.ERA_PROFILE_SPECS
    ] == list(era_profiles.STORY_ERAS)
    assert len({spec.key for spec in era_profiles.ERA_PROFILE_SPECS}) == 9
    assert len({spec.section_key for spec in era_profiles.ERA_PROFILE_SPECS}) == 9
    assert era_profiles.resolve_era("expansion").label == (
        "The continental republic"
    )
    assert era_profiles.resolve_era(
        "War, institutions & expansion"
    ).key == "expansion"
    assert era_profiles.resolve_era("The Cold War").key == "cold-war"
    with pytest.raises(KeyError, match="unknown era"):
        era_profiles.resolve_era("not-an-era")


def test_accession_boundary_speeches_follow_the_presidency_they_close():
    years = pd.Series([1809, 1809, 1869, 1869, 1953, 1953, 2017, 2017])
    presidents = pd.Series([
        "Thomas Jefferson",
        "James Madison",
        "Andrew Johnson",
        "Ulysses S. Grant",
        "Harry S. Truman",
        "Dwight D. Eisenhower",
        "Barack Obama",
        "Donald Trump",
    ])
    assert list(era_profiles.story_era_key_series(years, presidents)) == [
        "founding",
        "expansion",
        "civil-war-reconstruction",
        "gilded-age",
        "war-new-deal",
        "cold-war",
        "post-cold-war",
        "present",
    ]
    assert era_profiles.story_era_for_year(1809).key == "expansion"
    assert years.tolist() == [1809, 1809, 1869, 1869, 1953, 1953, 2017, 2017]
    with pytest.raises(ValueError, match="share an index"):
        era_profiles.story_era_key_series(
            years, presidents.set_axis(range(10, 18))
        )


def test_all_nine_profiles_have_the_complete_shared_contract(all_profiles):
    assert list(all_profiles) == [
        spec.key for spec in era_profiles.ERA_PROFILE_SPECS
    ]
    for spec in era_profiles.ERA_PROFILE_SPECS:
        profile = all_profiles[spec.key]
        assert profile["label"] == spec.label
        assert profile["years"] == f"{spec.start_year}–{spec.end_year}"
        assert profile["section_key"] == spec.section_key
        assert profile["presidents"]
        assert len(profile["distinctive_references"]["rows"]) == 5
        assert profile["distinctive_references"]["denominator"]["unit"] == (
            "speaker_audited_paragraphs"
        )
        landscape = profile["reference_landscape"]
        assert landscape["denominator"]["unit"] == "speaker_audited_paragraphs"
        assert [row["entity_type"] for row in landscape["types"]] == [
            "person", "institution", "group", "nation",
        ]
        assert landscape["selection"] == {
            "minimum_paragraphs": 5,
            "limited_record_minimum_paragraphs": 4,
            "limited_record_only_when_no_supported_highlight": True,
            "minimum_source_documents": 2,
            "minimum_favorable_or_neutral_share": .8,
            "excludes_displayed_named_adversaries": True,
            "one_highlight_per_type": True,
            "ner_supplies_stance": False,
        }
        adversary_names = {
            row["normalized_entity"] for row in profile["adversaries"]
        }
        for row in landscape["types"]:
            assert 0 <= row["paragraph_share"] <= 1
            assert 0 <= row["corpus_paragraph_share"] <= 1
            highlight = row["highlight"]
            if highlight is None:
                continue
            assert highlight["entity_type"] == row["entity_type"]
            assert highlight["normalized_entity"] not in adversary_names
            assert highlight["support_status"] in {"supported", "limited_record"}
            minimum_paragraphs = (
                5 if highlight["support_status"] == "supported" else 4
            )
            assert highlight["support"]["paragraphs"] >= minimum_paragraphs
            assert highlight["support"]["source_documents"] >= 2
            stance = highlight["stance_mix"]
            assert (
                stance["favorable"] + stance["neutral"]
            ) / sum(stance.values()) >= .8
        assert len(profile["adversaries"]) == 5
        assert len(profile["adversary_types"]) == 5
        assert len(profile["major_topics"]) == 6
        assert len(profile["distinctive_words"]) == 3
        assert profile["footprint"]["speeches"] > 0
        assert profile["footprint"]["paragraphs"] > 0
        assert profile["footprint"]["words"] > 0
        assert profile["footprint"]["unit"] == "source_document_corpus"
        assert profile["major_topics_receipt"]["unit"] == (
            "source_document_corpus"
        )
        assert profile["distinctive_words_receipt"]["unit"] == (
            "source_document_corpus"
        )
        assert profile["style"]["unit"] == "source_document_corpus"
        assert profile["style"]["audience_options"]
        assert profile["style"]["medium_options"]
        eligibility = profile["distinctive_eligibility"]
        for word in profile["distinctive_words"]:
            assert word["era_count"] + word["other_count"] >= (
                eligibility["min_corpus_uses"]
            )
            assert word["era_speech_count"] >= (
                eligibility["min_era_speeches"]
            )
            assert word["era_president_count"] >= (
                eligibility["min_era_presidents"]
            )


def test_progressives_depression_surfaces_american_legion_as_limited_record(
    all_profiles,
):
    group = next(
        row
        for row in all_profiles["progressives-depression"]["reference_landscape"]["types"]
        if row["entity_type"] == "group"
    )
    assert group["highlight"]["label"] == "American Legion"
    assert group["highlight"]["support_status"] == "limited_record"
    assert group["highlight"]["support"] == {
        "paragraphs": 4,
        "source_documents": 4,
    }


def test_disjoint_profile_footprints_reconcile_to_the_whole_corpus(all_profiles):
    assert sum(
        profile["footprint"]["speeches"] for profile in all_profiles.values()
    ) == 1057
    assert sum(
        profile["footprint"]["paragraphs"] for profile in all_profiles.values()
    ) == 36_229
    assert sum(
        profile["footprint"]["words"] for profile in all_profiles.values()
    ) == 4_179_266
    for share in ("speech_share", "paragraph_share", "word_share"):
        assert sum(
            profile["footprint"][share] for profile in all_profiles.values()
        ) == pytest.approx(100.0)


def test_founding_record_preserves_the_approved_profile(all_profiles):
    profile = all_profiles["founding"]
    assert profile["title"] == "Establishing the Republic"
    assert [row["name"] for row in profile["presidents"]] == [
        "George Washington",
        "John Adams",
        "Thomas Jefferson",
    ]
    assert [row["name"] for row in profile["adversaries"]] == [
        "France",
        "Great Britain",
        "Aaron Burr",
        "Spain",
        "Tripoli",
    ]
    assert [row["term"] for row in profile["distinctive_words"]] == [
        "militia",
        "information",
        "gentlemen",
    ]
    assert [row["label"] for row in profile["major_topics"]] == [
        "Indian & Native affairs",
        "Military preparedness",
        "Federal law enforcement",
        "Public finance",
        "Executive power & courts",
        "Barbary & War of 1812",
    ]
    assert [row["label"] for row in profile["distinctive_references"]["rows"]] == [
        "General Wilkinson",
        "Aaron Burr",
        "Cherokee Nation",
        "Tripoli",
        "French Republic",
    ]


def test_profiles_remove_legacy_constituency_fields(all_profiles):
    for profile in all_profiles.values():
        assert {
            "constituents", "constituency_status", "constituency_note"
        }.isdisjoint(profile)


def test_one_renderer_handles_every_profile_with_unique_ids(all_profiles):
    rendered = []
    rendered_by_key = {}
    for key, profile in all_profiles.items():
        body = site._era_profile_html(
            profile,
            panel_id=f"test-{key}-profile",
        )
        assert f'id="test-{key}-profile"' in body
        assert f">{profile['years']}</h3>" in body
        assert f"<strong>{profile['title'].replace('&', '&amp;')}</strong>" in body
        assert body.count('class="era-card-heading"') == 4
        highlight_count = sum(
            row["highlight"] is not None
            for row in profile["reference_landscape"]["types"]
        )
        assert body.count('class="era-reference-highlight"') == highlight_count
        assert body.count('class="era-reference-lane"') == 4
        assert body.count('class="era-reference-name"') == highlight_count
        assert "Distinctive era references</span></header>" in body
        assert (
            f'aria-labelledby="test-{key}-profile-reference-title"' in body
        )
        assert "Who and what enters the frame" not in body
        assert "Share of actual-president paragraphs naming each kind of reference" not in body
        assert "era-reference-track" not in body
        assert "AI</i>AI only" not in body
        assert "Positive Jeffreys-smoothed" not in body
        assert "Descending log odds" not in body
        assert "45-row reference CSV" not in body
        assert body.count('class="era-distinctive-word"') == 3
        assert body.count('class="era-adversary-bubble"') == 5
        assert body.count('class="era-president-portrait-link"') == len(
            profile["presidents"]
        )
        for president in profile["presidents"]:
            assert (
                f'href="presidents/{site.profiles.slug(president["name"])}.html"'
                in body
            )
        rendered.append(body)
        rendered_by_key[key] = body
    assert 'data-support-status="limited_record"' in rendered_by_key[
        "progressives-depression"
    ]
    assert "American Legion" in rendered_by_key["progressives-depression"]
    assert "Limited record" in rendered_by_key["progressives-depression"]
    assert ">AI only</span>" in rendered_by_key["progressives-depression"]
    assert 'class="era-reference-meta"' in rendered_by_key[
        "progressives-depression"
    ]
    joined = "".join(rendered)
    for spec in era_profiles.ERA_PROFILE_SPECS:
        assert joined.count(f'id="test-{spec.key}-profile"') == 1


def test_distinctive_word_rank_exposes_concentration_and_evidence(
    all_profiles,
):
    profile = all_profiles["civil-war-reconstruction"]
    words = profile["distinctive_words"]
    assert words[0]["term"] == "slavery"
    assert words[0]["era_count"] == 982
    assert words[0]["rate_ratio"] > 25
    assert words[0]["distinctiveness_score"] > words[1]["distinctiveness_score"]
    assert words[0]["evidence_weight"] > .99
    body = site._era_profile_html(profile, panel_id="ranking-profile")
    normalized = " ".join(body.split())
    assert "Rank #1</span>" in body
    assert (
        f'{words[0]["era_count"]:,} uses<br>'
        f'{words[0]["rate_ratio"]:.1f}× other eras' in body
    )
    assert "Evidence:" not in body
    assert f'{words[0]["era_speech_count"]} speeches' not in body
    assert "Score" not in body
    assert '<details class="era-distinctive-method">' in body
    assert '<details class="era-distinctive-method" open>' not in body
    assert "How the distinctive-word ranking works" in body
    assert "the same families as Explore" in normalized
    assert "slavery, slave, and slaves count together" in normalized
    assert "1 − exp(−era family uses ÷ 100)" in normalized
    assert "concentration beyond 25×" in normalized
    assert "The printed multiplier remains uncapped" in normalized
    assert "Presidential names are excluded" in normalized
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
        era_profile_sections={"union_crisis": body},
        page_kind="story",
    )
    assert "grid-auto-rows:auto" in page
    assert "min-height:230px;height:auto;overflow:hidden" in page


def test_corpus_footprint_words_shrink_without_wrapping(all_profiles):
    body = site._era_profile_html(
        all_profiles["civil-war-reconstruction"],
        panel_id="responsive-profile",
    )
    assert 'class="era-footprint-language"' in body
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
        era_profile_sections={"union_crisis": body},
        page_kind="story",
    )
    assert ".era-template-footprint { container-type:inline-size; }" in page
    assert "font:680 clamp(.54rem,3cqi,.9rem)/1.05 Georgia,serif" in page
    assert "white-space:nowrap;overflow-wrap:normal;hyphens:none" in page


def test_adversary_pack_scales_to_avoid_collisions_in_every_era(all_profiles):
    width, height = site.ERA_ADVERSARY_VIEWBOX
    for key, profile in all_profiles.items():
        layout = site._era_adversary_bubble_layout(profile["adversaries"])
        assert len(layout) == 5
        for bubble in layout:
            assert bubble["radius"] > 0
            assert bubble["x"] - bubble["radius"] >= 2
            assert bubble["x"] + bubble["radius"] <= width - 2
            assert bubble["y"] - bubble["radius"] >= 2
            assert bubble["y"] + bubble["radius"] <= height - 2
        for index, left in enumerate(layout):
            for right in layout[index + 1:]:
                distance = math.hypot(
                    right["x"] - left["x"],
                    right["y"] - left["y"],
                )
                assert distance >= (
                    left["radius"] + right["radius"] + 2 - 1e-9
                ), key


def test_profiles_publish_as_one_switchable_json_file(all_profiles, tmp_path):
    path = era_profiles.write_era_profiles(all_profiles, tmp_path)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == "era-profile-v6"
    assert payload["profile_order"] == list(all_profiles)
    assert payload["profiles"]["expansion"]["years"] == "1809–1849"


def test_president_names_never_rank_as_era_vocabulary(all_profiles):
    names = [
        president["name"]
        for profile in all_profiles.values()
        for president in profile["presidents"]
    ]
    excluded = era_profiles._president_name_terms(names)
    assert {"kennedy", "nixon", "bush", "trump"}.issubset(excluded)
    for key, profile in all_profiles.items():
        assert not (
            {row["term"] for row in profile["distinctive_words"]} & excluded
        ), key


def test_one_argument_loader_only_changes_the_requested_era(
    all_profiles, monkeypatch
):
    monkeypatch.setattr(
        era_profiles, "load_all_era_profiles", lambda: all_profiles
    )
    assert era_profiles.load_era_profile("founding") is all_profiles["founding"]
    assert era_profiles.load_era_profile("present") is all_profiles["present"]
