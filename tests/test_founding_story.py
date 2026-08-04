"""Founding-section data, narrative, interaction, and accessibility contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import attention, corpus, site


DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def founding_inputs() -> dict:
    return {
        "speeches": corpus.load(),
        "paragraphs": pd.read_parquet(DATA / "paragraphs.parquet"),
        "paragraph_annotations": pd.read_parquet(
            DATA / "llm_annotations" / "paragraph_annotations.parquet"
        ),
        "speech_annotations": pd.read_parquet(
            DATA / "llm_annotations" / "speech_annotations.parquet"
        ),
        "speech_stats": pd.read_parquet(DATA / "speech_stats.parquet"),
        "agreement": pd.read_parquet(
            DATA / "llm_annotations" / "agreement_v1.parquet"
        ),
        "paragraph_entities": pd.read_parquet(
            DATA / "llm_annotations" / "paragraph_entities.parquet"
        ),
        "constituency_claims": pd.DataFrame(
            columns=["doc_name", "para_idx", "outcome", "normalized_group"]
        ),
        "lifecycles": pd.read_parquet(
            DATA / "attention" / "topic_lifecycles.parquet"
        ),
        "taxonomy": attention.load_taxonomy(),
    }


@pytest.fixture(scope="module")
def founding_data(founding_inputs) -> dict:
    return site.founding_story_data(**founding_inputs)


def test_governing_problem_unions_are_recomputed_from_exact_paragraph_keys(
    founding_data,
):
    assert founding_data["founding_paragraphs"] == 682
    assert founding_data["founding_speeches"] == 54
    assert founding_data["corpus_paragraphs"] == 36_229
    assert founding_data["corpus_speeches"] == 1_057
    assert founding_data["founding_paragraph_corpus_share"] == pytest.approx(
        1.8824698
    )
    assert founding_data["founding_speech_corpus_share"] == pytest.approx(
        5.1087985
    )
    assert founding_data["founding_words"] == 86_326
    assert founding_data["corpus_words"] == 4_179_266
    assert founding_data["founding_word_corpus_share"] == pytest.approx(
        2.0655780
    )
    language = founding_data["language_profile"]
    assert language["words_per_sentence"] == pytest.approx(38.3906459)
    assert language["corpus_words_per_sentence"] == pytest.approx(23.4925561)
    assert language["shall_per_1000"] == pytest.approx(2.6338079)
    assert language["corpus_shall_per_1000"] == pytest.approx(1.1094407)
    distinctive = language["distinctive_word"]
    assert distinctive["term"] == "militia"
    assert distinctive["founding_count"] == 53
    assert distinctive["other_count"] == 208
    assert distinctive["rate_ratio"] == pytest.approx(12.4282182)
    assert [
        (row["rank"], row["term"], row["founding_count"], row["other_count"])
        for row in language["distinctive_words"]
    ] == [
        (1, "militia", 53, 208),
        (2, "gentlemen", 50, 216),
        (3, "happiness", 51, 273),
    ]
    assert [
        row["rate_ratio"] for row in language["distinctive_words"]
    ] == pytest.approx([12.4282182, 11.2904848, 9.1117934])
    assert [
        (row["era_speech_count"], row["era_president_count"])
        for row in language["distinctive_words"]
    ] == [(21, 3), (17, 3), (26, 3)]
    assert language["distinctive_eligibility"] == {
        "min_corpus_uses": 50,
        "min_era_speeches": 10,
        "min_era_presidents": 2,
    }
    problems = {row["key"]: row for row in founding_data["problems"]}
    assert problems["external"]["count"] == 260
    assert problems["external"]["share"] == pytest.approx(38.1231672)
    assert problems["capacity"]["count"] == 153
    assert problems["capacity"]["share"] == pytest.approx(22.4340176)
    assert problems["native"]["count"] == 125
    assert problems["native"]["share"] == pytest.approx(18.3284457)
    assert founding_data["any_problem_count"] == 479
    assert founding_data["any_problem_share"] == pytest.approx(70.2346041)
    assert founding_data["treaty_neutral_overlap_count"] == 33
    assert founding_data["sovereignty_security_count"] == 363
    assert founding_data["sovereignty_security_share"] == pytest.approx(
        53.2258065
    )
    assert founding_data["sovereignty_security_overlap_count"] == 22
    assert founding_data["sovereignty_security_overlap_share"] == pytest.approx(
        3.2258065
    )

    finance = next(
        row for row in problems["capacity"]["components"]
        if row["topic"] == "Public Debt, Revenue & Treasury Finance"
    )
    assert finance["count"] == 61
    assert finance["share"] == pytest.approx(8.9442815)


def test_four_functioning_country_tests_are_distinct_and_president_specific(
    founding_data,
):
    tests = {row["key"]: row for row in founding_data["founding_tests"]}
    assert {
        key: (row["count"], row["share"])
        for key, row in tests.items()
    } == {
        "union": (142, pytest.approx(20.8211144)),
        "finance": (84, pytest.approx(12.3167155)),
        "native": (125, pytest.approx(18.3284457)),
        "abroad": (180, pytest.approx(26.3929619)),
    }
    assert [
        (row["president"], row["count"], row["share"])
        for row in tests["abroad"]["by_president"]
    ] == [
        ("George Washington", 50, pytest.approx(19.7628458)),
        ("John Adams", 55, pytest.approx(47.4137931)),
        ("Thomas Jefferson", 75, pytest.approx(23.9616613)),
    ]
    topic_sets = [set(row["topics"]) for row in tests.values()]
    assert all(
        not left & right
        for index, left in enumerate(topic_sets)
        for right in topic_sets[index + 1:]
    )
    largest_overlap = max(
        founding_data["founding_test_overlaps"],
        key=lambda row: row["count"],
    )
    assert largest_overlap == {
        "left": "union",
        "right": "abroad",
        "count": 15,
        "share": pytest.approx(2.1994135),
    }


def test_presidential_agendas_channels_adversaries_and_next_era_are_derived(
    founding_data,
):
    agendas = {
        row["president"]: row["topics"]
        for row in founding_data["president_agendas"]
    }
    assert [row["topic"] for row in agendas["George Washington"][:2]] == [
        "Indian Affairs, Removal & Allotment",
        "Constitutional Union & Federalism",
    ]
    assert agendas["John Adams"][0]["topic"] == (
        "Neutral Rights & Maritime Depredations"
    )
    assert set(agendas) == {
        "George Washington",
        "John Adams",
        "Thomas Jefferson",
    }
    racetrack = founding_data["agenda_racetrack"]
    assert len(racetrack) == 8
    assert {
        agenda["topics"][0]["topic"]
        for agenda in founding_data["president_agendas"]
    }.issubset({row["topic"] for row in racetrack})
    maritime = next(
        row
        for row in racetrack
        if row["topic"] == "Neutral Rights & Maritime Depredations"
    )
    assert maritime["share"] == pytest.approx(13.6363636)
    assert [
        (row["president"], row["share"])
        for row in maritime["by_president"]
    ] == [
        ("George Washington", pytest.approx(6.7193676)),
        ("John Adams", pytest.approx(35.3448276)),
        ("Thomas Jefferson", pytest.approx(11.1821086)),
    ]
    channels = {
        row["president"]: row
        for row in founding_data["president_communications"]
    }
    assert channels["Thomas Jefferson"]["written_share"] == pytest.approx(
        75.0
    )
    assert channels["John Adams"]["congress_share"] == pytest.approx(66.6666667)
    assert all(
        sum(route["count"] for route in row["routes"]) == row["speeches"]
        for row in channels.values()
    )
    assert channels["George Washington"]["routes"][0] == {
        "audience": "Congress",
        "medium": "Written message",
        "count": 10,
        "share": pytest.approx(47.6190476),
    }

    edges = {
        (row["president"], row["adversary"]): row
        for row in founding_data["adversary_network"]
    }
    assert edges[("John Adams", "France")]["paragraphs"] == 14
    assert set(edges[("John Adams", "France")]["raw_entities"]) == {
        "Executive Directory",
        "France",
        "French Directory",
        "French Republic",
    }
    assert edges[("Thomas Jefferson", "Aaron Burr")]["paragraphs"] == 11
    assert edges[("Thomas Jefferson", "Great Britain")]["paragraphs"] == 14
    summaries = {
        row["president"]: row
        for row in founding_data["president_adversary_summary"]
    }
    assert summaries["George Washington"] == {
        "president": "George Washington",
        "total_words": 31_474,
        "adversary_words": 4_375,
        "word_share": pytest.approx(13.9003622),
        "named_adversaries": 33,
        "named_paragraphs": 32,
    }
    assert summaries["John Adams"]["word_share"] == pytest.approx(17.5873578)
    assert summaries["John Adams"]["named_adversaries"] == 10
    assert summaries["Thomas Jefferson"]["word_share"] == pytest.approx(
        22.4863708
    )
    assert summaries["Thomas Jefferson"]["named_adversaries"] == 21

    preview = {
        row["label"]: row for row in founding_data["next_era_preview"]
    }
    assert preview["National banking"]["founding_share"] == pytest.approx(
        0.4398827
    )
    assert preview["National banking"]["expansion_share"] == pytest.approx(
        13.0786744
    )
    assert preview["Maritime rights"]["expansion_share"] == pytest.approx(
        3.7945864
    )


def test_native_attention_leaders_and_communication_agreement_are_artifact_derived(
    founding_data,
):
    share_leader = founding_data["native_share_leader"]
    count_leader = founding_data["native_count_leader"]
    assert (share_leader["president"], share_leader["count"]) == (
        "George Washington", 57,
    )
    assert share_leader["share"] == pytest.approx(22.5296443)
    assert (count_leader["president"], count_leader["count"]) == (
        "Thomas Jefferson", 65,
    )
    assert founding_data["agreement"] == {
        "audience": {"rate": pytest.approx(96.6165414), "n": 266},
        "medium": {"rate": pytest.approx(86.4661654), "n": 266},
        "n": 266,
    }

    war = founding_data["communications"][5]
    public = next(row for row in war["audience"] if row["label"] == "General public")
    broadcast = next(
        row for row in war["medium"] if row["label"] == "Radio/TV broadcast"
    )
    assert war["total"] == 69
    assert (public["count"], public["share"]) == (54, pytest.approx(78.2608696))
    assert (broadcast["count"], broadcast["share"]) == (
        38, pytest.approx(55.0724638),
    )


def test_era_profile_uses_named_entities_and_constituency_fallback(founding_data):
    profile = founding_data["era_profile"]
    assert [row["name"] for row in profile["presidents"]] == [
        "George Washington",
        "John Adams",
        "Thomas Jefferson",
    ]
    assert [row["name"] for row in profile["adversaries"][:4]] == [
        "France",
        "Great Britain",
        "Aaron Burr",
        "Spain",
    ]
    assert [row["type"] for row in profile["adversaries"]] == [
        "nation",
        "nation",
        "person",
        "nation",
        "nation",
    ]
    assert [
        (row["type"], row["corpus_mentions"], row["present"])
        for row in profile["adversary_types"]
    ] == [
        ("nation", 9172, True),
        ("person", 9349, True),
        ("group", 3852, False),
        ("institution", 4160, False),
        ("other", 681, False),
    ]
    assert profile["style"]["medium"]["label"] == "Written message"
    assert profile["style"]["medium"]["share"] == pytest.approx(83.3333333)
    assert profile["style"]["audience"]["label"] == "Congress"
    assert profile["style"]["audience"]["share"] == pytest.approx(53.7037037)
    assert [row["label"] for row in profile["style"]["audience_options"]] == [
        "Congress",
        "General public",
        "Specific organizations or groups",
        "Other audiences",
    ]
    assert [row["label"] for row in profile["style"]["medium_options"]] == [
        "Written message",
        "Spoken address",
        "Radio/TV broadcast",
        "Press conference/debate or other performed form",
    ]
    assert profile["constituency_status"] == "fallback"
    assert [row["name"] for row in profile["constituents"]] == list(
        site.FOUNDING_CONSTITUENCY_FALLBACK
    )
    assert "Native nations and peoples" not in site.FOUNDING_CONSTITUENCY_FALLBACK


def test_promoted_constituency_rows_replace_the_fallback(founding_inputs):
    speech = founding_inputs["speeches"][
        founding_inputs["speeches"]["year"].between(1789, 1808)
    ].iloc[0]
    paragraph_indices = (
        founding_inputs["paragraphs"]
        .loc[
            founding_inputs["paragraphs"]["doc_name"].eq(speech["doc_name"]),
            "para_idx",
        ]
        .head(3)
        .tolist()
    )
    assert len(paragraph_indices) == 3
    claims = pd.DataFrame([
        {
            "doc_name": speech["doc_name"],
            "para_idx": paragraph_indices[0],
            "outcome": "claim",
            "normalized_group": "Merchants",
        },
        {
            "doc_name": speech["doc_name"],
            "para_idx": paragraph_indices[1],
            "outcome": "claim",
            "normalized_group": "Merchants",
        },
        {
            "doc_name": speech["doc_name"],
            "para_idx": paragraph_indices[2],
            "outcome": "claim",
            "normalized_group": "The national public",
        },
    ])
    inputs = dict(founding_inputs)
    inputs["constituency_claims"] = claims
    data = site.founding_story_data(**inputs)
    assert data["era_profile"]["constituency_status"] == "artifact"
    assert data["era_profile"]["constituents"] == [
        {"name": "Merchants", "paragraphs": 2},
        {"name": "The national public", "paragraphs": 1},
    ]


def test_persistence_score_order_uses_equal_later_era_weights_and_tie_break(
    founding_data,
):
    expected = [
        "Public finance",
        "Executive Power, Vetoes & the Courts",
        "Military Preparedness, Armed Forces & Veterans",
        "Crime, Insurrection & Federal Law Enforcement",
        "Early Naval Wars: Barbary & the War of 1812",
        "Indian Affairs, Removal & Allotment",
    ]
    threads = founding_data["threads"]
    assert [row["label"] for row in threads] == expected
    for row in threads:
        normalized = np.asarray(row["normalized"])
        assert normalized[0] == pytest.approx(1)
        assert row["persistence_score"] == pytest.approx(normalized[1:].mean())
        assert row["normalized_std"] == pytest.approx(normalized.std(ddof=0))

    military = next(row for row in threads if row["label"].startswith("Military"))
    crime = next(row for row in threads if row["label"].startswith("Crime"))
    assert military["persistence_score"] > crime["persistence_score"]
    assert military["normalized_std"] < crime["normalized_std"]
    assert threads.index(military) < threads.index(crime)

    finance = next(row for row in threads if row["label"] == "Public finance")
    assert len(founding_data["finance_sublanes"]) == 2
    assert all(
        sublane["color"] == finance["color"]
        for sublane in founding_data["finance_sublanes"]
    )
    for position in range(9):
        assert finance["counts"][position] >= max(
            sublane["counts"][position]
            for sublane in founding_data["finance_sublanes"]
        )


def test_attention_departures_keep_native_and_naval_claims_separate(
    founding_data,
):
    departures = {
        row["key"]: row for row in founding_data["attention_departures"]
    }
    assert set(departures) == {"native", "naval"}
    assert departures["native"]["topic"] == (
        "Indian Affairs, Removal & Allotment"
    )
    assert departures["naval"]["topic"] == (
        "Early Naval Wars: Barbary & the War of 1812"
    )
    assert len(departures["native"]["shares"]) == 9
    assert len(departures["naval"]["shares"]) == 9
    assert departures["native"]["shares"][0] == pytest.approx(18.3284457)
    assert departures["native"]["shares"][0] > departures["native"]["shares"][-1]
    assert departures["naval"]["shares"][0] > departures["naval"]["shares"][-1]


def test_rendered_founding_section_has_four_direct_peer_views(
    founding_data,
):
    body = site.founding_story_html(founding_data)
    normalized = " ".join(body.split())
    assert "Getting the Republic Up and Running" in body
    assert "The startup agenda combined unusually intense sovereignty" in body
    assert "20.8%" in body and "12.3%" in body and "18.3%" in body
    assert "26.4%" in body
    assert body.count("data-founding-tab=") == 4
    assert body.count("founding-story-card founding-panel") == 4
    assert body.count("data-founding-screen-group") == 0
    assert body.count("data-founding-screen-tab=") == 0
    assert body.count("data-founding-screen hidden") == 0
    assert 'aria-label="Establishing the republic era views"' in body
    assert 'id="founding-workspace"' in body
    assert 'id="founding-profile"' in body
    assert 'id="founding-screen-defined"' in body
    assert 'id="founding-screen-adversaries"' in body
    assert 'id="founding-screen-echoes"' in body
    assert 'id="founding-visualize"' not in body
    assert 'id="founding-contextualize"' not in body
    assert 'class="founding-screen-tabs founding-visualize-tabs"' not in body
    assert 'class="founding-screen-tabs founding-contextualize-tabs"' not in body
    route = body[
        body.index("founding-movement-route"):
        body.index("</nav>", body.index("founding-movement-route"))
    ]
    assert ">01<" not in route and ">02<" not in route
    assert "Era profile" in route
    assert "Era defined" in route
    assert "Adversaries" in route
    assert "Era echoes" in route
    profile_html = site._founding_era_profile_html(founding_data)
    assert profile_html.count("Establishing the Republic") == 1
    assert "<strong>Establishing the Republic</strong>" in profile_html
    assert "Sovereignty, federal capacity, and expansion" in normalized
    assert '<div class="era-template-title">' not in profile_html
    assert profile_html.count('class="era-card-heading"') == 4
    for heading in (
        "Claimed constituents",
        "Named adversaries",
        "Corpus footprint",
        "Major Topics",
    ):
        assert (
            f'<header class="era-card-heading"><span>{heading}</span></header>'
            in profile_html
        )
    assert "era-template-style" not in profile_html
    assert "Governing style" not in profile_html
    assert "era-template-presidents" not in profile_html
    assert profile_html.count('class="era-president-strip"') == 1
    assert profile_html.count('src="portraits/') == 3
    assert "portraits/george-washington.png" in profile_html
    assert profile_html.count('class="era-adversary-bubble"') == 5
    assert 'class="era-adversary-pack"' in profile_html
    assert "era-adversary-jar" not in profile_html
    assert "France · Nation or state · 20 adversarial paragraphs" in profile_html
    for entity_type in (
        "Nation or state",
        "Person",
        "Group",
        "Institution",
        "Other",
    ):
        assert entity_type in profile_html
    assert profile_html.count('class="is-present"') == 2
    assert profile_html.count('class="is-absent"') == 3
    assert "Hollow = not among these five" not in profile_html
    assert "Bubble size = adversarial paragraph count" not in profile_html
    assert 'class="era-style-scroll"' not in profile_html
    assert "Specific organizations or groups" not in profile_html
    assert "Press conference/debate or other performed form" not in profile_html
    assert "Specific organizations or groups" not in body
    assert "Press conference/debate or other performed form" not in body
    assert "Illustrative only · promoted claims pending" in body
    assert "5.1%" in body and "1.9%" in body and "2.1%" in body
    assert "54 of 1,057" in normalized
    assert "682 of 36,229" in normalized
    assert "86,326 of 4,179,266" in normalized
    assert "words / sentence" not in profile_html
    assert "“shall” / 1,000 words" not in profile_html
    assert "MILITIA" in profile_html
    assert "INFORMATION" in profile_html
    assert "GENTLEMEN" in profile_html
    assert "Rank #1</span>" in profile_html
    assert "Rank #2</span>" in profile_html
    assert "Rank #3</span>" in profile_html
    assert "53 uses<br>12.4× other eras" in profile_html
    assert "104 uses<br>4.3× other eras" in profile_html
    assert "52 uses<br>10.1× other eras" in profile_html
    assert "Evidence:" not in profile_html
    assert "How the distinctive-word ranking works" in profile_html
    assert "the same families as Explore" in normalized
    assert "slavery, slave, and slaves count together" in normalized
    assert "1 − exp(−era family uses ÷ 100)" in normalized
    assert "concentration beyond 25×" in normalized
    assert "Eligibility: ≥50" in normalized
    assert "≥10" in profile_html and "≥2" in profile_html
    assert "21 era speeches across 3 presidents" not in profile_html
    assert "26 era speeches across 3 presidents" not in profile_html
    assert "--footprint-width:" not in profile_html
    assert "Share bars use" not in profile_html
    assert "founding-step-footer" not in profile_html
    assert body.index("founding-movement-route") < body.index("founding-profile")
    assert 'class="era-template-topics"' in body
    assert "<span>Major Topics</span>" in body
    assert "Executive power &amp; courts" in body
    assert "Military preparedness" in body
    assert "Ranked by Founding-era paragraph share · labels overlap" in body
    assert profile_html.count("--topic-width:") == 6
    assert profile_html.index("Indian &amp; Native affairs") < profile_html.index(
        "Military preparedness"
    )
    assert profile_html.index("Military preparedness") < profile_html.index(
        "Public finance"
    )
    assert profile_html.index("Public finance") < profile_html.index(
        "Barbary &amp; War of 1812"
    )
    assert "Great Britain" in body and "Aaron Burr" in body
    assert "Specific organizations or groups" not in body
    assert "Other audiences" not in body
    assert "Radio/TV broadcast" not in body
    assert "Press conference/debate or other performed form" not in body
    assert "data-founding-next=" not in body
    assert "Next, turn the patterns into an era-level argument." not in body
    assert 'href="#expansion_conflict"' not in body
    assert "Presidential agendas" not in body
    assert "Governing channels" not in body
    assert "Adversary network" in body
    assert body.count('<header class="founding-visual-question">') == 1
    assert (
        "What each president emphasized, how they governed, and whom they "
        "opposed."
    ) not in body
    assert "Line width = distinct adversarial paragraphs" in normalized
    assert "President node score = √(adversary-word share × distinct named adversaries)" in normalized
    assert "13.9% · 33 names" in body
    assert "22.5% · 21 names" in body
    assert "Who controls federal power as the republic expands?" not in body
    tests_html = site._founding_four_tests_html(founding_data)
    assert "Era Defined" in tests_html
    assert "later-era average" not in tests_html
    assert 'class="era-defined-combined-chart"' in tests_html
    assert tests_html.count('class="era-defined-point-value"') == 36
    assert ">Native power 0.0%</text>" not in tests_html
    assert "Text alternative" not in tests_html
    assert "<summary>How the measure works</summary>" in tests_html
    assert "founding-next-era" not in tests_html
    assert "Topic Life" not in body and "Era Echoes" in body
    assert 'data-echo-direction="incoming"' in body
    assert 'data-echo-direction="outgoing"' in body
    assert "Invoked by" in body and "Invoking" in body
    assert "Who invoked the Founding—and around what?" in body
    assert "Native nations and peoples" not in body
    assert "Hold the union together" in tests_html
    assert "Finance the government" in tests_html
    assert "Navigate Native nations and continental power" in tests_html
    assert "Exercise independence abroad" in tests_html

    assert "Who the presidency addressed, and how" not in body
    assert "data-founding-era-button" not in body
    summary = site._standing_html(founding_data)
    assert "Who presidents addressed—and how" in summary
    assert summary.count('class="communication-era-card"') == 9
    assert "🏛️ Congress (53.7%)" in summary
    assert "📻 assigned broadcast form" in summary
    assert "78.3%" in summary and "55.1%" in summary
    assert "96.6% audience" in summary and "86.5% medium" in summary

    assert "Bubble area = paragraph share" not in body
    threads_html = site._founding_threads_html(founding_data)
    assert 'data-founding-label-toggle aria-expanded="false"' in threads_html
    assert "Show topic labels" in threads_html
    assert 'aria-label="Indian Affairs, Removal &amp; Allotment"' in threads_html
    assert "🪶" in threads_html
    assert 'class="founding-finance-breakdown"' not in threads_html
    assert 'class="founding-finance-summary"' not in threads_html
    assert 'class="founding-finance-arrow"' not in threads_html
    assert 'class="founding-finance-grid"' not in threads_html
    assert "founding-native-decline" not in threads_html
    attention = site._founding_attention_departures_html(founding_data)
    assert "What drops out of presidential attention" in attention
    assert "Native affairs" in attention
    assert "Barbary and War of 1812 naval affairs" in attention
    assert attention.count("<article>") == 2
    assert "Native presence, power, and policy do not" in attention
    assert "foreign and military speech continue" in attention
    assert "founding-screen-native" not in body
    agendas_html = site._founding_presidential_agendas_html(founding_data)
    assert (
        "What defined each president’s agenda?"
        in agendas_html
    )
    assert '<header class="founding-visual-question">' in agendas_html
    assert "<span>Presidential agendas</span>" in agendas_html
    assert "data-agenda-cards" in agendas_html
    assert agendas_html.count('class="agenda-card"') == 3
    assert agendas_html.count('class="agenda-card-priority"') == 15
    assert agendas_html.count('class="agenda-card-description"') == 15
    assert "agenda-card-grid-scroll" not in agendas_html
    assert (
        '<details class="agenda-card-description"><summary>'
        in agendas_html
    )
    assert (
        'role="group" aria-label="George Washington&#x27;s number 1'
        in agendas_html
    )
    assert "Within each card, the leading domain fills the bar" in agendas_html
    assert "George Washington&#x27;s number 1 policy domain" in agendas_html
    assert "Constitutional Order &amp; Governance" in agendas_html
    assert "The constitutional structure of the government" in agendas_html
    assert "Remaining policy" not in agendas_html
    assert "Non-policy" not in agendas_html
    assert "Text alternative" not in agendas_html
    assert '<details class="agenda-method-note">' in agendas_html
    assert '<details class="agenda-method-note" open' not in agendas_html
    assert "<summary>What the percentages mean</summary>" in agendas_html
    assert "data-agenda-sort-topic" not in agendas_html
    assert "data-agenda-sort-president" not in agendas_html
    assert "Era share" not in agendas_html
    assert "founding-agenda-card" not in agendas_html
    assert "data-agenda-matrix" not in agendas_html
    page = site.build_html(
        {},
        {
            "speeches": 1057,
            "words": 4_179_266,
            "presidents": 45,
            "start": 1789,
            "end": 2026,
        },
        {key: "<p>Evidence body.</p>" for key, *_ in site.SECTIONS},
        inline=False,
    )
    assert "grid-template-columns:repeat(4,minmax(0,1fr));gap:0" in page
    assert "[data-era-panel] {" in page
    assert 'data-era-panel="visualize"' not in body
    assert 'data-era-panel="contextualize"' not in body
    assert 'document.querySelectorAll("[data-routing-map]")' not in page
    assert ".agenda-card-grid {" in page
    assert ".agenda-card-grid-scroll {" in page
    assert "grid-auto-columns:calc((100% - 20px)/3)" in page
    assert ".agenda-card-priority {" in page
    assert (
        ".agenda-card-description:is(:hover,:focus-within,[open]) > p"
        in page
    )
    assert "text-overflow:ellipsis;white-space:nowrap" in page
    assert "margin-top:8px;padding-top:0" in page
    assert ".agenda-matrix {" not in page
    assert 'document.querySelectorAll("[data-governance-filter]")' not in page
    assert "founding-screen-topic-life" not in page
    assert "...defaultTopics.filter(defaultTopic => defaultTopic !== topic)" not in page
    assert ".inspector-row:has(.inspector-tile[open])" in page
    assert ".inspector-tile:not([open]) summary" in page
    assert "content:attr(data-tooltip)" in page
    assert 'document.querySelectorAll("[data-echo-gravity]")' in page
    assert 'document.querySelectorAll("[data-echo-direction-switch]")' in page
    assert "function activateEchoDirection(direction)" in page
    assert "function activateEchoNode(active)" in page
    assert ".era-echo-gravity-link.is-active" in page
    assert "[data-topic-president-link]" in page
    assert "presidentKeys.has(node.dataset.nodeKey)" in page
    assert "topicKeys.has(node.dataset.nodeKey)" in page
    assert "link.dataset.presidentKey === nodeKey" in page
    assert (
        ".era-echo-gravity-link.is-topic {\n"
        "    stroke:var(--echo-color);stroke-dasharray:2 3;"
    ) in page
    assert ".era-echo-topic-president-link.is-active" in page
    assert "Treaties, Diplomacy &amp; International Arbitration" not in threads_html
    assert "Neutral Rights &amp; Maritime Depredations" not in threads_html
    assert "Early Naval Wars: Barbary &amp; the War of 1812" in threads_html
    assert "Every row uses the same 0–20%" not in body
    assert 'class="topic-life-row is-' not in body
    assert "FOUNDING-CENTERED" not in body
    assert "DURABLE OR RECURRENT" not in body
    assert "Indian &amp; Native affairs" in body
    assert "Taxes, Budget Deficits &amp; Federal Spending" not in threads_html
    assert "Named presidential invocations only" in body
    assert "broader references such as “the Founders,”" in body


def test_rendered_story_replaces_the_written_republic_framing(founding_data):
    bodies = {key: "<p>Evidence body.</p>" for key, *_ in site.SECTIONS}
    bodies["written_republic"] = site.founding_story_html(founding_data)
    page = site.build_html(
        {},
        {
            "speeches": 1057,
            "words": 4_200_000,
            "presidents": 45,
            "start": 1789,
            "end": 2026,
        },
        bodies,
        inline=False,
    )
    assert "1789–1808 · Establishing the republic" in page
    assert 'data-story-era="Establishing the republic"' in page
    assert page.count("Establishing the Republic") == 1
    assert "<strong>Establishing the Republic</strong>" in page
    assert "The written republic" not in page
    assert ".founding-thread-label" in page
    assert "overflow-x:auto" in page
    assert 'id="story-progress-era"' in page
    assert 'id="story-progress-title"' in page
    assert "A chronological story of how the United States grew" not in page
    assert "data-founding-tab" in page
    assert "activateEraWorkspacePanel" in page
    assert "data-founding-screen-tab" in page
    assert "America Through Its Presidents" in page
    assert (
        '<p class="hero-subheadline">1789–2026 · '
        "A country told through presidential speech</p>"
    ) in page
    assert "Browse the 45 president profiles" not in page
    assert 'href="explorer.html">Look up any word or phrase' not in page
    assert "data-story-index" not in page


def test_mismatched_paragraph_key_sets_fail_before_aggregation(founding_inputs):
    broken = dict(founding_inputs)
    broken["paragraph_annotations"] = founding_inputs[
        "paragraph_annotations"
    ].iloc[:-1].copy()
    with pytest.raises(ValueError, match="key sets differ"):
        site.founding_story_data(**broken)

    broken = dict(founding_inputs)
    broken["speech_stats"] = founding_inputs["speech_stats"].iloc[:-1].copy()
    with pytest.raises(
        ValueError,
        match="founding story speeches x speech language statistics: "
        "key sets differ",
    ):
        site.founding_story_data(**broken)
