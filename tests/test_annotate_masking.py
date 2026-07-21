"""Target #5 — the masking seam: the judgment pass is blind to the speaker.

Two passes want opposite things. The MASKED judgment request must carry the
speech's decade and the para_idx tags the model echoes back, but NEVER the
president's name or the speech title — so the obscure middle of the corpus is
judged on its words, not its author's reputation. The UNMASKED factual request
DELIBERATELY shows the title + year, because the title is the single best signal
for speech typing.

These drive the REAL specs through ``build_requests`` (not a hand-rolled body)
so the assertion tracks whatever the shipped context builders actually emit.
"""

from __future__ import annotations

import pandas as pd

from presidential_profiles import annotate as A

# A synthetic speech whose author and title are distinctive strings that could
# not appear by accident — so "not in the request body" is a real assertion.
_PRESIDENT = "Zebediah Q. Fauxington"
_TITLE = "First Inaugural Address of Zebediah Q. Fauxington"
_DOC = "/the-presidency/presidential-speeches/synthetic-1853-first-inaugural"


def _synthetic_speech_and_paras():
    speeches = pd.DataFrame(
        {
            "doc_name": [_DOC],
            "title": [_TITLE],
            "president": [_PRESIDENT],
            "year": [1853],
            "decade": [1850],
            "date": [pd.Timestamp("1853-03-04")],
            "transcript": ["We gather in common purpose. Let liberty guide our counsel."],
        }
    )
    # Paragraph text deliberately does NOT contain the president/title, so the
    # only way either could leak into the judgment body is via the builder.
    paras = pd.DataFrame(
        {
            "doc_name": [_DOC, _DOC, _DOC],
            "para_idx": [0, 1, 2],
            "text": [
                "We gather in common purpose to renew the compact of the Republic.",
                "Let liberty be the lamp by which our counsel is guided.",
                "The Union endures because its people will it so.",
            ],
            "word_count": [11, 10, 9],
        }
    )
    return speeches, paras


def _user_body(req) -> str:
    return req["params"]["messages"][0]["content"]


# ---------------------------------------------------------------------------
# judgment pass — MASKED
# ---------------------------------------------------------------------------


def test_judgment_request_carries_decade_and_para_tags_but_not_speaker_or_title():
    speeches, paras = _synthetic_speech_and_paras()
    reqs, _ = A.build_requests(A.FIELD_SPECS["paragraph_annotations"], speeches, paras)
    assert len(reqs) == 1
    body = _user_body(reqs[0])

    # what the model IS allowed to see
    assert "1850s" in body  # the decade is injected
    assert "[para_idx=0]" in body and "[para_idx=2]" in body  # every echoed key

    # what must be withheld (the whole point of masking the judgment pass)
    assert _PRESIDENT not in body
    assert "Fauxington" not in body
    assert _TITLE not in body
    assert "Inaugural" not in body  # not even a title fragment


def test_judgment_max_tokens_scales_2000_plus_130_per_paragraph():
    """max_tokens rides on the per-request ``max_tokens_fn``. The slope is
    130/para (headroom over the pilot-measured ~112 out/para) so a long speech is
    not truncated into incomplete JSON, and a tiny one does not over-reserve."""
    speeches, paras = _synthetic_speech_and_paras()  # 3 paragraphs
    reqs, _ = A.build_requests(A.FIELD_SPECS["paragraph_annotations"], speeches, paras)
    assert reqs[0]["params"]["max_tokens"] == 2000 + 130 * 3
    # largest real speech (Lincoln Peoria, 295 paras) must stay under the 128K cap
    assert A.av1.judgment_max_tokens(295) == 2000 + 130 * 295 < 128_000


def test_judgment_effort_is_medium_factual_low_and_from_accepted_set():
    """The judgment pass runs at effort='medium' (it under-produced at 'low' in
    the pilot); the tiny factual pass stays 'low'. Both must be in the accepted
    Sonnet-5 effort set, and a built judgment request must carry the value."""
    accepted = {"low", "medium", "high"}
    assert A.FIELD_SPECS["paragraph_annotations"].effort == "medium"
    assert A.FIELD_SPECS["speech_annotations"].effort == "low"
    assert A.FIELD_SPECS["paragraph_annotations"].effort in accepted
    assert A.FIELD_SPECS["speech_annotations"].effort in accepted

    speeches, paras = _synthetic_speech_and_paras()
    reqs, _ = A.build_requests(A.FIELD_SPECS["paragraph_annotations"], speeches, paras)
    assert reqs[0]["params"]["output_config"]["effort"] == "medium"


def test_judgment_body_carries_the_exact_count_contract():
    """The count contract (opening + closing restatement) is what fixed the
    under-production. It must state the exact number of paragraphs and appear
    both before and after the paragraph block."""
    speeches, paras = _synthetic_speech_and_paras()  # 3 paragraphs: idx 0,1,2
    reqs, _ = A.build_requests(A.FIELD_SPECS["paragraph_annotations"], speeches, paras)
    body = _user_body(reqs[0])

    assert "Return EXACTLY 3 annotation object(s)" in body      # opening contract
    assert "must contain exactly 3 object(s)" in body            # closing restatement
    # the closing line follows the last paragraph (contract brackets the body)
    assert body.index("END OF PARAGRAPHS") > body.index("[para_idx=2]")


# ---------------------------------------------------------------------------
# factual pass — UNMASKED
# ---------------------------------------------------------------------------


def test_factual_request_deliberately_includes_title_and_year():
    speeches, paras = _synthetic_speech_and_paras()
    reqs, _ = A.build_requests(A.FIELD_SPECS["speech_annotations"], speeches, paras)
    assert len(reqs) == 1
    body = _user_body(reqs[0])

    # the title is the strongest speech-typing signal and is shown on purpose
    assert _TITLE in body
    assert "1853" in body  # the year


def test_both_masking_seam_requests_pass_the_paid_path_validator():
    """A masking leak that also broke request validity would be caught elsewhere;
    prove the real specs still produce requests the pre-flight validator accepts
    (thinking disabled, no sampling params, valid SDK shape)."""
    speeches, paras = _synthetic_speech_and_paras()
    for spec_name in ("paragraph_annotations", "speech_annotations"):
        reqs, _ = A.build_requests(A.FIELD_SPECS[spec_name], speeches, paras)
        for req in reqs:
            A._validate_request(req)  # must not raise
