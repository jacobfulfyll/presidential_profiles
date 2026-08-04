import json

import pandas as pd
import pytest

from presidential_profiles import invocations


def candidate_frame():
    return pd.DataFrame([{
        "candidate_id": "abc", "doc_name": "doc", "para_idx": 0,
        "char_start": 0, "char_end": 7, "target": "John Adams",
        "speaker": "Thomas Jefferson", "speech_date": "1801-01-01",
        "raw_mention": "Adams", "context": "President Adams set a wise precedent.",
        "quotation_status": "narration", "source_role": "presidential_speech",
        "target_status": "former_president", "excluded_reason": "",
    }])


def test_registry_covers_every_corpus_president():
    assert set(invocations.alias_registry()) == set(invocations.corpus.PARTY)


def test_reporter_and_audience_are_excluded():
    assert invocations._source_role("Reporter: President Adams?")[1] == "reporter_speech"
    assert invocations._source_role("Audience: Adams!")[1] == "audience_speech"


def test_classifier_uses_fixed_enums():
    result = invocations.classify_candidate(candidate_frame().iloc[0])
    assert result["function"] in invocations.FUNCTIONS
    assert result["stance"] in invocations.STANCES
    assert result["evidence_span"] in candidate_frame().iloc[0]["context"]


def test_bare_future_surnames_and_washington_places_are_excluded():
    assert invocations._mention_exclusion(
        "Woodrow Wilson", "Wilson",
        "Associate Justice James Wilson certified the result.",
        "not_yet_president",
    ) == "anachronistic_bare_surname"
    assert invocations._mention_exclusion(
        "George Washington", "Washington",
        "The governors met here in Washington to discuss the bill.",
        "former_president",
    ) == "place_or_institution"
    assert invocations._mention_exclusion(
        "George Washington", "Washington",
        "Washington warned against permanent alliances.",
        "former_president",
    ) == ""


def test_grant_verb_is_not_a_presidential_invocation():
    assert invocations._mention_exclusion(
        "Ulysses S. Grant", "grant",
        "We grant a full and entire pardon.",
        "former_president",
    ) == "lexical_surname"


@pytest.mark.parametrize(("target", "raw", "context", "reason"), [
    ("Lyndon B. Johnson", "Johnson", "Prime Minister Boris Johnson joined us.", "different_named_person"),
    ("Andrew Jackson", "Jackson", "Congresswoman Sheila Jackson Lee is here.", "different_named_person"),
    ("John F. Kennedy", "Kennedy", "We landed at the Kennedy Space Center.", "place_or_institution"),
    ("Gerald Ford", "Ford", "Sergeant Kenneth Ford was killed.", "different_named_person"),
    ("Barack Obama", "Obama", "Michelle Obama spoke today.", "different_named_person"),
])
def test_ambiguous_surnames_exclude_other_people_and_institutions(
    target, raw, context, reason
):
    assert invocations._mention_exclusion(
        target, raw, context, "former_president"
    ) == reason


def test_bare_ambiguous_surname_keeps_clear_presidential_context():
    assert invocations._mention_exclusion(
        "Abraham Lincoln", "Lincoln",
        "Lincoln issued the Emancipation Proclamation.",
        "former_president",
    ) == ""


def test_batch_import_rejects_missing_evidence_and_drift(tmp_path):
    root = tmp_path
    candidates = candidate_frame()
    candidates.to_parquet(root / "candidates.parquet", index=False)
    drift = [{"_meta": {"corpus_fingerprint": "wrong", "expected_ids": ["abc"]}},
             {"candidate_id": "abc", "function": "historical comparison",
              "stance": "neutral", "evidence_span": "Adams", "rationale": "x"}]
    with pytest.raises(ValueError, match="fingerprint"):
        invocations.import_batch(drift, root)
    bad = [{"candidate_id": "abc", "function": "historical comparison",
            "stance": "neutral", "evidence_span": "not present", "rationale": "x"}]
    with pytest.raises(ValueError, match="evidence"):
        invocations.import_batch(bad, root)
