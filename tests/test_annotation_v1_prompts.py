"""Target #4 — few-shot enum regression guard + frozen-taxonomy schema lock.

The judgment schema constrains topic labels to a closed ``enum``, so an invalid
label is structurally impossible rather than merely rejected at ingest. Two
things can silently break that guarantee without any test noticing:

  * a few-shot worked example that shows the model a topic string NOT in the
    frozen 50 (teaching it a label the schema will then reject), and
  * the schema's ``enum`` drifting away from the exact 50 level-2 names in the
    frozen ``taxonomy_v1.json`` (a hand-edited or re-ordered copy).

These tests lock both against the frozen artifact directly.
"""

from __future__ import annotations

import json
import re

from presidential_profiles.corpus import DATA_DIR
from presidential_profiles.prompts import annotation_v1 as av1

_TAXONOMY_PATH = DATA_DIR / "llm_annotations" / "taxonomy_v1.json"


def _frozen_level2_names() -> list[str]:
    """The 50 level-2 topic names, read straight from the frozen paid artifact —
    NOT via av1.TOPIC_NAMES, so a test that anchors on this catches av1 hardcoding
    a divergent list."""
    data = json.loads(_TAXONOMY_PATH.read_text())
    return [t["name"] for t in data["level2"]]


def _fewshot_annotations() -> list[dict]:
    """Every ``ANNOTATION: {json}`` object embedded in the era-anchored few-shots."""
    blobs = re.findall(r"^ANNOTATION: (\{.*\})$", av1._JUDGMENT_FEWSHOTS, flags=re.M)
    return [json.loads(b) for b in blobs]


# ---------------------------------------------------------------------------
# the judgment schema's topic enum == exactly the frozen 50 names
# ---------------------------------------------------------------------------


def test_judgment_schema_topic_enum_is_exactly_the_frozen_fifty_level2_names():
    frozen = _frozen_level2_names()
    assert len(frozen) == 50
    assert len(set(frozen)) == 50  # no collisions -> an unambiguous enum

    enum = av1.JUDGMENT_SCHEMA["properties"]["annotations"]["items"]["properties"][
        "topics"
    ]["items"]["enum"]
    # Exact list equality: same names AND same order as the frozen artifact.
    assert enum == frozen
    assert av1.TOPIC_NAMES == frozen


# ---------------------------------------------------------------------------
# every few-shot topic label is a real frozen level-2 name (the regression that
# was actually observed: ~0.7% of returned strings didn't match a level-2 name)
# ---------------------------------------------------------------------------


def test_every_fewshot_topic_label_is_in_the_frozen_topic_names():
    frozen = set(_frozen_level2_names())
    annotations = _fewshot_annotations()
    assert len(annotations) >= 6  # the rubric ships six era-anchored examples

    used = sorted({t for a in annotations for t in a["topics"]})
    assert used, "few-shots exercise at least one topic label"  # non-empty guard
    offenders = [t for t in used if t not in frozen]
    assert offenders == [], f"few-shot topics not in the frozen taxonomy: {offenders}"


def test_every_fewshot_entity_type_and_stance_is_in_its_closed_enum():
    """The few-shots also teach the entity type/stance vocabularies — an example
    showing an out-of-enum value would train the model toward a label the schema
    rejects."""
    entities = [e for a in _fewshot_annotations() for e in a["entities"]]
    assert entities, "few-shots demonstrate at least one entity"  # non-empty guard

    bad_types = [e["type"] for e in entities if e["type"] not in av1.ENTITY_TYPES]
    bad_stances = [e["stance"] for e in entities if e["stance"] not in av1.ENTITY_STANCES]
    assert bad_types == [], f"few-shot entity types outside the enum: {bad_types}"
    assert bad_stances == [], f"few-shot entity stances outside the enum: {bad_stances}"


def test_every_fewshot_proposal_values_class_is_in_its_closed_enum():
    classes = {a["proposal_values"] for a in _fewshot_annotations()}
    assert classes, "few-shots exercise the proposal/values classes"  # non-empty guard
    offenders = sorted(c for c in classes if c not in av1.PROPOSAL_VALUES_CLASSES)
    assert offenders == [], f"few-shot proposal_values outside the enum: {offenders}"


# ---------------------------------------------------------------------------
# per-request sizing formula (2000 + 130 * n_paragraphs — slope raised from 80
# after the pilot measured ~112 output tok/para; 80 would truncate long speeches)
# ---------------------------------------------------------------------------


def test_judgment_max_tokens_is_2000_plus_130_per_paragraph():
    assert av1.judgment_max_tokens(0) == 2000
    assert av1.judgment_max_tokens(1) == 2130
    assert av1.judgment_max_tokens(3) == 2390
    # Slope must exceed the measured ~112 out/para so a fully-produced long speech
    # is not truncated into incomplete JSON.
    assert av1.judgment_max_tokens(1) - av1.judgment_max_tokens(0) == 130 > 112
    # The largest real speech (Lincoln, Peoria 1854 — 295 paragraphs) must stay
    # well under Sonnet 5's 128K output ceiling.
    assert av1.judgment_max_tokens(295) == 2000 + 130 * 295 < 128_000
