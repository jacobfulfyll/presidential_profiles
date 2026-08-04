"""Pure agreement metrics (agreement.py) — tested on tiny synthetic sequences /
frames, no corpus, no I/O.

These are the falsification core: Cohen's kappa for the combativeness flags,
Jaccard for the multi-label topics, exact-match for proposal-vs-values, and
name-matched entity stance agreement. The edge cases below are exactly the ones
the verification strategy calls out — kappa=1.0 on identical inputs, <=0 on
inverted ones, the degenerate single-class NaN (rather than a crash/warning), and
the both-empty=1.0 Jaccard convention.
"""

from __future__ import annotations

import math
import warnings

import pandas as pd
import pytest

from presidential_profiles import agreement as G


# ---------------------------------------------------------------------------
# compute_kappa
# ---------------------------------------------------------------------------


def test_kappa_is_one_on_identical_label_sequences():
    value, n = G.compute_kappa([1, 0, 1, 0, 1], [1, 0, 1, 0, 1])
    assert value == pytest.approx(1.0)
    assert n == 5


def test_kappa_is_at_most_zero_on_perfectly_inverted_sequences():
    value, n = G.compute_kappa([1, 0, 1, 0], [0, 1, 0, 1])
    assert value <= 0
    assert value == pytest.approx(-1.0)
    assert n == 4


def test_kappa_is_nan_with_n_on_a_degenerate_single_class():
    """Every observation carries the same label across BOTH raters — kappa is
    0/0 undefined. It must return NaN WITH the n, not crash and not let sklearn
    emit a warning-laden value."""
    value, n = G.compute_kappa([True, True, True], [True, True, True])
    assert math.isnan(value)
    assert n == 3  # n is preserved so the report can still show sample size


def test_kappa_on_empty_input_is_nan_with_zero_n():
    value, n = G.compute_kappa([], [])
    assert math.isnan(value)
    assert n == 0


def test_kappa_below_threshold_constant_is_0_point_4():
    """The low-confidence flag threshold is a published contract (report flags
    kappa < 0.4, never gates). Pin the literal so a silent retune trips this."""
    assert G.KAPPA_LOW_THRESHOLD == 0.4


def test_kappa_is_chance_corrected_not_raw_agreement():
    """The falsification premise: kappa CORRECTS FOR CHANCE (per the design gloss —
    'two models both saying no-flag 95% of the time is not impressive'). Witness a
    class-imbalanced case where the two raters agree on 80% of items yet Cohen's
    kappa is NEGATIVE (worse than chance): 9/10 zeros each, disagreeing on the two
    minority positions. A metric that returned raw agreement (0.8) or the linear
    2*agreement-1 (0.6) — both of which also give 1.0 on identical and -1.0 on
    inverted, so every OTHER kappa test would still pass — is killed here."""
    a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    b = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    raw_agreement = sum(x == y for x, y in zip(a, b)) / len(a)
    assert raw_agreement == pytest.approx(0.8)  # high raw agreement...
    value, n = G.compute_kappa(a, b)
    assert n == 10
    assert value < 0.0  # ...but chance-corrected kappa is BELOW zero
    assert value == pytest.approx(-1 / 9, abs=1e-3)  # (0.8-0.82)/(1-0.82)


def test_kappa_degenerate_class_emits_no_sklearn_warning():
    """The degenerate single-class guard's REAL job is not the NaN — sklearn also
    returns NaN here — it is suppressing sklearn's UndefinedMetricWarning /
    UserWarning (documented: 'not let sklearn emit a warning-laden value'). Assert
    NO warning escapes, so deleting the guard (which would let sklearn warn on
    every all-same field x era cell) is actually caught, not silently tolerated."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an exception
        value, n = G.compute_kappa([True, True, True], [True, True, True])
    assert math.isnan(value)
    assert n == 3


# ---------------------------------------------------------------------------
# jaccard / compute_jaccard
# ---------------------------------------------------------------------------


def test_jaccard_both_empty_is_one_by_convention():
    """Two models that both assign NO topic to a paragraph agree perfectly."""
    assert G.jaccard([], []) == 1.0


def test_jaccard_disjoint_sets_is_zero():
    assert G.jaccard(["a", "b"], ["c", "d"]) == 0.0


def test_jaccard_partial_overlap_is_intersection_over_union():
    # {a,b} vs {b,c}: intersection {b}=1, union {a,b,c}=3
    assert G.jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)


def test_jaccard_ignores_duplicate_labels_within_a_side():
    # set semantics: ["a","a","b"] collapses to {a,b}
    assert G.jaccard(["a", "a", "b"], ["a", "b"]) == 1.0


def test_compute_jaccard_averages_per_item_and_reports_n():
    # item 1: {a} vs {a} = 1.0 ; item 2: {a,b} vs {b,c} = 1/3 ; mean = 2/3
    value, n = G.compute_jaccard([["a"], ["a", "b"]], [["a"], ["b", "c"]])
    assert value == pytest.approx(2 / 3)
    assert n == 2


def test_compute_jaccard_on_empty_sequence_is_nan_with_zero_n():
    value, n = G.compute_jaccard([], [])
    assert math.isnan(value)
    assert n == 0


def test_jaccard_null_topics_coerce_to_empty_set_m3_fix():
    """M3 (judge minor, fixed): a null topics cell (None, or float NaN as a
    parquet null loads) is coerced to the empty set instead of raising — one
    bad row in an ingested run must never crash the whole agreement build.
    Null-vs-labels scores 0.0 (disjoint); null-vs-null and null-vs-[] score
    1.0 under the both-empty convention."""
    assert G.jaccard(None, ["a"]) == 0.0
    assert G.jaccard(["a"], None) == 0.0
    assert G.jaccard(None, None) == 1.0
    assert G.jaccard(None, []) == 1.0
    assert G.jaccard(float("nan"), ["a"]) == 0.0
    assert G.jaccard(float("nan"), float("nan")) == 1.0


# ---------------------------------------------------------------------------
# compute_exact_match
# ---------------------------------------------------------------------------


def test_exact_match_rate_counts_equal_pairs():
    value, n = G.compute_exact_match(["x", "y", "z"], ["x", "q", "z"])
    assert value == pytest.approx(2 / 3)
    assert n == 3


def test_exact_match_all_agree_is_one():
    value, n = G.compute_exact_match(["a", "b"], ["a", "b"])
    assert value == 1.0
    assert n == 2


def test_exact_match_on_empty_is_nan_with_zero_n():
    value, n = G.compute_exact_match([], [])
    assert math.isnan(value)
    assert n == 0


# ---------------------------------------------------------------------------
# normalize_entity_name
# ---------------------------------------------------------------------------


def test_normalize_lowercases_and_collapses_whitespace():
    assert G.normalize_entity_name("  The   Governor ") == "the governor"


def test_normalize_is_conservative_and_does_not_strip_articles():
    """'Governor' must NOT collapse into 'the Governor' — a granularity/wording
    difference should surface as an UNMATCHED entity, not a silent match."""
    assert G.normalize_entity_name("Governor") != G.normalize_entity_name("the Governor")


# ---------------------------------------------------------------------------
# compute_entity_metrics
# ---------------------------------------------------------------------------


def _ent(rows: list[dict]) -> pd.DataFrame:
    cols = ["doc_name", "para_idx", "entity", "type", "stance"]
    return pd.DataFrame(rows, columns=cols)


def test_entity_metrics_separates_name_match_from_stance_and_unmatched_rates():
    """Two models name one entity in common (case-insensitively) with OPPOSITE
    stances, and each names one the other does not. Name-match, stance agreement,
    and each side's unmatched rate must be reported as distinct quantities."""
    primary = _ent([
        {"doc_name": "d", "para_idx": 0, "entity": "The Congress", "type": "org", "stance": "neg"},
        {"doc_name": "d", "para_idx": 0, "entity": "Britain", "type": "nation", "stance": "pos"},
    ])
    opus = _ent([
        {"doc_name": "d", "para_idx": 0, "entity": "the  congress", "type": "org", "stance": "pos"},
        {"doc_name": "d", "para_idx": 0, "entity": "France", "type": "nation", "stance": "neg"},
    ])
    m = G.compute_entity_metrics(primary, opus)

    # union = {congress, britain, france} = 3, matched = {congress} = 1
    assert m["n_union"] == 3
    assert m["n_matched"] == 1
    assert m["name_match_rate"] == pytest.approx(1 / 3)
    # the single matched entity has opposite stances -> 0 agreement
    assert m["stance_agreement"] == 0.0
    # each side named one entity the other did not, over its own count of 2
    assert m["unmatched_primary_rate"] == pytest.approx(0.5)
    assert m["unmatched_opus_rate"] == pytest.approx(0.5)
    assert m["n_primary"] == 2 and m["n_opus"] == 2


def test_entity_metrics_matched_same_stance_is_full_agreement():
    primary = _ent([{"doc_name": "d", "para_idx": 0, "entity": "Congress", "type": "org", "stance": "neg"}])
    opus = _ent([{"doc_name": "d", "para_idx": 0, "entity": "congress", "type": "org", "stance": "neg"}])
    m = G.compute_entity_metrics(primary, opus)
    assert m["name_match_rate"] == 1.0
    assert m["stance_agreement"] == 1.0
    assert m["n_matched"] == 1


def test_entity_metrics_both_sides_empty_is_full_name_match_and_nan_stance():
    """Both models naming no entity = perfect name agreement (1.0), but there is
    no matched entity to agree on a stance about -> NaN, not a crash."""
    empty = _ent([])
    m = G.compute_entity_metrics(empty, empty)
    assert m["name_match_rate"] == 1.0
    assert math.isnan(m["stance_agreement"])
    assert math.isnan(m["unmatched_primary_rate"])
    assert math.isnan(m["unmatched_opus_rate"])
    assert m["n_union"] == 0 and m["n_matched"] == 0


def test_entity_same_name_in_two_paragraphs_keyed_separately():
    """The key is (doc_name, para_idx, name) — the same entity named in two
    different paragraphs is two distinct keys, so a name match must be
    para-local, not global."""
    primary = _ent([{"doc_name": "d", "para_idx": 0, "entity": "Congress", "type": "org", "stance": "neg"}])
    opus = _ent([{"doc_name": "d", "para_idx": 1, "entity": "Congress", "type": "org", "stance": "neg"}])
    m = G.compute_entity_metrics(primary, opus)
    assert m["n_union"] == 2
    assert m["n_matched"] == 0  # different paragraphs -> not matched


def test_entity_metrics_unmatched_primary_and_opus_rates_are_not_swapped():
    """Gap: the earlier asymmetry test gives BOTH sides 0.5, so a bug swapping
    unmatched_primary_rate <-> unmatched_opus_rate (or n_primary <-> n_opus) would
    pass unseen. Here the sides are deliberately UNEQUAL: primary names 1 entity,
    opus names 3 (the granularity-split risk the design calls out). The primary
    fully matches (0 unmatched) while opus has 2 of its 3 unmatched — distinct
    values that pin each rate to its OWN side."""
    primary = _ent([
        {"doc_name": "d", "para_idx": 0, "entity": "Congress", "type": "org", "stance": "neg"},
    ])
    opus = _ent([
        {"doc_name": "d", "para_idx": 0, "entity": "congress", "type": "org", "stance": "neg"},
        {"doc_name": "d", "para_idx": 0, "entity": "France", "type": "nation", "stance": "pos"},
        {"doc_name": "d", "para_idx": 0, "entity": "Spain", "type": "nation", "stance": "pos"},
    ])
    m = G.compute_entity_metrics(primary, opus)
    assert m["n_primary"] == 1 and m["n_opus"] == 3
    assert m["n_matched"] == 1 and m["n_union"] == 3
    assert m["unmatched_primary_rate"] == 0.0          # all of primary matched
    assert m["unmatched_opus_rate"] == pytest.approx(2 / 3)  # 2 of opus's 3 unmatched
    assert m["name_match_rate"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# _entity_stance_map tie-break (documented: first row after the deterministic
# sort wins on a repeated normalized (doc, para, name) key)
# ---------------------------------------------------------------------------


def test_entity_stance_map_tiebreak_is_deterministic_by_sort_not_input_order():
    """gap B witness. Two entity rows collapse to the SAME normalized key
    (doc, para, 'congress') but carry conflicting stances. The documented rule is
    'first row after a deterministic sort wins' — and the sort's last key is
    `stance`, so the alphabetically-first stance ('neg' < 'pos') must win
    REGARDLESS of the order the rows arrive in. Feed both input orderings and
    assert the same winner: this proves the tie-break is the SORT, not incidental
    row order (a mutation dropping `stance` from the sort keys, or taking the last
    row instead of the first, is caught)."""
    rows_pos_first = _ent([
        {"doc_name": "d", "para_idx": 0, "entity": "Congress", "type": "org", "stance": "pos"},
        {"doc_name": "d", "para_idx": 0, "entity": "congress ", "type": "org", "stance": "neg"},
    ])
    rows_neg_first = _ent([
        {"doc_name": "d", "para_idx": 0, "entity": "congress ", "type": "org", "stance": "neg"},
        {"doc_name": "d", "para_idx": 0, "entity": "Congress", "type": "org", "stance": "pos"},
    ])
    key = ("d", 0, "congress")

    m_pos_first = G._entity_stance_map(rows_pos_first)
    m_neg_first = G._entity_stance_map(rows_neg_first)

    # exactly one entry for the collapsed key (deduped, not double-counted)
    assert list(m_pos_first) == [key]
    assert list(m_neg_first) == [key]
    # deterministic winner is the sort-first stance, independent of arrival order
    assert m_pos_first[key] == "neg"
    assert m_neg_first[key] == "neg"


# ---------------------------------------------------------------------------
# build_metrics_table
# ---------------------------------------------------------------------------


def _para_row(doc, idx, era_bin, *, pa, en, zs, topics_p, topics_o, pv_p, pv_o,
              pa_o=None, en_o=None, zs_o=None):
    return dict(
        doc_name=doc, para_idx=idx, era_bin=era_bin,
        party_attack_primary=pa, party_attack_opus=pa if pa_o is None else pa_o,
        enemy_naming_primary=en, enemy_naming_opus=en if en_o is None else en_o,
        zero_sum_primary=zs, zero_sum_opus=zs if zs_o is None else zs_o,
        topics_primary=topics_p, topics_opus=topics_o,
        proposal_values_primary=pv_p, proposal_values_opus=pv_o,
    )


def _merged_two_bins() -> pd.DataFrame:
    return pd.DataFrame([
        _para_row("d0", 0, 0, pa=True, en=False, zs=True,
                  topics_p=["x"], topics_o=["x"], pv_p="proposal", pv_o="proposal"),
        _para_row("d0", 1, 0, pa=False, en=False, zs=True, pa_o=True,
                  topics_p=["x", "y"], topics_o=["y"], pv_p="values", pv_o="proposal"),
        _para_row("d1", 0, 3, pa=True, en=True, zs=False,
                  topics_p=["z"], topics_o=["z"], pv_p="proposal", pv_o="proposal"),
    ])


def _ent_frame(rows):
    cols = ["doc_name", "para_idx", "entity", "type", "stance", "era_bin"]
    return pd.DataFrame(rows, columns=cols)


def test_metrics_table_has_the_field_era_metric_value_n_shape():
    merged = _merged_two_bins()
    pe = _ent_frame([{"doc_name": "d0", "para_idx": 0, "entity": "A", "type": "t", "stance": "pos", "era_bin": 0}])
    oe = _ent_frame([{"doc_name": "d0", "para_idx": 0, "entity": "A", "type": "t", "stance": "neg", "era_bin": 0}])
    table = G.build_metrics_table(merged, pe, oe, None)
    assert list(table.columns) == ["field", "era_bin", "era_label", "metric", "value", "n"]


def test_metrics_table_contains_the_overall_row_for_every_field():
    merged = _merged_two_bins()
    empty = _ent_frame([])
    table = G.build_metrics_table(merged, empty, empty, None)

    overall = table[table["era_bin"] == G.OVERALL_BIN]
    assert not overall.empty  # non-empty guard before the per-row check below
    assert (overall["era_label"] == "overall").all()
    # every judgment field has an overall cell
    fields = set(overall["field"])
    assert {"party_attack", "enemy_naming", "zero_sum", "topics",
            "proposal_values", "entities"} <= fields


def test_metrics_table_reports_correct_per_bin_and_overall_n():
    """overall n = all 3 paragraphs; bin 0 has 2, bin 3 has 1. n must be the
    real per-scope sample size, not a constant."""
    merged = _merged_two_bins()
    empty = _ent_frame([])
    table = G.build_metrics_table(merged, empty, empty, None)

    pa = table[(table["field"] == "party_attack") & (table["metric"] == "cohen_kappa")]
    n_by_bin = dict(zip(pa["era_bin"], pa["n"]))
    assert n_by_bin[G.OVERALL_BIN] == 3
    assert n_by_bin[0] == 2
    assert n_by_bin[3] == 1


def test_metrics_table_only_covers_bins_present_in_the_merged_frame():
    merged = _merged_two_bins()  # bins 0 and 3 only
    empty = _ent_frame([])
    table = G.build_metrics_table(merged, empty, empty, None)
    bins = set(table["era_bin"])
    assert bins == {G.OVERALL_BIN, 0, 3}


def test_metrics_table_omits_factual_fields_when_no_speech_frame():
    merged = _merged_two_bins()
    empty = _ent_frame([])
    table = G.build_metrics_table(merged, empty, empty, None)
    assert not (set(G.FACTUAL_FIELDS) & set(table["field"]))


def test_metrics_table_includes_factual_fields_when_speech_frame_present():
    merged = _merged_two_bins()
    empty = _ent_frame([])
    speech = pd.DataFrame([
        {"doc_name": "d0", "era_bin": 0, "speech_type_primary": "address",
         "speech_type_opus": "address", "audience_primary": "nation",
         "audience_opus": "nation", "medium_primary": "spoken", "medium_opus": "written"},
        {"doc_name": "d1", "era_bin": 3, "speech_type_primary": "address",
         "speech_type_opus": "letter", "audience_primary": "congress",
         "audience_opus": "congress", "medium_primary": "written", "medium_opus": "written"},
    ])
    table = G.build_metrics_table(merged, empty, empty, speech)
    factual = table[table["field"].isin(G.FACTUAL_FIELDS)]
    assert not factual.empty
    # each factual field is measured with BOTH cohen_kappa and exact_match
    for f in G.FACTUAL_FIELDS:
        metrics = set(factual[factual["field"] == f]["metric"])
        assert metrics == {"cohen_kappa", "exact_match"}


def test_metrics_table_entity_rows_carry_all_four_entity_metrics():
    merged = _merged_two_bins()
    pe = _ent_frame([{"doc_name": "d0", "para_idx": 0, "entity": "A", "type": "t", "stance": "pos", "era_bin": 0}])
    oe = _ent_frame([{"doc_name": "d0", "para_idx": 0, "entity": "A", "type": "t", "stance": "pos", "era_bin": 0}])
    table = G.build_metrics_table(merged, pe, oe, None)
    ent = table[table["field"] == "entities"]
    overall_metrics = set(ent[ent["era_bin"] == G.OVERALL_BIN]["metric"])
    assert overall_metrics == {
        "name_match_rate", "stance_agreement",
        "unmatched_primary_rate", "unmatched_opus_rate",
    }


# ---------------------------------------------------------------------------
# paragraph_disagreement (drives the qualitative "top 10 disagreements" section)
# ---------------------------------------------------------------------------


def test_paragraph_disagreement_is_zero_on_agreement_and_positive_on_divergence():
    merged = pd.DataFrame([
        _para_row("d0", 0, 0, pa=True, en=False, zs=True,
                  topics_p=["x"], topics_o=["x"], pv_p="p", pv_o="p"),
        _para_row("d0", 1, 0, pa=False, en=False, zs=True, pa_o=True,
                  topics_p=["x", "y"], topics_o=["y"], pv_p="v", pv_o="p"),
    ])
    empty = _ent_frame([])
    scored = G.paragraph_disagreement(merged, empty, empty)
    by_key = {(r.doc_name, r.para_idx): r.disagreement for r in scored.itertuples(index=False)}
    # fully agreeing paragraph scores exactly 0
    assert by_key[("d0", 0)] == pytest.approx(0.0)
    # party_attack flip (+1) + topics {x,y} vs {y} (1 - 0.5) + proposal_values
    # flip (+1) = 2.5
    assert by_key[("d0", 1)] == pytest.approx(2.5)


def test_paragraph_disagreement_counts_the_entity_name_term():
    """The 4th disagreement component — (1 - Jaccard) of the paragraphs' normalized
    entity-name sets — was previously exercised only with EMPTY entity frames, so a
    dropped/zeroed entity term went unseen. Two paragraphs identical on every
    judgment field: one where the models name DISJOINT entities (term = +1) and one
    where they name the SAME entity (term = 0). This pins the entity contribution
    as a genuine set-difference, not mere presence."""
    merged = pd.DataFrame([
        _para_row("d0", 0, 0, pa=True, en=False, zs=True,
                  topics_p=["x"], topics_o=["x"], pv_p="p", pv_o="p"),
        _para_row("d0", 1, 0, pa=True, en=False, zs=True,
                  topics_p=["x"], topics_o=["x"], pv_p="p", pv_o="p"),
    ])
    primary_ent = _ent_frame([
        {"doc_name": "d0", "para_idx": 0, "entity": "Alpha", "type": "org", "stance": "neg", "era_bin": 0},
        {"doc_name": "d0", "para_idx": 1, "entity": "Gamma", "type": "org", "stance": "neg", "era_bin": 0},
    ])
    opus_ent = _ent_frame([
        {"doc_name": "d0", "para_idx": 0, "entity": "Beta", "type": "org", "stance": "neg", "era_bin": 0},
        {"doc_name": "d0", "para_idx": 1, "entity": "gamma", "type": "org", "stance": "neg", "era_bin": 0},
    ])
    scored = G.paragraph_disagreement(merged, primary_ent, opus_ent)
    by_key = {(r.doc_name, r.para_idx): r.disagreement for r in scored.itertuples(index=False)}
    # disjoint entity names {alpha} vs {beta} -> (1 - 0) = +1, nothing else differs
    assert by_key[("d0", 0)] == pytest.approx(1.0)
    # same entity name (case-normalized) -> entity term 0, total 0
    assert by_key[("d0", 1)] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# era binning helpers
# ---------------------------------------------------------------------------


def test_era_bin_anchors_at_1789_in_30_year_blocks():
    assert G.era_bin(1789) == 0
    assert G.era_bin(1818) == 0
    assert G.era_bin(1819) == 1
    assert G.era_bin(2000) == 7


def test_era_label_renders_bin_range_and_overall_sentinel():
    assert G.era_label(0) == "1789-1818"
    assert G.era_label(1) == "1819-1848"
    assert G.era_label(G.OVERALL_BIN) == "overall"


def test_kappa_excludes_null_pairs_pairwise_with_reduced_n():
    """A null on EITHER side of a pair is a missing observation: the pair is
    excluded and n counts only compared pairs (sklearn would raise on mixed
    None/bool labels — one bad ingested row must never crash the build)."""
    value, n = G.compute_kappa([True, False, None, True], [True, True, False, None])
    assert n == 2  # the two null-touched pairs dropped
    assert value == pytest.approx(0.0)  # a=[T,F] vs b=[T,T]: agreement at chance
