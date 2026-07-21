"""Unit tests for `convergence.py`.

Everything here runs on hand-countable synthetic frames — never the 36k-row
corpus and never a real permutation run. The heavy statistical claims (flat on
a clustered null, detects injected convergence, permutation size ~5%) are the
job of `convergence._selftest`, which gates the real run; these tests cover the
pieces that a selftest pass would not notice: the composition equation, the
merge guards, the window grid, the trust gate, the banned-p-value wrapper, the
permutation bijection and the decision table.
"""

from __future__ import annotations

import ast
import json

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import convergence as C


# --------------------------------------------------------------------------
# the composition equation (prereg section 3)
# --------------------------------------------------------------------------


class TestCompositionEquation:
    def test_every_paragraph_contributes_exactly_mass_one(self):
        mass = C._mass_from_label_lists([[0], [0, 1], [0, 1, 2], []], 4)
        assert np.allclose(mass.sum(axis=1), 1.0)

    def test_multi_label_paragraph_splits_one_over_k(self):
        mass = C._mass_from_label_lists([[0, 2]], 4)
        assert mass[0, 0] == pytest.approx(0.5)
        assert mass[0, 2] == pytest.approx(0.5)
        assert mass[0, 1] == 0.0

    def test_unlabelled_paragraph_lands_in_the_no_topic_bin(self):
        mass = C._mass_from_label_lists([[]], 4)
        assert mass[0, 3] == pytest.approx(1.0)
        assert mass[0, :3].sum() == 0.0

    def test_label_density_alone_cannot_move_the_measure(self):
        """The whole reason for the 1/k split: the corpus's labels-per-paragraph
        drifts 1.49 -> 0.83, and a density-sensitive normalization would read
        that drift as a change in agenda."""
        sparse = C._mass_from_label_lists([[0], [1]], 3).mean(axis=0)
        dense = C._mass_from_label_lists([[0, 1, 2], [0, 1, 2]], 3).mean(axis=0)
        assert sparse.sum() == pytest.approx(dense.sum()) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# build_compositions: guards
# --------------------------------------------------------------------------


def _issue_frame(n_docs: int = 2, per_doc: int = 3) -> pd.DataFrame:
    rows = []
    for d in range(n_docs):
        for i in range(per_doc):
            row = {
                "doc_name": f"doc{d}", "para_idx": i,
                "president": f"P{d}", "year": 1800 + d,
            }
            row.update({name: False for name in C.ISSUES})
            row[C.ISSUES[i % len(C.ISSUES)]] = True
            rows.append(row)
    return pd.DataFrame(rows)


def _speech_frame(n_docs: int = 2) -> pd.DataFrame:
    return pd.DataFrame({
        "doc_name": [f"doc{d}" for d in range(n_docs)],
        "speech_type": [C.SOTU_TYPE] * n_docs,
    })


class TestBuildCompositionsGuards:
    def test_missing_speech_type_raises_rather_than_dropping_paragraphs(self):
        with pytest.raises(ValueError, match="speech_type"):
            C.build_compositions(
                "corex", issues=_issue_frame(2),
                speech_annotations=_speech_frame(1),
            )

    def test_diverging_annotation_keys_raise_instead_of_inner_joining(self):
        issues = _issue_frame(2)
        taxonomy = {"level2": [{"name": "Trade", "level1": "Economy"}]}
        ann = pd.DataFrame({
            "doc_name": ["doc0"] * 3, "para_idx": [0, 1, 2],
            "topics": [["Trade"]] * 3,
        })
        with pytest.raises(ValueError, match="keyed merge changed the row count"):
            C.build_compositions(
                "llm", issues=issues, annotations=ann,
                speech_annotations=_speech_frame(2), taxonomy=taxonomy,
            )

    def test_unmapped_topic_label_raises(self):
        issues = _issue_frame(1, per_doc=1)
        taxonomy = {"level2": [{"name": "Trade", "level1": "Economy"}]}
        ann = pd.DataFrame({
            "doc_name": ["doc0"], "para_idx": [0], "topics": [["Nonexistent Topic"]],
        })
        with pytest.raises(ValueError, match="not in taxonomy_v1"):
            C.build_compositions(
                "llm", issues=issues, annotations=ann,
                speech_annotations=_speech_frame(1), taxonomy=taxonomy,
            )

    def test_case_variant_labels_normalize_to_one_bin(self):
        issues = _issue_frame(1, per_doc=2)
        taxonomy = {"level2": [{"name": "War Of 1812", "level1": "War"}]}
        ann = pd.DataFrame({
            "doc_name": ["doc0", "doc0"], "para_idx": [0, 1],
            "topics": [["war of 1812"], ["War Of 1812"]],
        })
        comp = C.build_compositions(
            "llm", issues=issues, annotations=ann,
            speech_annotations=_speech_frame(1), taxonomy=taxonomy,
        )
        assert comp.bins == ["War Of 1812", C.NO_TOPIC]
        assert comp.mass[:, 0].tolist() == [1.0, 1.0]

    def test_duplicate_topic_pairs_are_deduplicated_before_counting(self):
        issues = _issue_frame(1, per_doc=1)
        taxonomy = {"level2": [{"name": "Trade", "level1": "Economy"}]}
        ann = pd.DataFrame({
            "doc_name": ["doc0"], "para_idx": [0], "topics": [["Trade", "trade"]],
        })
        comp = C.build_compositions(
            "llm", issues=issues, annotations=ann,
            speech_annotations=_speech_frame(1), taxonomy=taxonomy,
        )
        assert comp.n_label_assignments == 1
        assert comp.mass[0, 0] == pytest.approx(1.0)

    def test_rows_come_back_grouped_by_doc_name(self):
        """`build_design` indexes each speech as a contiguous row block, so an
        ungrouped Composition would silently mix speeches together."""
        issues = _issue_frame(3).sample(frac=1.0, random_state=0)
        comp = C.build_compositions(
            "corex", issues=issues, speech_annotations=_speech_frame(3)
        )
        codes, _ = pd.factorize(comp.doc_name)
        assert (np.diff(codes) >= 0).all()


# --------------------------------------------------------------------------
# window grid + trust gate
# --------------------------------------------------------------------------


class TestWindows:
    def test_rolling_grid_covers_the_whole_corpus_span(self):
        spans = C.rolling_windows(1789, 2026)
        assert len(spans) == 105
        assert spans[0] == (1789, 1818)
        assert spans[-1] == (1997, 2026)
        assert all(e - s + 1 == C.WINDOW_LEN for s, e in spans)

    def test_every_corpus_year_falls_inside_at_least_one_window(self):
        spans = C.rolling_windows(1789, 2026)
        covered = {y for s, e in spans for y in range(s, e + 1)}
        assert set(range(1789, 2027)) <= covered

    def test_nonoverlapping_windows_are_disjoint_and_eight(self):
        spans = C.nonoverlapping_windows(1789, 2026)
        assert len(spans) == 8
        for (a_s, a_e), (b_s, _) in zip(spans, spans[1:]):
            assert a_e < b_s

    @pytest.mark.parametrize(
        "n,expected",
        [(0, "no_data"), (1, "no_data"), (2, "suppressed_n_floor"),
         (3, "low_cluster_caution"), (4, "low_cluster_caution"), (5, "ok"), (9, "ok")],
    )
    def test_trust_gate_thresholds(self, n, expected):
        assert C.window_status(n) == expected

    def test_windows_below_the_floor_never_enter_a_trend(self):
        comp = C._synthetic_composition(
            np.random.default_rng([1]), n_presidents=8, speeches_first=10,
            speeches_last=12, paragraphs_per_speech=8,
        )
        design = C._selftest_design(comp)
        curve = C.dispersion_curve(
            design, [1, 1], with_floor=False, with_entropy_match=False
        )
        suppressed = [
            i for i, s in enumerate(curve.status)
            if s in {"no_data", "suppressed_n_floor"}
        ]
        assert all(not curve.used[i] for i in suppressed)
        assert all(np.isnan(curve.dispersion[i]) for i in suppressed)


# --------------------------------------------------------------------------
# the banned p-value
# --------------------------------------------------------------------------


class TestSpearmanWrapper:
    def test_returns_a_bare_float_so_no_caller_can_reach_the_pvalue(self):
        rho = C.spearman_rho([1, 2, 3, 4], [1, 2, 3, 4])
        assert isinstance(rho, float)
        assert rho == pytest.approx(1.0)

    def test_monotone_decline_is_minus_one(self):
        assert C.spearman_rho([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)

    def test_degenerate_input_is_nan_not_a_warning(self):
        assert np.isnan(C.spearman_rho([1, 2], [1, 2]))
        assert np.isnan(C.spearman_rho([1, 1, 1], [1, 2, 3]))

    def test_no_executable_line_ever_reads_scipys_pvalue(self):
        """The ban is enforced in CODE, not by discipline. Scanned over the AST
        so that the docstring which EXPLAINS the ban cannot satisfy the test —
        a source-text scan would pass on the comment alone."""
        tree = ast.parse(open(C.__file__).read())
        reads = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "pvalue"
        ]
        assert not reads, f"{len(reads)} executable reads of `.pvalue`"

    def test_that_scan_would_catch_the_defect_it_was_written_for(self):
        """Mutate the original defect back in and confirm the guard fails —
        a guard that is blind to its own defect shape is not a guard."""
        tree = ast.parse("from scipy.stats import spearmanr\np = spearmanr(a, b).pvalue")
        reads = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "pvalue"
        ]
        assert reads


# --------------------------------------------------------------------------
# the metric
# --------------------------------------------------------------------------


class TestJensenShannon:
    def test_identical_distributions_are_zero(self):
        c = np.array([[[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]]])
        assert C.pairwise_jsd(c)[0, 0, 1] == pytest.approx(0.0, abs=1e-9)

    def test_disjoint_distributions_are_one_bit(self):
        c = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        assert C.pairwise_jsd(c)[0, 0, 1] == pytest.approx(1.0)

    def test_symmetric(self):
        rng = np.random.default_rng(0)
        c = rng.dirichlet(np.ones(5), size=(1, 4))
        j = C.pairwise_jsd(c)
        assert np.allclose(j, np.swapaxes(j, -1, -2))

    def test_bin_shuffle_preserves_entropy_exactly(self):
        """The entropy-matched rival test rests on this: shuffling bin order is
        a permutation of the same probability values, so breadth is held fixed
        while agreement is destroyed."""
        rng = np.random.default_rng(0)
        c = rng.dirichlet(np.ones(6), size=(3, 4))
        perm = np.argsort(rng.random(c.shape), axis=-1, kind="stable")
        shuffled = np.take_along_axis(c, perm, axis=-1)
        assert np.allclose(C._entropy(c), C._entropy(shuffled))


# --------------------------------------------------------------------------
# the sampler
# --------------------------------------------------------------------------


def _small_design(seed: int = 5) -> C.Design:
    comp = C._synthetic_composition(
        np.random.default_rng([seed]), n_presidents=10, speeches_first=10,
        speeches_last=20, paragraphs_per_speech=10,
    )
    return C._selftest_design(comp)


class TestSampler:
    def test_cluster_rarefaction_always_uses_exactly_m_paragraphs(self):
        design = _small_design()
        window = next(w for w in design.windows if len(w.presidents) >= 2)
        c = C.draw_compositions(
            design, list(window.pools), np.random.default_rng(0), 4
        )
        assert c.shape == (4, len(window.pools), design.n_bins)
        assert np.allclose(c.sum(axis=-1), 1.0)

    def test_draws_are_reproducible_from_the_seed(self):
        design = _small_design()
        a = C.dispersion_curve(design, [7, 7], with_floor=False, with_entropy_match=False)
        b = C.dispersion_curve(design, [7, 7], with_floor=False, with_entropy_match=False)
        assert np.array_equal(np.nan_to_num(a.dispersion), np.nan_to_num(b.dispersion))

    def test_a_different_seed_gives_a_different_curve(self):
        design = _small_design()
        a = C.dispersion_curve(design, [7, 7], with_floor=False, with_entropy_match=False)
        b = C.dispersion_curve(design, [7, 8], with_floor=False, with_entropy_match=False)
        assert not np.array_equal(np.nan_to_num(a.dispersion), np.nan_to_num(b.dispersion))

    def test_speech_block_floor_is_finite_and_bounded(self):
        design = _small_design()
        window = next(w for w in design.windows if len(w.presidents) >= 3)
        value, per_slot = C.speech_block_floor(
            design, list(window.pools), np.random.default_rng(0), 8
        )
        assert 0.0 <= value <= 1.0
        assert len(per_slot) == len(window.pools)

    def test_paragraph_eligibility_is_never_stricter_than_speech_eligibility(self):
        """The superseded design's rule admits presidents the pre-registered one
        excludes — which is exactly where its design effect is worst."""
        comp = C._synthetic_composition(
            np.random.default_rng([9]), n_presidents=10, speeches_first=10,
            speeches_last=20, paragraphs_per_speech=10,
        )
        strict = C._selftest_design(comp)
        loose = C._selftest_design(comp, eligibility="paragraphs")
        for a, b in zip(strict.windows, loose.windows):
            assert set(a.presidents.tolist()) <= set(b.presidents.tolist())

    def test_centered_tilts_have_the_intended_mean_at_every_n(self):
        rng = np.random.default_rng(0)
        mu = np.full(6, 1 / 6)
        for n in (5, 40):
            theta = C._centered_tilts(rng, mu, n, 9.0)
            assert np.allclose(theta.mean(axis=0), mu, atol=5e-3)
            assert (theta >= 0).all()


# --------------------------------------------------------------------------
# permutation machinery
# --------------------------------------------------------------------------


class TestPermutation:
    def test_donor_map_is_a_bijection_over_the_ever_eligible_set(self):
        design = _small_design()
        pool = C.ever_eligible(design)
        donor = C._donor_map(design, np.random.default_rng(0))
        assert sorted(donor[pool].tolist()) == sorted(pool.tolist())

    def test_donor_map_leaves_never_eligible_presidents_alone(self):
        design = _small_design()
        pool = set(C.ever_eligible(design).tolist())
        donor = C._donor_map(design, np.random.default_rng(0))
        for i in range(len(design.presidents)):
            if i not in pool:
                assert donor[i] == i

    def test_permutation_does_not_change_which_windows_are_used(self):
        """The design is held fixed; only the content moves. If a permutation
        could change the retained window set, the null and the observed
        statistic would not be computed on the same windows."""
        design = _small_design()
        obs = C.dispersion_curve(design, [1, 1], with_floor=False, with_entropy_match=False)
        donor = C._donor_map(design, np.random.default_rng(3))
        perm = C.dispersion_curve(
            design, [1, 2], donor=donor, with_floor=False, with_entropy_match=False
        )
        assert np.array_equal(obs.used, perm.used)
        assert obs.status == perm.status

    @pytest.mark.parametrize("observed,expected", [(-1.0, 1 / 5), (1.0, 5 / 5)])
    def test_one_sided_p_uses_the_plus_one_correction(self, observed, expected):
        null = np.array([-0.5, 0.0, 0.5, 1.0])
        assert C._one_sided_p(observed, null) == pytest.approx(expected)

    def test_p_is_never_zero(self):
        null = np.zeros(1000)
        assert C._one_sided_p(-5.0, null) > 0


# --------------------------------------------------------------------------
# decision table (prereg section 8)
# --------------------------------------------------------------------------


def _null_table(**flags) -> pd.DataFrame:
    rows = []
    for arm in ("corex_all", "corex_sotu"):
        for statistic in ("dispersion", "floor_corrected", "excess_ratio"):
            rows.append({
                "arm": arm, "window_kind": "rolling", "treatment": "all_windows",
                "statistic": statistic,
                "significant_decline": flags.get(f"{arm}.{statistic}", False),
            })
    return pd.DataFrame(rows)


class TestDecisionTable:
    def test_no_convergence_when_neither_arm_declines(self):
        assert C.score_decision_table(_null_table())["cell"] == "no_convergence"

    def test_artifact_when_the_two_arms_disagree(self):
        table = _null_table(**{"corex_all.dispersion": True})
        assert C.score_decision_table(table)["cell"] == "artifact_arms_disagree"

    def test_artifact_when_the_floor_explains_the_decline(self):
        table = _null_table(**{
            "corex_all.dispersion": True, "corex_sotu.dispersion": True,
            "corex_all.floor_corrected": True,
        })
        assert C.score_decision_table(table)["cell"] == "artifact_floor"

    def test_broadening_when_the_entropy_matched_null_accounts_for_it(self):
        table = _null_table(**{
            "corex_all.dispersion": True, "corex_sotu.dispersion": True,
            "corex_all.floor_corrected": True, "corex_sotu.floor_corrected": True,
        })
        assert C.score_decision_table(table)["cell"] == "broadening"

    def test_convergence_needs_every_gate(self):
        table = _null_table(**{
            "corex_all.dispersion": True, "corex_sotu.dispersion": True,
            "corex_all.floor_corrected": True, "corex_sotu.floor_corrected": True,
            "corex_all.excess_ratio": True, "corex_sotu.excess_ratio": True,
        })
        assert C.score_decision_table(table)["cell"] == "convergence"

    def test_every_cell_has_its_literal_pre_committed_headline(self):
        prereg = (
            C.DATA_DIR.parent / "notes" / "convergence-prereg-v1.md"
        ).read_text()
        for cell, headline in C.HEADLINES.items():
            # The pre-registration writes the sentences with typographic
            # punctuation; compare on a normalized form rather than byte-equal.
            needle = headline.replace("1789-2026", "1789–2026").replace(" - ", " — ")
            assert needle in prereg, f"headline for {cell} is not in the prereg"

    def test_a_missing_input_raises_rather_than_defaulting_to_a_cell(self):
        table = _null_table()
        with pytest.raises(ValueError, match="decision table needs"):
            C.score_decision_table(table[table["arm"] != "corex_sotu"])


# --------------------------------------------------------------------------
# published artifact contract
# --------------------------------------------------------------------------


class TestArtifactContract:
    def test_meta_carries_no_wall_clock_stamp(self):
        if not C.META_PATH.exists():
            pytest.skip("data/convergence/ has not been generated in this tree")
        text = C.META_PATH.read_text()
        for banned in ("generated_at", "timestamp", "run_date", "today"):
            assert banned not in text

    def test_every_plottable_row_carries_a_trust_gate(self):
        if not C.DISPERSION_PATH.exists():
            pytest.skip("data/convergence/ has not been generated in this tree")
        curves = pd.read_parquet(C.DISPERSION_PATH)
        assert curves["ci_status"].notna().all()
        assert set(curves["ci_status"]) <= {
            "no_data", "suppressed_n_floor", "low_cluster_caution", "ok"
        }
        blind = curves[curves["ci_status"].isin(["no_data", "suppressed_n_floor"])]
        assert blind["used_in_trend"].eq(False).all()

    def test_meta_records_zero_api_calls(self):
        if not C.META_PATH.exists():
            pytest.skip("data/convergence/ has not been generated in this tree")
        assert json.loads(C.META_PATH.read_text())["api_calls"] == 0

    def test_module_never_imports_anthropic(self):
        """$0 GUARD, scanned over the AST so that the docstring PROMISING it
        cannot be what satisfies the test."""
        tree = ast.parse(open(C.__file__).read())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "anthropic" not in imported
