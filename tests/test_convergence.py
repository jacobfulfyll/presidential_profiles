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
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import convergence as C


# --------------------------------------------------------------------------
# hand-built corpora
# --------------------------------------------------------------------------
#
# `C._synthetic_composition` is the *selftest's* generator: it exists to carry
# the review's measured confound (drifting speech supply, within-speech ICC) and
# nothing else. Several tests below need to dictate a president's agenda exactly
# — who is eligible when, and what each of them talks about — so they build a
# `Composition` directly from an explicit per-speech spec instead. Both routes
# produce the same NamedTuple, and every test drives the real `build_design` /
# `dispersion_curve` from it.


def _hand_corpus(
    presidents: list[tuple[str, list[int], np.ndarray | None]],
    n_bins: int,
    *,
    paras_per_speech: int = 10,
    seed: int = 0,
    speech_type: str = C.SOTU_TYPE,
) -> C.Composition:
    """A Composition from `[(president, [year of each speech], bin weights)]`.

    `weights is None` puts every paragraph in bin 0 — enough for the design-level
    tests, which care about who is eligible where and not about what was said.
    """
    rng = np.random.default_rng([seed])
    docs, idxs, pres, years, labels = [], [], [], [], []
    for name, speech_years, weights in presidents:
        for s, year in enumerate(speech_years):
            doc = f"{name}-s{s:04d}"
            picks = (
                np.zeros(paras_per_speech, dtype=int) if weights is None
                else rng.choice(n_bins, size=paras_per_speech, p=weights)
            )
            for i, b in enumerate(picks):
                docs.append(doc)
                idxs.append(i)
                pres.append(name)
                years.append(year)
                labels.append([int(b)])
    bins = [f"bin{j}" for j in range(n_bins)] + [C.NO_TOPIC]
    return C.Composition(
        label_source="synthetic",
        bins=bins,
        mass=C._mass_from_label_lists(labels, len(bins)),
        doc_name=np.array(docs),
        para_idx=np.array(idxs),
        president=np.array(pres),
        year=np.array(years, dtype=np.int64),
        speech_type=np.array([speech_type] * len(docs)),
        n_label_assignments=len(labels),
    )


# A deliberately small arm for the design-level tests: S=2 speeches x B=2
# paragraphs, so eligibility can be reasoned about by hand.
_TINY_ARM = C.ArmSpec("tiny", "corex", None, 2, 2, "test")


def _coextensive_corpus(
    n_presidents: int, n_bins: int = 4, distinct_agendas: bool = False
) -> C.Composition:
    """`n_presidents` presidents who each speak every other year, 1800-1860.

    Every president is therefore eligible in every 30-year window, which makes
    the eligible count per window exactly `n_presidents` — the cleanest possible
    handle on the `MIN_PRESIDENTS` floor.

    `distinct_agendas` gives each president their own tilt. The design-level
    tests do not care what anyone said, but `paradox_table`'s leave-one-out z
    divides by the sd of the OTHER presidents' corrected distances, which is
    exactly 0 when every president is a copy of every other.
    """
    speech_years = list(range(1800, 1861, 2))
    specs = []
    for i in range(n_presidents):
        weights = None
        if distinct_agendas:
            weights = np.full(n_bins, 0.15 / (n_bins - 1))
            weights[i % n_bins] = 0.85
            weights /= weights.sum()
        specs.append((f"P{i}", speech_years, weights))
    return _hand_corpus(specs, n_bins)

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

    def test_an_unknown_label_source_raises(self):
        with pytest.raises(ValueError, match="unknown label_source"):
            C.build_compositions("corex_but_better", issues=_issue_frame(1))

    def test_a_missing_anchored_issue_column_raises(self):
        """`paragraph_issues` losing a column would silently shrink the
        composition's bin set and change every JSD in the module."""
        issues = _issue_frame(2).drop(columns=[C.ISSUES[3]])
        with pytest.raises(ValueError, match="missing anchored issue columns"):
            C.build_compositions(
                "corex", issues=issues, speech_annotations=_speech_frame(2)
            )

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


# --------------------------------------------------------------------------
# the banned p-value: the scanner
# --------------------------------------------------------------------------
#
# `scipy.stats.spearmanr` returns a (statistic, pvalue) pair, and every way of
# getting at element 1 of it is banned from this analysis — measured type-I
# error on these 93%-overlapping windows is 58%. The original scan looked only
# for `.pvalue` attribute access and was therefore blind to `spearmanr(a,b)[1]`
# and `_, p = spearmanr(a,b)`, which read exactly the same number. There is no
# live instance of either; this is guard COMPLETENESS, and every spelling below
# is proved to fire by mutating it back in.

_SPEARMAN_FUNCTIONS = {"spearmanr"}


def _is_spearman_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        (isinstance(func, ast.Name) and func.id in _SPEARMAN_FUNCTIONS)
        or (isinstance(func, ast.Attribute) and func.attr in _SPEARMAN_FUNCTIONS)
    )


def _takes_element_zero(node: ast.Subscript) -> bool:
    """True only for a literal `[0]`. Anything the scan cannot resolve — a
    variable index, a slice — is deliberately NOT element zero, so the scan
    fails towards a false leak rather than a silent miss."""
    index = node.slice
    return isinstance(index, ast.Constant) and index.value == 0


def _pvalue_reads(source: str) -> list[str]:
    """Every executable read of scipy's p-value, in any of its spellings."""
    tree = ast.parse(source)
    findings: list[str] = []

    # Names anywhere in the source that hold a spearmanr result.
    bound: set[str] = set()
    for node in ast.walk(tree):
        value = getattr(node, "value", None)
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not _is_spearman_call(value):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                bound.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                findings.append(
                    f"line {node.lineno}: tuple-unpack of spearmanr() binds the "
                    f"p-value as well as rho"
                )

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "pvalue":
            findings.append(f"line {node.lineno}: `.pvalue` attribute access")
        elif isinstance(node, ast.Subscript):
            value = node.value
            on_a_result = _is_spearman_call(value) or (
                isinstance(value, ast.Name) and value.id in bound
            )
            if on_a_result and not _takes_element_zero(node):
                findings.append(
                    f"line {node.lineno}: subscript of a spearmanr() result that "
                    f"is not the literal `[0]`"
                )
    return findings


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
        reads = _pvalue_reads(open(C.__file__).read())
        assert reads == [], f"{len(reads)} executable reads of scipy's p-value"

    @pytest.mark.parametrize(
        "spelling,source",
        [
            ("attribute on the call",
             "from scipy.stats import spearmanr\np = spearmanr(a, b).pvalue"),
            ("attribute on a bound name",
             "from scipy.stats import spearmanr\nr = spearmanr(a, b)\np = r.pvalue"),
            ("subscript on the call",
             "from scipy.stats import spearmanr\np = spearmanr(a, b)[1]"),
            ("subscript on a bound name",
             "from scipy.stats import spearmanr\nr = spearmanr(a, b)\np = r[1]"),
            ("tuple unpack",
             "from scipy.stats import spearmanr\n_, p = spearmanr(a, b)"),
            ("tuple unpack, both named",
             "from scipy.stats import spearmanr\nrho, p = spearmanr(x, y)"),
            ("qualified call",
             "import scipy.stats as st\np = st.spearmanr(a, b)[1]"),
            ("computed subscript index",
             "from scipy.stats import spearmanr\np = spearmanr(a, b)[i]"),
        ],
    )
    def test_that_the_scan_would_catch_every_defect_spelling(self, spelling, source):
        """Mutate each defect back in and confirm the guard fires. A guard blind
        to the shape it was written for is not a guard — and the original scan
        WAS blind to four of these eight."""
        assert _pvalue_reads(source), spelling

    @pytest.mark.parametrize(
        "source",
        [
            # the module's own spelling: `.statistic` off a bound result
            "from scipy.stats import spearmanr\nr = spearmanr(a, b)\nreturn float(r.statistic)",
            # element 0 is rho, which is the one thing callers may have
            "from scipy.stats import spearmanr\nrho = spearmanr(a, b)[0]",
            # an unrelated pair-unpack must not be mistaken for one
            "iu, ju = np.triu_indices(n, k=1)",
            # an unrelated subscript on an unrelated name
            "rows = frame[1]",
        ],
    )
    def test_the_scan_does_not_fire_on_legitimate_spellings(self, source):
        assert _pvalue_reads(source) == []

    def test_an_unrecognised_subscript_fails_safe_towards_a_false_leak(self):
        """A scan that cannot tell which element is being taken must report a
        violation, never wave it through: a missing alias has to produce a false
        positive, not a silent miss."""
        assert _pvalue_reads(
            "from scipy.stats import spearmanr\np = spearmanr(a, b)[k]"
        )
        assert _pvalue_reads(
            "from scipy.stats import spearmanr\np = spearmanr(a, b)[0:2]"
        )


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

    def test_an_unknown_window_grid_or_eligibility_rule_raises(self):
        comp = _coextensive_corpus(3)
        with pytest.raises(ValueError, match="unknown window_kind"):
            C.build_design(comp, _TINY_ARM, "sliding")
        with pytest.raises(ValueError, match="unknown eligibility"):
            C.build_design(comp, _TINY_ARM, "rolling", eligibility="documents")

    def test_an_ungrouped_composition_is_refused_not_silently_mixed(self):
        """The sampler indexes each speech as a contiguous block of rows. A
        Composition whose speeches interleave would have every draw silently
        span two speeches, destroying the cluster rarefaction the whole design
        rests on."""
        comp = _coextensive_corpus(2)
        order = np.argsort(comp.para_idx, kind="stable")   # groups by para_idx
        shuffled = comp._replace(
            mass=comp.mass[order], doc_name=comp.doc_name[order],
            para_idx=comp.para_idx[order], president=comp.president[order],
            year=comp.year[order], speech_type=comp.speech_type[order],
        )
        with pytest.raises(ValueError, match="not grouped by doc_name"):
            C.build_design(shuffled, _TINY_ARM)

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

    @pytest.mark.parametrize(
        "observed,expected",
        # null = [-2,-1,1,2], median 0. |v - 0| = [2,1,1,2].
        [(1.5, 3 / 5),   # two draws at distance >= 1.5
         (3.0, 1 / 5),   # none: only the +1 correction survives
         (0.0, 5 / 5)],  # all four
    )
    def test_two_sided_p_measures_distance_from_the_null_centre(
        self, observed, expected
    ):
        null = np.array([-2.0, -1.0, 1.0, 2.0])
        assert C._two_sided_p(observed, null) == pytest.approx(expected)

    def test_both_p_values_are_nan_when_there_is_nothing_to_compare(self):
        null = np.array([0.1, 0.2, 0.3])
        assert np.isnan(C._one_sided_p(float("nan"), null))
        assert np.isnan(C._two_sided_p(float("nan"), null))
        assert np.isnan(C._one_sided_p(0.5, np.full(3, np.nan)))
        assert np.isnan(C._two_sided_p(0.5, np.full(3, np.nan)))


# --------------------------------------------------------------------------
# which statistics each window grid contributes (prereg 7.4, 7.6, 7.7)
# --------------------------------------------------------------------------


class TestCurveStatisticsScopeRules:
    def _curve(self, window_kind: str) -> tuple[C.Curve, str]:
        design = C.build_design(
            _coextensive_corpus(5, distinct_agendas=True), _TINY_ARM, window_kind
        )
        return C.dispersion_curve(design, [3, 1], n_draws=4, floor_draws=4), window_kind

    def test_the_rolling_grid_contributes_ends_2014_and_the_pair_regression(self):
        curve, kind = self._curve("rolling")
        keys = set(C.curve_statistics(curve, kind))
        assert keys == {
            (treatment, statistic)
            for treatment in ("all_windows", "ends_2014")
            for statistic in ("dispersion", "floor_corrected", "excess_ratio")
        } | {("all_windows", "pair_regression")}

    def test_the_nonoverlapping_grid_contributes_neither(self):
        """`ends_2014` is a subset of the rolling grid and the pair regression is
        pre-registered on the rolling grid only; a non-overlapping run must not
        quietly manufacture either."""
        curve, kind = self._curve("nonoverlapping")
        keys = set(C.curve_statistics(curve, kind))
        assert keys == {
            ("all_windows", statistic)
            for statistic in ("dispersion", "floor_corrected", "excess_ratio")
        }

    def test_an_unknown_statistic_raises_rather_than_returning_a_default(self):
        curve, _ = self._curve("rolling")
        with pytest.raises(ValueError, match="unknown statistic"):
            C._series_for(curve, "made_up")

    def test_the_pair_regression_is_nan_below_three_president_pairs(self):
        design = C.build_design(_coextensive_corpus(2), _TINY_ARM)
        curve = C.dispersion_curve(
            design, [3, 1], n_draws=2, with_floor=False, with_entropy_match=False
        )
        assert np.isnan(C._pair_regression_rho(curve))


# --------------------------------------------------------------------------
# H2b: the paradox table, and the H2a anchor it may never score
# --------------------------------------------------------------------------


def _paradox_curve(design: C.Design, pres_jsd, pres_floor, pres_n) -> C.Curve:
    """A Curve carrying only the per-president accumulators `paradox_table`
    reads, so every published number below is hand-derivable."""
    n_win, n_pres = len(design.windows), len(design.presidents)
    zeros = np.zeros(n_win)
    return C.Curve(
        start=zeros, end=zeros, center=zeros, n_eligible=np.zeros(n_win, dtype=int),
        status=["ok"] * n_win, used=np.ones(n_win, dtype=bool),
        dispersion=zeros, entropy_matched=zeros, floor=zeros, mean_entropy=zeros,
        pair_jsd=np.zeros((n_pres, n_pres)), pair_year=np.zeros((n_pres, n_pres)),
        pair_n=np.zeros((n_pres, n_pres)),
        pres_jsd=np.asarray(pres_jsd, dtype=float),
        pres_floor=np.asarray(pres_floor, dtype=float),
        pres_n=np.asarray(pres_n, dtype=float),
    )


class TestParadoxTable:
    def _table(self) -> pd.DataFrame:
        design = C.build_design(_coextensive_corpus(4), _TINY_ARM)
        n = [1.0, 3.0, 10.0, 12.0]
        # mean JSD per president is 0.1 / 0.2 / 0.3 / 0.4; the floor is 0.
        curve = _paradox_curve(
            design, [0.1 * n[0], 0.2 * n[1], 0.3 * n[2], 0.4 * n[3]], [0.0] * 4, n
        )
        return C.paradox_table(design, curve, anchor=pd.Series(dtype=float))

    def test_corrected_distance_subtracts_the_speech_block_floor(self):
        design = C.build_design(_coextensive_corpus(3), _TINY_ARM)
        # mean JSD is 0.3 for all three; only the floor they are corrected
        # against differs, which is the whole point of the subtraction.
        curve = _paradox_curve(
            design, [0.6, 0.9, 1.2], [0.2, 0.5, 0.3], [2.0, 3.0, 4.0]
        )
        out = C.paradox_table(design, curve, anchor=pd.Series(dtype=float))
        by_president = out.set_index("president")["corrected_distance"]
        assert by_president["P0"] == pytest.approx(0.3 - 0.1)
        assert by_president["P1"] == pytest.approx(0.3 - 1 / 6)
        assert by_president["P2"] == pytest.approx(0.3 - 0.075)

    def test_the_z_score_compares_each_president_against_the_OTHERS(self):
        """A plain z would let an extreme president deflate the very mean and sd
        they are being scored on. With d = [0.1, 0.2, 0.3, 0.4] the others' mean
        for P0 is 0.3 and their ddof=1 sd is 0.1, so z = -2.0 exactly; a
        whole-sample z would give -1.162."""
        out = self._table().set_index("president")
        assert out.loc["P0", "z_vs_others"] == pytest.approx(-2.0)
        assert out.loc["P3", "z_vs_others"] == pytest.approx(2.0)

    def test_rows_are_ranked_by_corrected_distance(self):
        out = self._table()
        assert out["corrected_distance"].is_monotonic_increasing
        assert out["rank_corrected"].tolist() == [1, 2, 3, 4]
        assert out["percentile_corrected"].tolist() == [0.25, 0.5, 0.75, 1.0]

    def test_the_trust_gate_reads_windows_eligible(self):
        out = self._table().set_index("president")
        assert out.loc["P0", "ci_status"] == "suppressed_n_floor"   # 1 window
        assert out.loc["P1", "ci_status"] == "low_cluster_caution"  # 3 windows
        assert out.loc["P2", "ci_status"] == "ok"                   # 10 windows
        assert _trust_gate_violations({"paradox": self._table()}) == []

    def test_a_president_with_no_eligible_window_is_absent_not_nan(self):
        design = C.build_design(_coextensive_corpus(4), _TINY_ARM)
        curve = _paradox_curve(
            design, [0.1, 0.2, 0.4, 0.0], [0.0] * 4, [2.0, 2.0, 2.0, 0.0]
        )
        out = C.paradox_table(design, curve, anchor=pd.Series(dtype=float))
        assert set(out["president"]) == {"P0", "P1", "P2"}
        assert out["n_presidents_ranked"].eq(3).all()

    def test_the_h2a_anchor_is_carried_but_never_scored(self):
        design = C.build_design(_coextensive_corpus(3), _TINY_ARM)
        curve = _paradox_curve(design, [0.1, 0.2, 0.4], [0.0] * 3, [2.0] * 3)
        anchor = pd.Series({"P0": 0.42, "P1": 0.11, "P2": 0.77})
        out = C.paradox_table(design, curve, anchor=anchor)
        assert out.set_index("president")["stylistic_similarity_anchor"].to_dict() == {
            "P0": 0.42, "P1": 0.11, "P2": 0.77
        }
        assert not out["anchor_is_scored"].any()
        assert out["anchor_source"].eq(C.PRESIDENT_EMBEDDINGS_PATH.name).all()

    def test_a_missing_anchor_is_recorded_as_absent_not_fabricated(self):
        design = C.build_design(_coextensive_corpus(3), _TINY_ARM)
        curve = _paradox_curve(design, [0.1, 0.2, 0.4], [0.0] * 3, [2.0] * 3)
        out = C.paradox_table(design, curve, anchor=pd.Series(dtype=float))
        assert out["stylistic_similarity_anchor"].isna().all()
        assert out["anchor_source"].eq(
            f"absent:{C.PRESIDENT_EMBEDDINGS_PATH.name}"
        ).all()


class TestStylisticAnchor:
    def test_it_is_the_mean_cosine_to_every_other_president(self, tmp_path):
        path = tmp_path / "president_embeddings.parquet"
        pd.DataFrame({
            "president": ["A", "B", "C"],
            "e0": [1.0, 2.0, 0.0], "e1": [0.0, 0.0, 5.0],
        }).to_parquet(path, index=False)
        anchor = C._stylistic_anchor(path)
        # A and B are the same direction (cos 1); C is orthogonal to both.
        assert anchor["A"] == pytest.approx(0.5)
        assert anchor["B"] == pytest.approx(0.5)
        assert anchor["C"] == pytest.approx(0.0, abs=1e-12)

    def test_a_missing_artifact_yields_nothing_rather_than_a_magnitude(self, tmp_path):
        assert C._stylistic_anchor(tmp_path / "nope.parquet").empty


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


# --------------------------------------------------------------------------
# the trust gate, on EVERY output table (not just the window curve)
# --------------------------------------------------------------------------
#
# The layer's stated contract is "a machine-readable trust gate on every row a
# consumer could plot". Every table under `data/convergence/` has such rows:
# window rows, leave-one-president-out rows, permutation rows, president rows.
# The sweep below is driven by `C.OUTPUT_PATHS`, the module's own registry of
# output tables, so a NEW table added later without a `ci_status` fails this
# test without anyone having to remember to extend it.


def _trust_gate_violations(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Every way a published table can fail the trust-gate contract.

    Factored out so the live sweep and its mutation control run byte-identical
    logic — a guard checked with different code from the code it guards is not
    a guard.
    """
    problems = []
    for name, table in tables.items():
        if "ci_status" not in table.columns:
            problems.append(f"{name}: no ci_status column")
            continue
        if table["ci_status"].isna().any():
            problems.append(f"{name}: null ci_status on some rows")
        unknown = set(table["ci_status"].dropna()) - set(C.CI_STATUS_VALUES)
        if unknown:
            problems.append(f"{name}: ci_status values outside the vocabulary {unknown}")
    return problems


def _published_tables() -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(path) for name, path in C.OUTPUT_PATHS.items()
    }


class TestTrustGateOnEveryOutputTable:
    def test_every_output_table_carries_a_trust_gate(self):
        missing = [n for n, p in C.OUTPUT_PATHS.items() if not p.exists()]
        if missing:
            pytest.skip(f"data/convergence/ not generated in this tree ({missing})")
        assert _trust_gate_violations(_published_tables()) == []

    def test_the_registry_covers_every_parquet_actually_written(self):
        """The sweep is only future-proof if `OUTPUT_PATHS` is the whole truth.
        A table written to `data/convergence/` but absent from the registry
        would slip past the gate check entirely."""
        if not C.CONVERGENCE_DIR.exists():
            pytest.skip("data/convergence/ has not been generated in this tree")
        on_disk = {p.name for p in C.CONVERGENCE_DIR.glob("*.parquet")}
        registered = {p.name for p in C.OUTPUT_PATHS.values()}
        assert on_disk == registered

    @pytest.mark.parametrize(
        "damage,expected",
        [
            ("drop_column", "no ci_status column"),
            ("null_value", "null ci_status"),
            ("foreign_vocabulary", "outside the vocabulary"),
        ],
    )
    def test_that_the_sweep_would_catch_a_table_that_lost_its_gate(
        self, damage, expected
    ):
        """Mutate each defect back in and confirm the sweep fires. A guard blind
        to the shape it was written for is not a guard."""
        good = pd.DataFrame({"rho": [0.1, 0.2], "ci_status": ["ok", "ok"]})
        broken = good.drop(columns=["ci_status"]) if damage == "drop_column" else (
            good.assign(ci_status=["ok", None]) if damage == "null_value"
            else good.assign(ci_status=["ok", "fine_probably"])
        )
        assert _trust_gate_violations({"good": good}) == []
        problems = _trust_gate_violations({"broken": broken})
        assert len(problems) == 1 and expected in problems[0]

    def test_the_gate_vocabulary_is_literally_one_vocabulary(self):
        """`window_status`, `jackknife_status`, `null_status` and
        `paradox_table` must not each invent their own words."""
        produced = {C.window_status(n) for n in range(0, 8)}
        produced |= {C.jackknife_status(u, 100) for u in (0, 2, 3, 70, 80, 95, 100)}
        produced |= {
            C.null_status(rho, v, 100)
            for rho in (-0.5, float("nan"))
            for v in (0, 50, 90, 100)
        }
        assert produced == set(C.CI_STATUS_VALUES)


class TestJackknifeTrustGate:
    @pytest.mark.parametrize(
        "n_used,n_full,expected",
        [
            (0, 16, "no_data"),
            (2, 100, "no_data"),          # spearman_rho is NaN below 3 points
            (3, 0, "no_data"),            # a design with no trend windows at all
            (74, 100, "suppressed_n_floor"),
            (75, 100, "low_cluster_caution"),   # boundary: 0.75 is not < 0.75
            (89, 100, "low_cluster_caution"),
            (90, 100, "ok"),                    # boundary: 0.90 is not < 0.90
            (100, 100, "ok"),
        ],
    )
    def test_thresholds_are_two_sided_and_hand_derived(self, n_used, n_full, expected):
        assert C.jackknife_status(n_used, n_full) == expected

    def test_deleting_a_pivotal_president_is_flagged(self):
        """Three coextensive presidents sit exactly on the `MIN_PRESIDENTS`
        floor, so deleting ANY of them empties the trend entirely. A row whose
        rho is computed on a design that no longer resembles the full one must
        not be published as `ok`."""
        design = C.build_design(_coextensive_corpus(3), _TINY_ARM)
        full = C.n_windows_in_trend(design)
        assert full > 0
        for p in range(3):
            left = C.n_windows_in_trend(design, drop=p)
            assert left == 0
            assert C.jackknife_status(left, full) == "no_data"

    def test_deleting_a_redundant_president_is_not_flagged(self):
        """The counter to the fixture above: with six coextensive presidents the
        floor is nowhere near binding, so no deletion moves the window set and
        every row is `ok`. Without this the flagging test would pass on a gate
        hard-wired to a pessimistic answer."""
        design = C.build_design(_coextensive_corpus(6), _TINY_ARM)
        full = C.n_windows_in_trend(design)
        assert full > 0
        for p in range(6):
            left = C.n_windows_in_trend(design, drop=p)
            assert left == full
            assert C.jackknife_status(left, full) == "ok"

    def test_jackknife_frame_carries_the_gate_on_every_row(self):
        design = C.build_design(_coextensive_corpus(6), _TINY_ARM)
        frame = C.jackknife(design, 0, np.linspace(-1, 1, 21), n_draws=4)
        assert len(frame) == 6
        assert _trust_gate_violations({"jackknife": frame}) == []
        assert (frame["n_windows_full_design"] == C.n_windows_in_trend(design)).all()

    def test_n_windows_in_trend_reads_the_design_not_the_values(self):
        """It must agree with an actual curve's `used` mask, or the gate would be
        pricing a different window set from the one the rho was computed on."""
        design = C.build_design(_coextensive_corpus(4), _TINY_ARM)
        for drop in (None, 0, 2):
            curve = C.dispersion_curve(
                design, [4, drop or 0], drop=drop, n_draws=2,
                with_floor=False, with_entropy_match=False,
            )
            assert int(curve.used.sum()) == C.n_windows_in_trend(design, drop=drop)


class TestPermutationNullTrustGate:
    @pytest.mark.parametrize(
        "rho,n_valid,expected",
        [
            (-0.5, 100, "ok"),
            (-0.5, 99, "ok"),                    # boundary: 0.99 is not < 0.99
            (-0.5, 98, "low_cluster_caution"),
            (-0.5, 75, "low_cluster_caution"),   # boundary: 0.75 is not < 0.75
            (-0.5, 74, "suppressed_n_floor"),
            (-0.5, 0, "no_data"),
            (float("nan"), 100, "no_data"),
        ],
    )
    def test_thresholds(self, rho, n_valid, expected):
        assert C.null_status(rho, n_valid, 100) == expected

    def test_null_rows_carry_the_gate(self):
        observed = {("rolling", "all_windows", "dispersion"): -0.4}
        null = {("rolling", "all_windows", "dispersion"): np.linspace(-1, 1, 10)}
        rows = pd.DataFrame(C.null_rows(C.ARMS[0], observed, null, 10))
        assert _trust_gate_violations({"permutation_null": rows}) == []
        assert rows["ci_status"].iloc[0] == "ok"

    def test_a_null_that_mostly_evaporated_is_not_ok(self):
        draws = np.full(10, np.nan)
        draws[:4] = [-0.9, -0.5, 0.1, 0.4]
        observed = {("rolling", "all_windows", "dispersion"): -0.4}
        rows = pd.DataFrame(C.null_rows(
            C.ARMS[0], observed,
            {("rolling", "all_windows", "dispersion"): draws}, 10,
        ))
        assert rows["ci_status"].iloc[0] == "suppressed_n_floor"


# --------------------------------------------------------------------------
# build_convergence: the bail condition, the bypass gate, the whole pipeline
# --------------------------------------------------------------------------


@pytest.fixture
def tiny_pipeline(monkeypatch):
    """`build_convergence` wired to a 6-president synthetic corpus.

    Only three seams are replaced: the three-leg gate (minutes of Monte Carlo,
    exercised on its own below), the corpus loader, and the arm list. Everything
    else — designs, sampler, floor, entropy-match, permutation null, jackknife,
    paradox, decision table, meta — is the production code path.
    """
    comp = _coextensive_corpus(6, n_bins=5, distinct_agendas=True)
    monkeypatch.setattr(
        C, "_selftest", lambda verbose=True: {"passed": True, "stub": True}
    )
    monkeypatch.setattr(C, "build_compositions", lambda source: comp)
    monkeypatch.setattr(C, "ARMS", (
        C.ArmSpec("corex_all", "corex", None, 2, 2, "primary"),
        C.ArmSpec("corex_sotu", "corex", (C.SOTU_TYPE,), 2, 2, "co_primary"),
    ))
    return comp


class TestSelftestIsTheBailCondition:
    def test_a_failing_selftest_leaves_no_partial_artifact_behind(
        self, tmp_path, monkeypatch
    ):
        """The pre-registered STOP gate. Not merely "an exception is raised":
        the point is that a failed gate can never leave a half-written
        `data/convergence/` that a later reader mistakes for a real result."""
        out = tmp_path / "convergence"

        def _boom(verbose=True):
            raise AssertionError("_selftest FAILED on ['a_cluster_flat']. STOP")

        monkeypatch.setattr(C, "_selftest", _boom)
        monkeypatch.setattr(C, "build_compositions", _never_called)

        with pytest.raises(AssertionError, match="STOP"):
            C.build_convergence(out_dir=out, n_permutations=2, progress=False)

        assert not out.exists()

    def test_the_same_call_does_write_when_the_gate_passes(
        self, tmp_path, tiny_pipeline
    ):
        """The control for the test above. Without it, `not out.exists()` would
        also pass on a `build_convergence` that never writes anything at all."""
        out = tmp_path / "convergence"
        C.build_convergence(out_dir=out, n_permutations=2, progress=False)
        written = {p.name for p in out.iterdir()}
        assert written == {p.name for p in C.OUTPUT_PATHS.values()} | {
            C.META_PATH.name
        }

    def test_the_gate_runs_before_any_corpus_is_touched(self, tmp_path, monkeypatch):
        """Ordering, not just presence: the selftest must be the first thing
        that happens, so a failing gate costs nothing and risks nothing."""
        order = []
        monkeypatch.setattr(
            C, "_selftest",
            lambda verbose=True: (order.append("selftest"), {"passed": True})[1],
        )

        def _load(source):
            order.append("corpus")
            raise RuntimeError("stop here")

        monkeypatch.setattr(C, "build_compositions", _load)
        with pytest.raises(RuntimeError):
            C.build_convergence(out_dir=tmp_path / "o", n_permutations=1,
                                progress=False)
        assert order == ["selftest", "corpus"]


def _never_called(*args, **kwargs):
    raise AssertionError(
        "reached real computation after a failing selftest — the bail condition "
        "did not fire first"
    )


@pytest.fixture
def cheap_selftest(monkeypatch):
    """The real `_selftest`, shrunk to a size a unit test can afford.

    Only the SIZE of the Monte Carlo is reduced (2 corpora, 1 size replicate, 3
    permutations, 10 short presidencies) — every leg, every check and the raise
    itself are the production code. The pass thresholds are untouched by this
    fixture; individual tests move them to force a specific leg. Two corpora
    rather than one because the legs report a `ddof=1` sd, which is undefined
    on a single observation.
    """
    real = C._synthetic_composition

    def _small(rng, **kw):
        return real(rng, **{
            "n_presidents": 10, "speeches_first": 8, "speeches_last": 10,
            "paragraphs_per_speech": 8, **kw,
        })

    monkeypatch.setattr(C, "_synthetic_composition", _small)
    monkeypatch.setattr(C, "SELFTEST_CORPORA", 2)
    monkeypatch.setattr(C, "SELFTEST_REPLICATES", 1)
    monkeypatch.setattr(C, "SELFTEST_PERMUTATIONS", 3)


class TestSelftestItself:
    def test_a_failing_leg_raises_and_names_the_leg(self, monkeypatch, cheap_selftest):
        """`_selftest` is the STOP gate, so failure must be loud and specific —
        not a returned `passed: False` a caller could ignore."""
        monkeypatch.setattr(C, "SELFTEST_CLUSTER_FLAT_MAX_ABS_RHO", -1.0)
        with pytest.raises(AssertionError) as excinfo:
            C._selftest(verbose=False)
        message = str(excinfo.value)
        assert "a_cluster_flat" in message
        assert "STOP" in message

    def test_a_passing_run_returns_every_leg_and_its_verdict(
        self, monkeypatch, cheap_selftest
    ):
        for name, value in (
            ("SELFTEST_CLUSTER_FLAT_MAX_ABS_RHO", 1.0),
            ("SELFTEST_PARAGRAPH_BIAS_MAX_RHO", 1.0),
            ("SELFTEST_INJECTED_MAX_RHO", 1.0),
            ("SELFTEST_MAX_SIZE", 1.0),
        ):
            monkeypatch.setattr(C, name, value)
        results = C._selftest(verbose=False)
        assert results["passed"] is True
        assert set(results["checks"]) == {
            "a_cluster_flat", "a_paragraph_contrast", "b_detects_injected",
            "c_size_at_nominal_5pct",
        }

    def test_it_publishes_no_window_count_that_belongs_to_one_corpus(
        self, monkeypatch, cheap_selftest
    ):
        """`n_windows` was read off the LAST leg-(a) corpus after the loop had
        ended and published as if it described the selftest. The corpora differ
        from each other and leg (c) uses a shorter timeline entirely."""
        for name in ("SELFTEST_CLUSTER_FLAT_MAX_ABS_RHO",
                     "SELFTEST_PARAGRAPH_BIAS_MAX_RHO",
                     "SELFTEST_INJECTED_MAX_RHO", "SELFTEST_MAX_SIZE"):
            monkeypatch.setattr(C, name, 1.0)
        results = C._selftest(verbose=False)
        assert "n_windows" not in results
        assert results["a_corpus_windows_in_trend_min"] > 0
        assert (results["a_corpus_windows_in_trend_max"]
                >= results["a_corpus_windows_in_trend_min"])

    @pytest.mark.parametrize("alpha,expected_size", [(1.0, 1.0), (0.0, 0.0)])
    def test_leg_c_counts_rejections_against_alpha(
        self, monkeypatch, cheap_selftest, alpha, expected_size
    ):
        """Leg (c) is the measured type-I error, and the whole point is that it
        is COUNTED rather than assumed. Driving alpha to its two extremes makes
        the count move with it: a hard-wired size would survive one of these."""
        monkeypatch.setattr(C, "ALPHA", alpha)
        monkeypatch.setattr(C, "SELFTEST_MAX_SIZE", 1.0)
        for name in ("SELFTEST_CLUSTER_FLAT_MAX_ABS_RHO",
                     "SELFTEST_PARAGRAPH_BIAS_MAX_RHO", "SELFTEST_INJECTED_MAX_RHO"):
            monkeypatch.setattr(C, name, 1.0)
        results = C._selftest(verbose=False)
        assert results["c_permutation_size"] == expected_size
        assert results["c_replicates"] == C.SELFTEST_REPLICATES

    def test_verbose_mode_prints_a_verdict_for_every_leg(
        self, monkeypatch, cheap_selftest, capsys
    ):
        """The printed block is an operator's only view of the gate. A leg whose
        verdict is not printed is a leg nobody checks."""
        for name in ("SELFTEST_CLUSTER_FLAT_MAX_ABS_RHO",
                     "SELFTEST_PARAGRAPH_BIAS_MAX_RHO",
                     "SELFTEST_INJECTED_MAX_RHO", "SELFTEST_MAX_SIZE"):
            monkeypatch.setattr(C, name, 1.0)
        C._selftest(verbose=True)
        printed = capsys.readouterr().out
        assert printed.count("PASS") == 5      # four legs plus the overall line
        assert "FAIL" not in printed
        for marker in ("(a) clustered null", "(a) same corpora", "(b) injected",
                       "(c) permutation-null size", "overall"):
            assert marker in printed

    def test_the_published_meta_carries_no_mislabelled_window_count(self):
        if not C.META_PATH.exists():
            pytest.skip("data/convergence/ has not been generated in this tree")
        selftest = json.loads(C.META_PATH.read_text())["selftest"]
        assert "n_windows" not in selftest
        assert "a_corpus_windows_in_trend_min" in selftest


class TestSelftestBypassIsGated:
    def test_the_old_boolean_bypass_no_longer_exists(self):
        """`run_selftest=False` was an ungated public bypass of the pre-registered
        bail condition. It must not be silently accepted."""
        with pytest.raises(TypeError):
            C.build_convergence(run_selftest=False)

    def test_a_truthy_value_is_not_a_bypass(self, monkeypatch, tmp_path):
        monkeypatch.setattr(C, "_selftest", _never_called)
        for wrong in (True, "", "yes", "SELFTEST_BYPASS_TOKEN"):
            with pytest.raises(ValueError, match="verbatim"):
                C.build_convergence(
                    out_dir=tmp_path / "o", selftest_bypass=wrong, progress=False
                )

    def test_a_bypassed_run_may_not_write_the_published_layer(self, monkeypatch):
        monkeypatch.setattr(C, "_selftest", _never_called)
        monkeypatch.setattr(C, "build_compositions", _never_called)
        for out_dir in (None, C.CONVERGENCE_DIR):
            with pytest.raises(ValueError, match="published layer"):
                C.build_convergence(
                    out_dir=out_dir,
                    selftest_bypass=C.SELFTEST_BYPASS_TOKEN,
                    progress=False,
                )

    def test_a_bypassed_run_declares_itself_unpublishable_in_its_meta(
        self, tmp_path, tiny_pipeline, monkeypatch
    ):
        monkeypatch.setattr(C, "_selftest", _never_called)
        out = tmp_path / "scratch"
        C.build_convergence(
            out_dir=out, n_permutations=2, progress=False,
            selftest_bypass=C.SELFTEST_BYPASS_TOKEN,
        )
        selftest = json.loads((out / C.META_PATH.name).read_text())["selftest"]
        assert selftest["ran"] is False
        assert selftest["publishable"] is False


class TestBuildConvergence:
    def test_it_writes_every_registered_table_plus_the_meta(
        self, tmp_path, tiny_pipeline
    ):
        out = tmp_path / "convergence"
        tables = C.build_convergence(out_dir=out, n_permutations=3, progress=False)
        assert set(tables) == set(C.OUTPUT_PATHS)
        for name, path in C.OUTPUT_PATHS.items():
            written = pd.read_parquet(out / path.name)
            assert len(written) > 0
            assert len(written) == len(tables[name])

    def test_every_table_it_writes_carries_the_trust_gate(
        self, tmp_path, tiny_pipeline
    ):
        """The end-to-end counterpart of the artifact sweep: the contract holds
        for a freshly built layer, not only for the committed one."""
        out = tmp_path / "convergence"
        C.build_convergence(out_dir=out, n_permutations=3, progress=False)
        tables = {
            name: pd.read_parquet(out / path.name)
            for name, path in C.OUTPUT_PATHS.items()
        }
        assert _trust_gate_violations(tables) == []

    def test_the_meta_it_writes_has_no_wall_clock_stamp(self, tmp_path, tiny_pipeline):
        out = tmp_path / "convergence"
        C.build_convergence(out_dir=out, n_permutations=3, progress=False)
        text = (out / C.META_PATH.name).read_text()
        for banned in ("generated_at", "timestamp", "run_date", "today"):
            assert banned not in text
        meta = json.loads(text)
        assert meta["api_calls"] == 0
        assert meta["decision"]["cell"] in C.HEADLINES

    def test_two_runs_of_the_same_build_are_byte_identical(
        self, tmp_path, tiny_pipeline
    ):
        """The layer's determinism contract, proved on the real write path: a
        dirty `git status data/convergence/` must mean the numbers moved."""
        first, second = tmp_path / "a", tmp_path / "b"
        C.build_convergence(out_dir=first, n_permutations=3, progress=False)
        C.build_convergence(out_dir=second, n_permutations=3, progress=False)
        names = [p.name for p in C.OUTPUT_PATHS.values()] + [C.META_PATH.name]
        for name in names:
            assert (first / name).read_bytes() == (second / name).read_bytes(), name

    def test_the_jackknife_covers_every_ever_eligible_president_in_every_arm(
        self, tmp_path, tiny_pipeline
    ):
        out = tmp_path / "convergence"
        tables = C.build_convergence(out_dir=out, n_permutations=3, progress=False)
        jack = tables["jackknife"]
        assert set(jack["arm"]) == {a.name for a in C.ARMS}
        for arm in C.ARMS:
            rows = jack[jack["arm"] == arm.name]
            assert set(rows["left_out_president"]) == set(tiny_pipeline.president)
        assert not jack["scipy_pvalue_used"].any()

    def test_progress_output_names_every_arm_it_is_working_on(
        self, tmp_path, tiny_pipeline, capsys
    ):
        C.build_convergence(
            out_dir=tmp_path / "convergence", n_permutations=2, progress=True
        )
        printed = capsys.readouterr().out
        for arm in C.ARMS:
            assert f"arm {arm.name} ({arm.role})" in printed
        assert "permutation 0/2" in printed

    def test_the_paradox_table_is_built_for_the_primary_arm_only(
        self, tmp_path, tiny_pipeline
    ):
        out = tmp_path / "convergence"
        tables = C.build_convergence(out_dir=out, n_permutations=3, progress=False)
        assert set(tables["paradox"]["arm"]) == {"corex_all"}

    def test_the_rolling_binding_survives_a_reordered_window_kind_tuple(
        self, tmp_path, tiny_pipeline, monkeypatch
    ):
        """`rolling_curve` / `rolling_design` used to leak out of the `for kind`
        loop, which made the post-loop jackknife and paradox depend on how
        `WINDOW_KINDS` happened to be ordered. Reordering it must change nothing
        structural."""
        monkeypatch.setattr(C, "WINDOW_KINDS", ("nonoverlapping", "rolling"))
        out = tmp_path / "convergence"
        tables = C.build_convergence(out_dir=out, n_permutations=2, progress=False)
        # The jackknife is a ROLLING-grid statistic: it must be scored on the
        # rolling design's ~16 windows, not the 3-window non-overlapping grid.
        rolling = tables["dispersion_curves"]
        n_rolling = int(
            rolling[(rolling["arm"] == "corex_all")
                    & (rolling["window_kind"] == "rolling")]["used_in_trend"].sum()
        )
        jack = tables["jackknife"]
        assert (jack[jack["arm"] == "corex_all"]["n_windows_full_design"]
                == n_rolling).all()
        assert n_rolling > 3


# --------------------------------------------------------------------------
# the command line
# --------------------------------------------------------------------------


class TestCommandLine:
    def test_selftest_only_runs_the_gate_and_stops(self, monkeypatch):
        """`--selftest-only` must never reach the real build — it is the cheap
        way to check the gate before committing to a full run."""
        calls = []
        monkeypatch.setattr(
            C, "_selftest",
            lambda verbose=True: (calls.append(verbose), {"passed": True})[1],
        )
        monkeypatch.setattr(C, "build_convergence", _never_called)
        C.main(["--selftest-only", "--quiet"])
        assert calls == [False]

    def test_a_failing_gate_stops_the_cli_too(self, monkeypatch):
        def _boom(verbose=True):
            raise AssertionError("_selftest FAILED. STOP")

        monkeypatch.setattr(C, "_selftest", _boom)
        monkeypatch.setattr(C, "build_convergence", _never_called)
        with pytest.raises(AssertionError, match="STOP"):
            C.main(["--selftest-only", "--quiet"])

    def test_the_default_invocation_builds_with_the_preregistered_r(
        self, monkeypatch
    ):
        seen = {}

        def _build(**kwargs):
            seen.update(kwargs)
            return {"permutation_null": pd.DataFrame()}

        monkeypatch.setattr(C, "build_convergence", _build)
        monkeypatch.setattr(C, "score_decision_table", _never_called)
        C.main(["--quiet"])
        assert seen["n_permutations"] == C.N_PERMUTATIONS
        assert seen["progress"] is False
        assert "selftest_bypass" not in seen

    def test_a_loud_run_prints_the_scored_decision_cell_and_its_headline(
        self, monkeypatch, capsys
    ):
        """The headline sentence is pre-committed in the prereg, and the CLI is
        where a human first reads which of the four cells fired."""
        null_table = _null_table(**{
            "corex_all.dispersion": True, "corex_sotu.dispersion": True,
            "corex_all.floor_corrected": True, "corex_sotu.floor_corrected": True,
            "corex_all.excess_ratio": True, "corex_sotu.excess_ratio": True,
        }).assign(rho=-0.7, p_one_sided=0.001, null_crit_05=-0.4)
        monkeypatch.setattr(
            C, "build_convergence",
            lambda **kw: {"permutation_null": null_table},
        )
        C.main(["--permutations", "10"])
        printed = capsys.readouterr().out
        assert "decision cell: convergence" in printed
        assert C.HEADLINES["convergence"] in printed


# --------------------------------------------------------------------------
# no local may leak out of a loop into the code that follows it
# --------------------------------------------------------------------------


def _loop_leaked_names(func: ast.FunctionDef) -> list[str]:
    """Names READ at a function's top statement level but bound ONLY inside one
    of its loops.

    That is the shape of the `rolling_curve` / `rolling_design` defect: correct
    today purely because `"rolling"` happens to be in `WINDOW_KINDS`, and a
    `NameError` at a distance — or, worse, a silent reuse of the PREVIOUS arm's
    design — the moment that stops being true.
    """
    bound_at_top = {a.arg for a in func.args.args} | {
        a.arg for a in func.args.kwonlyargs
    }
    bound_in_loop: set[str] = set()
    read_at_top: set[str] = set()
    for stmt in func.body:
        target = bound_in_loop if isinstance(stmt, (ast.For, ast.While)) else bound_at_top
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                target.add(node.id)
        if not isinstance(stmt, (ast.For, ast.While)):
            for node in ast.walk(stmt):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    read_at_top.add(node.id)
    # No allow-list of module globals or builtins: a name that is STORED inside
    # the loop is a local of this function regardless of what else shares its
    # spelling, so subtracting those sets would only hide real leaks.
    return sorted((read_at_top & bound_in_loop) - bound_at_top)


def _function_defs(source: str) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
    }


class TestNoLoopVariableLeaks:
    def test_no_function_in_the_module_reads_a_loop_only_local(self):
        for name, func in _function_defs(inspect.getsource(C)).items():
            assert _loop_leaked_names(func) == [], name

    def test_that_the_scan_would_catch_the_defect_it_was_written_for(self):
        """The exact pre-fix spelling of `build_convergence`'s loop."""
        defect = _function_defs(
            "def build(designs):\n"
            "    for kind, design in designs.items():\n"
            "        curve = compute(design)\n"
            "        if kind == 'rolling':\n"
            "            rolling_curve, rolling_design = curve, design\n"
            "    return summarize(rolling_curve, rolling_design)\n"
        )["build"]
        assert _loop_leaked_names(defect) == ["rolling_curve", "rolling_design"]

    def test_the_scan_does_not_fire_on_the_explicit_binding(self):
        fixed = _function_defs(
            "def build(designs):\n"
            "    curves = {}\n"
            "    for kind, design in designs.items():\n"
            "        curves[kind] = compute(design)\n"
            "    rolling_curve = curves['rolling']\n"
            "    rolling_design = designs['rolling']\n"
            "    return summarize(rolling_curve, rolling_design)\n"
        )["build"]
        assert _loop_leaked_names(fixed) == []


# --------------------------------------------------------------------------
# the rival hypothesis: "more ALIKE" vs "each one got BROADER"
# --------------------------------------------------------------------------
#
# A declining dispersion curve has a mundane rival explanation the original
# design could not distinguish: presidents need not have converged on a shared
# agenda, they may each simply have started talking about everything. The
# entropy-matched null is the separator, and the two corpora below are its
# behavioural test — one of each hypothesis, built so that the ANSWER is known
# in advance and the machinery has to recover it.


_RIVAL_PRESIDENTS = 20


def _broader_corpus() -> C.Composition:
    """Every president BROADENS; nobody becomes more aligned with anybody.

    President `p` owns bin `p` — twenty presidents, twenty private bins, so
    there is no shared agenda at any point in time and no trend in alignment.
    What changes is only breadth: president `p` puts a fraction `b_p` of their
    mass on a flat background over all 40 bins, and `b_p` rises 0 -> 0.95 with
    time. Dispersion must fall (broad distributions overlap), but every bit of
    that fall is explained by breadth, so `excess_ratio` must stay at 1.
    """
    n_bins = 40
    specs = []
    for p in range(_RIVAL_PRESIDENTS):
        b = 0.95 * p / (_RIVAL_PRESIDENTS - 1)
        w = np.full(n_bins, b / n_bins)
        w[p] += 1 - b
        start = 1800 + p * 6
        specs.append((f"P{p:02d}", [start + y % 6 for y in range(10)], w / w.sum()))
    return _hand_corpus(specs, n_bins, seed=0)


def _alike_corpus() -> C.Composition:
    """Every president keeps the SAME breadth; they converge on a shared agenda.

    Everyone spreads their mass evenly over exactly three of twelve bins, so
    entropy is log2(3) from first president to last and no broadening is
    available as an explanation. What changes is WHICH three: the share drawn
    from the common core {bin0, bin1, bin2} rises 0 -> 3 with time. Dispersion
    must fall AND `excess_ratio` must fall with it.
    """
    n_bins, core = 12, [0, 1, 2]
    others = [b for b in range(n_bins) if b not in core]
    rng = np.random.default_rng(11)
    specs = []
    for p in range(_RIVAL_PRESIDENTS):
        n_core = int(round(3 * p / (_RIVAL_PRESIDENTS - 1)))
        support = core[:n_core] + list(
            rng.choice(others, size=3 - n_core, replace=False)
        )
        w = np.zeros(n_bins)
        w[support] = 1 / 3
        start = 1800 + p * 6
        specs.append((f"P{p:02d}", [start + y % 6 for y in range(10)], w))
    return _hand_corpus(specs, n_bins, seed=0)


def _rival_curve(comp: C.Composition) -> C.Curve:
    return C.dispersion_curve(
        C._selftest_design(comp), [5, 2, 3], with_floor=False
    )


def _rho(curve: C.Curve, values: np.ndarray) -> float:
    return C.spearman_rho(curve.center[curve.used], values[curve.used])


class TestRivalHypothesisSeparator:
    """The most load-bearing behavioural claim in the module: that
    `excess_ratio` reads "broader" and "more alike" differently."""

    def test_the_broader_corpus_really_does_broaden(self):
        curve = _rival_curve(_broader_corpus())
        used = curve.used
        assert used.sum() >= 40
        assert _rho(curve, curve.mean_entropy) >= 0.9
        entropy = curve.mean_entropy[used]
        assert entropy[-1] > 4 * entropy[0]

    def test_broadening_alone_drives_dispersion_down(self):
        """Without this the flat-excess_ratio assertion would be vacuous: there
        would be no decline for the rival null to have to explain away."""
        curve = _rival_curve(_broader_corpus())
        assert _rho(curve, curve.dispersion) <= -0.9
        disp = curve.dispersion[curve.used]
        assert disp[-1] < 0.65 * disp[0]

    def test_broadening_alone_leaves_excess_ratio_flat_at_one(self):
        """THE rival-null test. Every window's observed dispersion is within
        10% of what the presidents' own breadths already predict, so the design
        correctly refuses to call this corpus a convergence."""
        curve = _rival_curve(_broader_corpus())
        ratio = curve.excess_ratio[curve.used]
        assert np.isfinite(ratio).all()
        assert ratio.min() >= 0.9
        assert ratio.max() <= 1.1

    def test_the_alike_corpus_holds_breadth_fixed(self):
        curve = _rival_curve(_alike_corpus())
        entropy = curve.mean_entropy[curve.used]
        assert entropy.max() / entropy.min() <= 1.05

    def test_real_alignment_drives_excess_ratio_down_too(self):
        """The counter that proves `excess_ratio` is not simply pinned at 1 by
        construction: on a corpus that DID converge, with breadth held fixed,
        the same statistic collapses."""
        curve = _rival_curve(_alike_corpus())
        assert _rho(curve, curve.dispersion) <= -0.9
        assert _rho(curve, curve.excess_ratio) <= -0.8
        assert curve.excess_ratio[curve.used][-1] <= 0.3

    def test_the_separator_actually_separates(self):
        """The two corpora are indistinguishable on `dispersion` — both decline
        hard — and are told apart only by `excess_ratio`. That is the whole
        reason the entropy-matched null is mandatory."""
        broader, alike = _rival_curve(_broader_corpus()), _rival_curve(_alike_corpus())
        assert _rho(broader, broader.dispersion) <= -0.9
        assert _rho(alike, alike.dispersion) <= -0.9
        assert broader.excess_ratio[broader.used].min() > 3 * (
            alike.excess_ratio[alike.used].min()
        )

    def test_the_decision_table_reads_the_two_corpora_differently(self):
        """End-to-end: the same evidence pattern each corpus produces routes to
        the pre-committed cell it should."""
        broadening = C.score_decision_table(_null_table(**{
            "corex_all.dispersion": True, "corex_sotu.dispersion": True,
            "corex_all.floor_corrected": True, "corex_sotu.floor_corrected": True,
        }))
        convergence = C.score_decision_table(_null_table(**{
            "corex_all.dispersion": True, "corex_sotu.dispersion": True,
            "corex_all.floor_corrected": True, "corex_sotu.floor_corrected": True,
            "corex_all.excess_ratio": True, "corex_sotu.excess_ratio": True,
        }))
        assert broadening["cell"] == "broadening"
        assert convergence["cell"] == "convergence"
        assert broadening["headline"] != convergence["headline"]
