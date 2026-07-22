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


def _per_speech_corpus(
    bin_of,
    n_presidents: int,
    n_bins: int,
    *,
    paras_per_speech: int = 6,
    speeches_per_president: int = 10,
    first_year: int = 1800,
    year_step: int = 6,
    president_step: int = 0,
    year_cycle: int | None = None,
) -> C.Composition:
    """A corpus whose label is dictated for every single paragraph.

    `bin_of(p, s, i)` gives the bin index of paragraph `i` of president `p`'s
    speech `s`, or `None` for an UNLABELLED paragraph. `_hand_corpus` draws a
    president's paragraphs i.i.d. from one weight vector, which cannot express
    "this SPEECH is internally homogeneous" — and speech-level structure is
    precisely what the cluster rarefaction and the speech-block floor exist to
    respect, so the tests that discriminate them need this instead.

    `president_step = 0` makes every president speak in the same years, so
    eligibility is uniform in every window and a test can talk about what was
    said rather than about who was in the room.
    """
    cycle = year_cycle or speeches_per_president
    docs, idxs, pres, years, labels = [], [], [], [], []
    for p in range(n_presidents):
        for s in range(speeches_per_president):
            doc = f"P{p:02d}-s{s:04d}"
            for i in range(paras_per_speech):
                b = bin_of(p, s, i)
                docs.append(doc)
                idxs.append(i)
                pres.append(f"P{p:02d}")
                years.append(first_year + president_step * p + year_step * (s % cycle))
                labels.append([] if b is None else [int(b)])
    bins = [f"bin{j}" for j in range(n_bins)] + [C.NO_TOPIC]
    return C.Composition(
        label_source="synthetic",
        bins=bins,
        mass=C._mass_from_label_lists(labels, len(bins)),
        doc_name=np.array(docs),
        para_idx=np.array(idxs),
        president=np.array(pres),
        year=np.array(years, dtype=np.int64),
        speech_type=np.array([C.SOTU_TYPE] * len(docs)),
        n_label_assignments=sum(len(ls) for ls in labels),
    )


def _internally_pure_speeches(n_presidents: int = 3, n_bins: int = 4) -> C.Composition:
    """Every SPEECH sits entirely in one bin, and consecutive speeches differ.

    Under cluster rarefaction with S=1 a draw is one whole speech, so its
    composition can only ever be a PURE CORNER. Under paragraph rarefaction the
    same pool yields mixtures, because it never sees the speech boundary.
    """
    return _per_speech_corpus(
        lambda p, s, i: (p + s) % n_bins, n_presidents, n_bins
    )


def _mutually_disjoint_pure_speeches(n_presidents: int = 2) -> C.Composition:
    """President `p`'s every speech is entirely bin `p`.

    Internally homogeneous AND mutually disjoint: observed dispersion is exactly
    1 bit, so anything the speech-block floor does to it is visible.
    """
    return _per_speech_corpus(
        lambda p, s, i: p, n_presidents, n_presidents + 2
    )


# A one-speech arm: S=1 makes "a draw is a whole speech" an observable property
# rather than a statistical one.
_ONE_SPEECH_ARM = C.ArmSpec("one_speech", "corex", None, 1, 6, "test")

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

    def test_a_repeated_label_still_leaves_the_paragraph_weighing_one(self):
        """The equation ACCUMULATES into a bin rather than assigning to it.

        De-duplication upstream means a repeated label should never reach here,
        so this is the belt to that braces — and it is the difference between a
        paragraph weighing 1 and weighing 1/k. Overwriting instead of
        accumulating breaks mass conservation the instant the dedup contract
        does, which is the failure mode that would be hardest to notice."""
        mass = C._mass_from_label_lists([[0, 0]], 3)
        assert mass[0].tolist() == [1.0, 0.0, 0.0]
        assert mass.sum() == pytest.approx(1.0)

    def test_unlabelled_paragraph_lands_in_the_no_topic_bin(self):
        mass = C._mass_from_label_lists([[]], 4)
        assert mass[0, 3] == pytest.approx(1.0)
        assert mass[0, :3].sum() == 0.0

    def test_every_paragraph_contributes_the_same_total_mass_regardless_of_label_count(
        self
    ):
        """The 1/k split, stated as what it actually buys.

        The corpus's labels-per-paragraph drifts 1.49 -> 0.83, and an unnormalized
        count would let a paragraph carrying 3 labels outvote one carrying 1.
        This test was previously named `test_label_density_alone_cannot_move_the
        _measure`, which its own fixture contradicts: `[0.5, 0.5, 0]` and
        `[1/3, 1/3, 1/3]` are DIFFERENT distributions with the same sum, and a
        JSD between them is not zero. Mass conservation is the guarantee; agenda
        invariance is not. See the boundary test below.
        """
        sparse = C._mass_from_label_lists([[0], [1]], 3).mean(axis=0)
        dense = C._mass_from_label_lists([[0, 1, 2], [0, 1, 2]], 3).mean(axis=0)
        assert sparse.sum() == pytest.approx(1.0)
        assert dense.sum() == pytest.approx(1.0)
        # The row-level statement, which is the one the equation makes: every
        # paragraph weighs 1, whatever k is.
        rows = C._mass_from_label_lists([[0], [0, 1], [0, 1, 2], []], 4)
        assert rows.sum(axis=1) == pytest.approx(np.ones(4))
        # ... and the two composition VECTORS above are nonetheless far apart,
        # which is exactly why mass conservation is not agenda invariance.
        assert C.pairwise_jsd(np.array([[sparse, dense]]))[0, 0, 1] > 0.05


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
        """722 duplicate `(paragraph, topic)` pairs survive normalization on the
        real corpus, and CLAUDE.md requires the POST-dedup figure to be the one
        quoted. The second paragraph carries two distinct topics so that
        `n_label_assignments` (3) cannot be confused with the paragraph count
        (2) — with one label per paragraph the two are equal and the assertion
        proves nothing."""
        issues = _issue_frame(1, per_doc=2)
        taxonomy = {
            "level2": [{"name": "Trade", "level1": "Economy"},
                       {"name": "Tariffs", "level1": "Economy"}],
        }
        ann = pd.DataFrame({
            "doc_name": ["doc0", "doc0"], "para_idx": [0, 1],
            "topics": [["Trade", "trade"], ["Trade", "Tariffs"]],
        })
        comp = C.build_compositions(
            "llm", issues=issues, annotations=ann,
            speech_annotations=_speech_frame(1), taxonomy=taxonomy,
        )
        assert comp.n_label_assignments == 3        # not 4 raw, and not 2 paragraphs
        assert comp.mass[0].tolist() == [0.0, 1.0, 0.0]     # Tariffs, Trade, no-topic
        assert comp.mass[1].tolist() == [0.5, 0.5, 0.0]

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

    def test_a_two_president_window_is_suppressed_in_every_window(self):
        """The test above is only as good as its fixture: on a corpus whose
        eligible counts never actually hit 2 it asserts nothing. Two coextensive
        presidents put EVERY window on `suppressed_n_floor`, so the
        `MIN_PRESIDENTS` floor is exercised rather than assumed. A single pair is
        not a dispersion; prereg 4.4 keeps it out of every trend.
        """
        design = C.build_design(_coextensive_corpus(2), _TINY_ARM)
        curve = C.dispersion_curve(
            design, [1, 1], n_draws=2, with_floor=False, with_entropy_match=False
        )
        assert set(curve.status) == {"suppressed_n_floor"}
        assert not curve.used.any()
        assert np.isnan(curve.dispersion).all()
        assert C.n_windows_in_trend(design) == 0

    def test_a_three_president_window_does_enter_the_trend(self):
        """The control. Without it the test above would also pass on a
        `dispersion_curve` that never uses any window at all."""
        design = C.build_design(_coextensive_corpus(3), _TINY_ARM)
        curve = C.dispersion_curve(
            design, [1, 1], n_draws=2, with_floor=False, with_entropy_match=False
        )
        assert set(curve.status) == {"low_cluster_caution"}
        assert curve.used.all()
        assert C.n_windows_in_trend(design) == len(design.windows)


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

# Every scipy.stats correlation function that returns a `(statistic, pvalue)`
# pair. The ban is written against `spearmanr` because that is the estimator
# this module uses, but the DEFECT is "reading element 1 of a scipy correlation
# result on 93%-overlapping windows" — swapping in `pearsonr` would reproduce it
# exactly, so the scan covers the family rather than the one live name.
_SPEARMAN_FUNCTIONS = {"spearmanr", "pearsonr", "kendalltau", "weightedtau"}


def _is_spearman_call(node: ast.AST) -> bool:
    if isinstance(node, ast.NamedExpr):
        # `(r := spearmanr(a, b))[1]` subscripts the walrus, not the call.
        node = node.value
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

    # Names anywhere in the source that hold a spearmanr result. `NamedExpr` is
    # the walrus (`if (r := spearmanr(a, b))[1] < .05`), which binds exactly the
    # same way and would otherwise be invisible.
    bound: set[str] = set()
    for node in ast.walk(tree):
        value = getattr(node, "value", None)
        if not isinstance(
            node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)
        ) or not _is_spearman_call(value):
            continue
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
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
            ("walrus binding, then subscript",
             "from scipy.stats import spearmanr\nif (r := spearmanr(a, b))[1] < 0.05:\n    pass"),
            ("walrus binding, then attribute",
             "from scipy.stats import spearmanr\nif (r := spearmanr(a, b)).pvalue < 0.05:\n    pass"),
            ("a different scipy correlation, same defect",
             "from scipy.stats import pearsonr\np = pearsonr(a, b)[1]"),
            ("kendalltau attribute",
             "from scipy.stats import kendalltau\np = kendalltau(a, b).pvalue"),
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
    def test_a_cluster_draw_is_a_whole_speech_not_loose_paragraphs(self):
        """THE property this whole task exists to protect.

        Paragraph rarefaction on speech-clustered data returns rho = -0.605 from
        pure noise; cluster rarefaction is the fix. The previous version of this
        test asserted only `c.shape` and `c.sum(axis=-1) == 1`, both of which a
        mean of rows that each sum to 1 satisfies at ANY S and B — so it passed
        verbatim against `draw_compositions_paragraph`, the broken estimator.

        Here every speech is internally homogeneous and S=1, so a cluster draw
        is one whole speech and its composition MUST be a pure corner. A sampler
        that pools the paragraphs first cannot produce that: it mixes the two
        bins a president's speeches sit in.
        """
        design = C.build_design(_internally_pure_speeches(), _ONE_SPEECH_ARM)
        window = next(w for w in design.windows if len(w.presidents) >= 2)
        c = C.draw_compositions(
            design, list(window.pools), np.random.default_rng(0), 200
        )
        assert c.shape == (200, len(window.pools), design.n_bins)
        assert np.allclose(c.sum(axis=-1), 1.0)
        assert np.isclose(c.max(axis=-1), 1.0).all(), (
            "a draw mixed two speeches: the sampler is not cluster-rarefying"
        )

    def test_the_paragraph_rarefied_contrast_really_does_mix_speeches(self):
        """The counter that keeps the test above from being vacuous: on the same
        pools the superseded sampler produces mixtures almost every time (0.5%
        of its 600 slot-draws come out pure, against 100% for the cluster
        sampler). Without this, `max == 1` might be a property of the fixture."""
        design = C.build_design(_internally_pure_speeches(), _ONE_SPEECH_ARM)
        window = next(w for w in design.windows if len(w.presidents) >= 2)
        c = C.draw_compositions_paragraph(
            design, list(window.pools), np.random.default_rng(0), 200
        )
        assert np.isclose(c.max(axis=-1), 1.0).mean() < 0.05

    def test_a_cluster_draw_averages_exactly_S_times_B_paragraph_rows(self):
        """m = S*B, pinned from both sides.

        Paragraph `i` of every speech sits in its own bin, so a draw's masses
        are `count / m`. Integrality of `c * m` pins the denominator to a
        DIVISOR of m (an S or B larger than the design's would break it), and
        the presence of an exact `1/m` entry pins it to m itself (a smaller
        denominator can never produce that value).
        """
        design = C.build_design(
            _per_speech_corpus(lambda p, s, i: i, 3, 8, paras_per_speech=8),
            C.ArmSpec("m", "corex", None, 2, 3, "test"),
        )
        window = next(w for w in design.windows if len(w.presidents) >= 2)
        c = C.draw_compositions(
            design, list(window.pools), np.random.default_rng(1), 50
        )
        m = design.n_speeches * design.n_paragraphs
        scaled = c * m
        assert np.allclose(scaled, np.round(scaled)), "denominator is not a divisor of m"
        assert np.isclose(scaled, 1.0).any(), "no draw ever weighted one row at 1/m"

    def test_dispersion_curve_DEFAULTS_to_the_cluster_sampler(self):
        """The default of `dispersion_curve(rarefaction=...)` decides which
        estimator every published curve was computed with.

        Flipping it to `"paragraph"` ships the rho = -0.605 sampler — the exact
        thing this task exists to avoid — into `data/convergence/`, and until
        now that mutant was killed by ONE test in the whole suite, living in the
        selftest class where sibling tests monkeypatch the threshold. This is
        the same claim asserted directly at the call site.

        On internally-pure speeches with S=1 the two samplers disagree by
        construction (a cluster draw is a whole speech, so a pure corner; the
        paragraph sampler mixes the bins a president's speeches sit in), so the
        contrast leg is what keeps the equality from being vacuous.
        """
        design = C.build_design(_internally_pure_speeches(), _ONE_SPEECH_ARM)
        kwargs = dict(n_draws=64, with_floor=False, with_entropy_match=False)
        default = C.dispersion_curve(design, [11, 1], **kwargs)
        cluster = C.dispersion_curve(design, [11, 1], rarefaction="cluster", **kwargs)
        paragraph = C.dispersion_curve(design, [11, 1], rarefaction="paragraph", **kwargs)
        used = default.used
        assert used.any(), "the fixture produced no trend window"
        # the fixture HAS the property: the two samplers genuinely differ here
        assert not np.allclose(
            cluster.dispersion[used], paragraph.dispersion[used], atol=0.05
        ), "this fixture cannot tell the two samplers apart"
        assert np.array_equal(default.dispersion[used], cluster.dispersion[used])

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

    def _disjoint_floor_design(self) -> tuple[C.Design, list[np.ndarray]]:
        design = C.build_design(
            _mutually_disjoint_pure_speeches(), _ONE_SPEECH_ARM
        )
        window = next(w for w in design.windows if len(w.presidents) >= 2)
        return design, list(window.pools)

    def test_the_floor_actually_redeals_speeches_between_the_presidents(self):
        """`0 <= floor <= 1` — what this test used to assert — is guaranteed by
        the `np.clip` at the end of `pairwise_jsd`, so it could not fail. This
        one can: two presidents whose speeches are pure and mutually disjoint
        have an OBSERVED dispersion of exactly 1 bit, and the floor is the
        dispersion they would show with no agendas at all. A floor that forgot
        to re-deal would return that same 1.0.
        """
        design, pools = self._disjoint_floor_design()
        observed = C._mean_offdiag(
            C.pairwise_jsd(
                C.draw_compositions(design, pools, np.random.default_rng(0), 100)
            )
        ).mean()
        assert observed == pytest.approx(1.0)
        value, per_slot = C.speech_block_floor(
            design, pools, np.random.default_rng(0), 200
        )
        assert len(per_slot) == len(pools)
        assert value < 0.8, "the floor did not pool the two presidents' speeches"

    def test_the_floor_deals_WHOLE_SPEECHES_not_loose_paragraphs(self):
        """The defect the docstring names, made observable.

        With S=1 and internally homogeneous speeches, a whole-block re-deal
        gives every slot a pure corner, so each draw's pairwise JSD is exactly 0
        or exactly 1 and the mean over `n_draws` is an exact multiple of
        `1/n_draws`. A floor that dealt PARAGRAPHS would hand each slot a
        mixture of both presidents' bins — the measured contrast is 0.580 for
        the block deal against 0.042 for a paragraph deal, and the paragraph
        deal's mean is not a multiple of 1/n_draws either.
        """
        design, pools = self._disjoint_floor_design()
        n_draws = 200
        value, _ = C.speech_block_floor(
            design, pools, np.random.default_rng(0), n_draws
        )
        assert value == pytest.approx(0.580, abs=0.005)   # measured
        scaled = value * n_draws
        assert scaled == pytest.approx(round(scaled), abs=1e-3), (
            "some re-dealt slot was a MIXTURE of bins: the floor is dealing "
            "paragraphs, not whole speech blocks"
        )

    def test_the_per_slot_floor_resolves_WHICH_slot_it_priced(self):
        """The returned vector is one entry per slot, IN THE ORDER GIVEN.

        The previous version of this test asserted `per_slot == [value, value]`
        on TWO slots, which is an algebraic identity rather than a behaviour:
        the JSD is symmetric with an exact-zero diagonal, so at `n_slots == 2`
        each slot's mean distance to "the others" IS the single pairwise
        distance, for ANY re-deal — including one that priced the wrong window
        entirely. It could not fail.

        At three slots the entries genuinely differ, and WHICH one is largest is
        a fact about the pools this call was handed. The thin slot re-deals a
        2-speech block, so its S=8 with-replacement picks stay concentrated,
        while the two 10-speech slots both land near the pooled average and are
        therefore close to each other. Moving the thin pool to a different slot
        index must move the largest entry with it; a floor that misindexed slots,
        or equalized the blocks, or ignored the pools' order, would not.
        """
        design = C.build_design(
            _mutually_disjoint_pure_speeches(3),
            C.ArmSpec("s8", "corex", None, 8, 6, "test"),
        )
        thin, fat_a, fat_b = (
            design.global_pool[0][:2], design.global_pool[1], design.global_pool[2]
        )
        assert [len(thin), len(fat_a), len(fat_b)] == [2, 10, 10]

        value_first, thin_first = C.speech_block_floor(
            design, [thin, fat_a, fat_b], np.random.default_rng(0), 400
        )
        _, thin_last = C.speech_block_floor(
            design, [fat_a, fat_b, thin], np.random.default_rng(0), 400
        )
        assert len(thin_first) == len(thin_last) == 3
        # The per-slot vector must be normalized by "the number of OTHER slots"
        # (E-1), not by E. Each unordered pair appears in exactly two rows of the
        # symmetric matrix, so summing the rows double-counts every pair; only
        # dividing by E-1 makes the mean of `per_slot` equal the scalar mean over
        # unordered pairs. This is an exact identity under the correct divisor and
        # is off by a factor (E-1)/E under the wrong one -- a pin on the divisor,
        # not a tautology. `per_slot` is not a diagnostic: it accumulates into
        # `pres_floor` -> `mean_speech_block_floor` / `corrected_distance` in
        # `paradox.parquet`, i.e. the published H2b inference.
        assert thin_first.mean() == pytest.approx(value_first)
        # The fixture HAS the property: at three slots the entries are not all
        # equal, so an ordering claim about them is not vacuous.
        assert thin_first.max() - thin_first.min() > 0.02, (
            "the three slots priced identically; this fixture cannot resolve order"
        )
        assert int(np.argmax(thin_first)) == 0
        assert int(np.argmax(thin_last)) == 2

    def test_the_floor_keeps_each_slots_OWN_speech_count_when_it_redeals(self):
        """A thin slot must stay thin in the floor, or the floor is optimistic
        exactly where it matters.

        The re-deal pools every eligible speech, but it hands slot `e` a block
        of `len(pools[e])` of them — so a president with 2 qualifying speeches
        still draws their S=8 speeches, with replacement, from a 2-speech block
        and can come back pure. Equalizing the blocks (7/7/7 here) averages that
        thin slot out and understates the floor: measured 0.260-0.274 across 8
        seeds at 500 draws for the real deal, against 0.193-0.220 for an
        equalized one. That is CLAUDE.md's "two floors, two different n's" in
        the floor's own machinery.
        """
        design = C.build_design(
            _mutually_disjoint_pure_speeches(3),
            C.ArmSpec("s8", "corex", None, 8, 6, "test"),
        )
        pools = [
            design.global_pool[0][:2], design.global_pool[1], design.global_pool[2]
        ]
        assert [len(p) for p in pools] == [2, 10, 10]
        value, _ = C.speech_block_floor(
            design, pools, np.random.default_rng(0), 500
        )
        assert value == pytest.approx(0.274, abs=0.02)
        assert value > 0.24, "the thin slot was averaged away: blocks were equalized"

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

    def test_a_speech_with_fewer_than_B_paragraphs_never_qualifies(self):
        """A speech shorter than B cannot supply B distinct paragraphs, so the
        sampler would draw the same handful of rows over and over and call it a
        cluster. The floor is part of the design, not an optimisation: with
        B one higher than the corpus's speech length NOTHING qualifies, and the
        arm reports no eligible president rather than a thin one."""
        comp = _coextensive_corpus(3)   # 10 paragraphs in every speech
        n_speeches = len(np.unique(comp.doc_name))
        fits = C.build_design(comp, C.ArmSpec("b10", "corex", None, 2, 10, "test"))
        too_long = C.build_design(comp, C.ArmSpec("b11", "corex", None, 2, 11, "test"))
        assert len(fits.qualifying) == n_speeches
        assert len(too_long.qualifying) == 0
        assert all(len(w.presidents) == 0 for w in too_long.windows)

    def test_exactly_S_qualifying_speeches_is_eligible_and_S_minus_one_is_not(self):
        """The pre-registered rule is "at least S", and which side of it the
        boundary falls on decides who is in the room in the thinnest windows —
        which is where the `MIN_PRESIDENTS` floor and the whole coverage claim
        live."""
        span = list(range(1800, 1861, 2))
        comp = _hand_corpus([
            ("P_exactly_S", [1800, 1802], None),
            ("P_one_short", [1800], None),
            ("P_spanner", span, None),
        ], 3)
        design = C.build_design(comp, _TINY_ARM)      # S = 2
        first = design.windows[0]
        assert (first.start, first.end) == (1800, 1829)
        eligible = {design.presidents[i] for i in first.presidents}
        assert "P_exactly_S" in eligible
        assert "P_one_short" not in eligible

    def test_the_observed_curve_samples_the_WINDOWS_pool_not_the_whole_career(self):
        """A window's estimate must be about that window.

        Here P00 talks bin 0 for the first half of the corpus and bin 1 for the
        second, while P01 and P02 only ever talk bin 0. In the first window
        every eligible speech is bin 0, so the observed dispersion is exactly
        zero. A curve that reached for each president's GLOBAL career pool —
        which is what the permuted curve deliberately does — reads 0.424 there
        instead, and the whole rolling grid stops meaning anything.
        """
        comp = _per_speech_corpus(
            lambda p, s, i: 1 if (p == 0 and s >= 5) else 0, 3, 3
        )
        design = C.build_design(comp, _TINY_ARM)
        curve = C.dispersion_curve(
            design, [6, 1], n_draws=16, with_floor=False, with_entropy_match=False
        )
        first = int(np.flatnonzero(curve.used)[0])
        assert curve.end[first] <= 1829
        assert curve.dispersion[first] == pytest.approx(0.0, abs=1e-6)

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
        """The donor set is restricted to the ever-eligible presidents so that
        every slot draws S speeches WITH replacement from a pool guaranteed to
        hold at least S of them. A one-speech presidency in the donor set would
        fill a slot from a pool of one — eight identical speeches dressed as a
        cluster draw.

        The fixture must actually CONTAIN a never-eligible president: on
        `_small_design()` every president is ever-eligible, so the loop below
        never executes and the assertion proves nothing.
        """
        span = list(range(1800, 1901, 2))
        comp = _hand_corpus([
            ("Pa", span, None), ("Pb", span, None), ("Pc", span, None),
            ("Pz_one_speech", [1830], None),
        ], 3)
        design = C.build_design(comp, _TINY_ARM)
        pool = set(C.ever_eligible(design).tolist())
        outsiders = [
            i for i in range(len(design.presidents)) if i not in pool
        ]
        assert [design.presidents[i] for i in outsiders] == ["Pz_one_speech"]
        assert len(design.global_pool[outsiders[0]]) < _TINY_ARM.n_speeches
        rng = np.random.default_rng(0)
        for _ in range(25):
            donor = C._donor_map(design, rng)
            for i in outsiders:
                assert donor[i] == i
                assert i not in donor[list(pool)].tolist()

    def test_the_donor_map_actually_MOVES_presidents(self):
        """"Is a bijection" is satisfied by the identity, and an identity donor
        map would make every null draw a re-run of the observed curve — a null
        that can never reject, dressed as one that measured 5% size. A uniform
        random bijection leaves about ONE fixed point however large the pool is;
        measured over 50 draws on a 10-president pool: min 0 fixed points, mean
        1.16.
        """
        design = _small_design()
        pool = C.ever_eligible(design)
        assert len(pool) >= 5
        rng = np.random.default_rng(0)
        fixed = [
            int((C._donor_map(design, rng)[pool] == pool).sum()) for _ in range(50)
        ]
        assert min(fixed) < len(pool), "the donor map never moved anybody"
        assert np.mean(fixed) < 0.5 * len(pool)

    def test_a_permuted_slot_is_filled_from_the_DONORS_pool(self):
        """The permutation must move CONTENT, not just a label.

        A donor map pointing every slot at one president makes every slot hold
        the same president's speeches, so dispersion has to collapse to sampling
        noise (measured max 0.228 against an observed mean of 0.823). A permuted
        curve that quietly kept each president's own pool would not move at all,
        and the entire inference — every p-value this module publishes — would
        be computed against the observed curve itself.
        """
        design = C.build_design(
            _coextensive_corpus(4, distinct_agendas=True), _TINY_ARM
        )
        observed = C.dispersion_curve(
            design, [2, 1], n_draws=8, with_floor=False, with_entropy_match=False
        )
        one_donor = np.zeros(len(design.presidents), dtype=np.int64)
        permuted = C.dispersion_curve(
            design, [2, 1], donor=one_donor, n_draws=8,
            with_floor=False, with_entropy_match=False,
        )
        assert np.nanmean(observed.dispersion) > 0.6
        assert np.nanmax(permuted.dispersion[permuted.used]) < 0.35

    def test_permutation_null_computes_every_statistic_it_was_asked_for(self):
        """`permutation_null` decides for itself whether to compute the floor
        and the entropy match, from the statistics it was asked for. If either
        switch were wrong the corresponding rho would come back NaN — silently,
        on both the observed value and every null draw — and the decision table
        would never be able to reach the cell that reads it."""
        design = C.build_design(
            _coextensive_corpus(5, distinct_agendas=True), _TINY_ARM
        )
        observed, null = C.permutation_null(
            {"rolling": design}, 0, n_permutations=3, n_draws=4, floor_draws=4
        )
        for statistic in ("dispersion", "floor_corrected", "excess_ratio"):
            key = ("rolling", "all_windows", statistic)
            assert key in observed, statistic
            assert np.isfinite(observed[key]), statistic
            assert np.isfinite(null[key]).all(), statistic

    @pytest.mark.parametrize("n_permutations", [1, 3])
    def test_every_null_draw_is_a_VALID_spearman_rho(self, n_permutations):
        """Every element of the null is a Spearman rho and therefore lives in
        [-1, 1].

        The arrays used to be allocated with `np.empty`, so a draw the loop
        never wrote carried recycled memory straight into `_one_sided_p` and the
        published `null_mean` / `null_sd` / `null_crit_05` — at R=1 the recycled
        value was observed to be 1829.5, a window centre year from an earlier
        array. `permutation_null` now allocates with `np.full(..., np.nan)`, so
        an unwritten draw is structurally visible: it drops out of the null and
        downgrades `ci_status` through the existing `n_permutations_valid` path.
        The change was byte-identical on a complete run, because every slot is
        written today — this test is what keeps a future slot that is not from
        being read as a rho.
        """
        design = C.build_design(
            _coextensive_corpus(5, distinct_agendas=True), _TINY_ARM
        )
        _, null = C.permutation_null(
            {"rolling": design}, 0, n_permutations=n_permutations,
            n_draws=4, floor_draws=4,
        )
        for key, draws in null.items():
            assert len(draws) == n_permutations, key
            finite = draws[np.isfinite(draws)]
            assert np.all(np.abs(finite) <= 1.0), (key, draws)
        # `pair_regression` is legitimately NaN on this 5-president fixture, so
        # the sweep above would pass vacuously on an all-NaN array. The three
        # window statistics must be non-empty for it to have proved anything.
        for statistic in ("dispersion", "floor_corrected", "excess_ratio"):
            draws = null[("rolling", "all_windows", statistic)]
            assert int(np.isfinite(draws).sum()) == n_permutations, statistic

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


def _hand_curve(dispersion, *, floor=None, entropy_matched=None, start=None):
    """A `Curve` whose derived columns are hand-derivable.

    `floor_corrected` and `excess_ratio` are properties, not stored values, and
    each of them IS a published statistic and a decision-table cell — so they
    need to be pinned on values a reader can check, not only observed through a
    full sampling run.
    """
    dispersion = np.asarray(dispersion, dtype=float)
    n = len(dispersion)
    start = np.arange(1800, 1800 + 2 * n, 2) if start is None else np.asarray(start)
    z = np.zeros(n)
    return C.Curve(
        start=start, end=start + C.WINDOW_LEN - 1,
        center=start + (C.WINDOW_LEN - 1) / 2.0,
        n_eligible=np.full(n, 5), status=["ok"] * n, used=np.ones(n, dtype=bool),
        dispersion=dispersion,
        entropy_matched=np.ones(n) if entropy_matched is None
        else np.asarray(entropy_matched, dtype=float),
        floor=z if floor is None else np.asarray(floor, dtype=float),
        mean_entropy=z,
        pair_jsd=np.zeros((2, 2)), pair_year=np.zeros((2, 2)), pair_n=np.zeros((2, 2)),
        pres_jsd=np.zeros(2), pres_floor=np.zeros(2), pres_n=np.zeros(2),
    )


class TestDerivedCurveStatistics:
    def test_floor_corrected_subtracts_the_per_window_floor(self):
        curve = _hand_curve([0.8, 0.6, 0.5], floor=[0.1, 0.2, 0.4])
        assert curve.floor_corrected == pytest.approx([0.7, 0.4, 0.1])

    def test_a_flat_curve_over_a_rising_floor_is_a_DECLINING_floor_corrected(self):
        """The `artifact_floor` cell of the decision table exists for exactly
        this shape, and it is invisible in `dispersion`: nothing about the
        raw curve moves, and the corrected one falls monotonically."""
        curve = _hand_curve([0.6] * 5, floor=[0.05, 0.10, 0.15, 0.20, 0.25])
        stats = C.curve_statistics(curve, "rolling")
        assert stats[("all_windows", "dispersion")] == pytest.approx(0.0) or np.isnan(
            stats[("all_windows", "dispersion")]
        )
        assert stats[("all_windows", "floor_corrected")] == pytest.approx(-1.0)

    def test_excess_ratio_is_dispersion_over_the_entropy_matched_null(self):
        curve = _hand_curve([0.8, 0.6], entropy_matched=[0.8, 1.2])
        assert curve.excess_ratio == pytest.approx([1.0, 0.5])

    def test_excess_ratio_is_nan_where_the_entropy_matched_null_is_zero(self):
        """Never an inf and never a silent 0: a window whose entropy-matched
        null vanished has no ratio, and must drop out of the trend rather than
        dominate it."""
        curve = _hand_curve([0.8, 0.6], entropy_matched=[0.0, 1.2])
        assert np.isnan(curve.excess_ratio[0])
        assert curve.excess_ratio[1] == pytest.approx(0.5)

    def test_ends_2014_selects_on_the_window_END_not_its_START(self):
        """Prereg 7.4's dating procedure estimates the trend on data ending
        2014. A mask on the window START would silently admit windows running
        to 2043 — every one of them containing post-2014 content, which is the
        one thing the treatment exists to exclude. Here 8 of 20 windows end by
        2014 while all 20 start by then, and the two masks disagree in SIGN:
        -1.0 against +0.747.
        """
        curve = _hand_curve(
            np.concatenate([np.linspace(0.9, 0.5, 8), np.linspace(0.55, 2.0, 12)]),
            start=np.arange(1970, 2010, 2),
        )
        assert int((curve.end <= C.DATING_END_YEAR).sum()) == 8
        assert int((curve.start <= C.DATING_END_YEAR).sum()) == 20
        stats = C.curve_statistics(curve, "rolling")
        assert stats[("ends_2014", "dispersion")] == pytest.approx(-1.0)
        assert stats[("all_windows", "dispersion")] > 0.5


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

    def test_the_pair_regression_averages_over_each_pairs_shared_windows(self):
        """Prereg 7.6 puts ONE point per president pair: mean pairwise JSD
        against the mean centre year of the windows the pair shared. The
        accumulators are SUMS, so the division by `pair_n` is what turns them
        into that. Pairs share wildly different numbers of windows (1 to 6
        here), so regressing the raw sums ranks pairs by how long they
        overlapped rather than by how far apart their agendas were — and flips
        the answer from rho = -1.0 to rho = +0.943 on this fixture.
        """
        iu, ju = np.triu_indices(4, k=1)
        years = np.array([1800.0, 1810, 1820, 1830, 1840, 1850])
        jsd = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        shared = np.array([1.0, 2, 3, 4, 5, 6])
        pair_jsd, pair_year, pair_n = (np.zeros((4, 4)) for _ in range(3))
        pair_jsd[iu, ju] = jsd * shared
        pair_year[iu, ju] = years * shared
        pair_n[iu, ju] = shared
        curve = _hand_curve([0.0, 0.0, 0.0])._replace(
            pair_jsd=pair_jsd, pair_year=pair_year, pair_n=pair_n
        )
        assert C._pair_regression_rho(curve) == pytest.approx(-1.0)

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


class TestPerPresidentAccumulators:
    """`paradox_table` publishes `pres_jsd / pres_n` and `n_windows_eligible`
    straight out of these two arrays, so both have to mean exactly what their
    names say."""

    def _curve(self) -> tuple[C.Design, C.Curve]:
        design = C.build_design(_mutually_disjoint_pure_speeches(3), _TINY_ARM)
        curve = C.dispersion_curve(
            design, [9, 1], n_draws=4, with_floor=False, with_entropy_match=False
        )
        return design, curve

    def test_pres_n_counts_windows_not_president_slots(self):
        design, curve = self._curve()
        n_used = int(curve.used.sum())
        assert n_used == len(design.windows) == 13
        assert curve.pres_n.tolist() == [float(n_used)] * 3

    def test_pres_jsd_is_the_mean_distance_to_the_OTHERS_not_to_everyone(self):
        """Three presidents whose agendas are mutually disjoint are exactly one
        bit apart, so each one's mean distance to the other TWO is 1.0 in every
        window. Dividing by E instead of E-1 would report 2/3 — every published
        `mean_jsd_to_contemporaries` deflated by the same factor, and the
        `corrected_distance` ranking silently re-scaled."""
        _, curve = self._curve()
        n_used = float(curve.used.sum())
        assert curve.pres_jsd == pytest.approx(np.full(3, n_used))
        assert (curve.pres_jsd / curve.pres_n) == pytest.approx(np.ones(3))

    def test_president_first_year_is_their_FIRST_year_not_their_last(self):
        """It is published on every jackknife row (`left_out_first_year`) and
        every paradox row (`first_year`), and it is also what orders the
        presidents."""
        comp = _hand_corpus([
            ("Pa", list(range(1800, 1861, 2)), None),
            ("Pb", list(range(1810, 1871, 2)), None),
            ("Pc", list(range(1820, 1881, 2)), None),
        ], 3)
        design = C.build_design(comp, _TINY_ARM)
        assert dict(zip(design.presidents, design.president_first_year.tolist())) == {
            "Pa": 1800, "Pb": 1810, "Pc": 1820,
        }


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

    @pytest.mark.parametrize(
        "n_windows,expected",
        [
            (0, "suppressed_n_floor"),
            (1, "suppressed_n_floor"),
            (2, "suppressed_n_floor"),          # boundary: 2 is below the floor
            (3, "low_cluster_caution"),         # boundary: 3 is not below it
            (9, "low_cluster_caution"),
            (10, "ok"),                         # boundary: 10 is not below `ok`
            (11, "ok"),
        ],
    )
    def test_thresholds_are_two_sided_and_hand_derived(self, n_windows, expected):
        """Both sides of both boundaries, like the jackknife and null gates.
        The previous coverage was one-sided (only values at or above each cut),
        so a `>= 3` that had drifted to `>= 2` passed."""
        assert C.paradox_status(n_windows) == expected

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

    @pytest.mark.parametrize("declining_arm", ["corex_all", "corex_sotu"])
    def test_convergence_needs_BOTH_arms_on_the_entropy_matched_null(
        self, declining_arm
    ):
        """"No vote rule" is a pre-registered commitment (task Key Decisions): a
        ">= 3 of 4 methods decline" rule fires 12-46% under a true null, and the
        arms are not independent evidence. One arm clearing the entropy-matched
        null is therefore NOT convergence — it is broadening. Both arms are
        parametrized because the previous suite tested only the `corex_all` side
        of every `and` in this function.
        """
        table = _null_table(**{
            "corex_all.dispersion": True, "corex_sotu.dispersion": True,
            "corex_all.floor_corrected": True, "corex_sotu.floor_corrected": True,
            f"{declining_arm}.excess_ratio": True,
        })
        assert C.score_decision_table(table)["cell"] == "broadening"

    @pytest.mark.parametrize("surviving_arm", ["corex_all", "corex_sotu"])
    def test_the_floor_cell_needs_BOTH_arms_to_survive_the_floor(
        self, surviving_arm
    ):
        """The same commitment one rule earlier: a decline that survives the
        noise floor in one arm only is `artifact_floor`, not evidence."""
        table = _null_table(**{
            "corex_all.dispersion": True, "corex_sotu.dispersion": True,
            f"{surviving_arm}.floor_corrected": True,
        })
        assert C.score_decision_table(table)["cell"] == "artifact_floor"

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

    def test_the_committed_artifact_still_carries_the_values_that_were_reported(self):
        """ONE anchor on the published numbers themselves.

        Every other test in this file runs the estimator on a synthetic corpus,
        so a change that moved every real number while preserving every
        synthetic behaviour would go green — that is why one mutant survived the
        199-test suite as "equivalent": nothing anywhere asserted what
        `data/convergence/` actually says. These four values are the ones the
        findings note leads with; if a refactor moves any of them, the artifact
        and the note have diverged and the run must be re-verified rather than
        the anchor re-fitted.

        Tolerances are loose enough not to fail on a platform's last ULP and
        tight enough that no real change hides inside them.
        """
        if not (C.META_PATH.exists() and C.PERMUTATION_PATH.exists()):
            pytest.skip("data/convergence/ has not been generated in this tree")
        meta = json.loads(C.META_PATH.read_text())
        assert meta["decision"]["cell"] == "no_convergence"
        assert meta["selftest"]["c_permutation_size"] == pytest.approx(0.05, abs=1e-9)
        assert meta["selftest"]["a_cluster_null_rho"] == pytest.approx(
            -0.0893, abs=5e-4
        )
        primary = pd.read_parquet(C.PERMUTATION_PATH).set_index(
            ["arm", "window_kind", "treatment", "statistic"]
        ).loc[("corex_all", "rolling", "all_windows", "dispersion")]
        assert float(primary["rho"]) == pytest.approx(-0.0047, abs=5e-4)
        assert not bool(primary["significant_decline"])

    def test_module_never_imports_anthropic(self):
        """$0 GUARD, scanned over the AST so that the docstring PROMISING it
        cannot be what satisfies the test.

        SCOPE, stated because this test is structurally blind past it: it ASTs
        THIS FILE only. `anthropic` is in fact resident in `sys.modules` during
        every run, transitively — `convergence.py` -> `eras.check_staleness` ->
        `annotate._rates` -> `anthropic.types`. So a green result here does NOT
        mean the process is anthropic-free, and no strengthening of this scan
        could tell you that. The behavioural twin below is what actually holds
        the $0 line.
        """
        tree = ast.parse(open(C.__file__).read())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "anthropic" not in imported

    def test_a_full_build_runs_with_client_construction_booby_trapped(
        self, tmp_path, monkeypatch, tiny_pipeline
    ):
        """The behavioural half of the $0 guard, which the AST scan above cannot
        provide: drive a full `build_convergence` with every Anthropic client
        constructor rigged to explode. If any code path reached for the paid API
        the build dies; completing proves it did not.

        This is the pattern `combat.py` already uses
        (`test_combat_contracts.py::test_a_full_build_runs_with_client_construction_booby_trapped`).
        `convergence` had only the static scan, which is blind to the transitive
        import by construction — a real gap, since the transitive import is
        exactly the thing a reader would want the guard to be watching.
        """
        import anthropic

        def _bomb(*a, **k):  # pragma: no cover - must never run
            raise AssertionError(
                "a client was constructed on the $0 path; this module must never "
                "reach for the paid API"
            )

        for name in ("Anthropic", "AsyncAnthropic", "AnthropicBedrock", "AnthropicVertex"):
            if hasattr(anthropic, name):
                monkeypatch.setattr(anthropic, name, _bomb)

        out = tmp_path / "convergence"
        C.build_convergence(out_dir=out, n_permutations=3, progress=False)
        assert (out / C.META_PATH.name).exists()
        assert json.loads((out / C.META_PATH.name).read_text())["api_calls"] == 0


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
        `paradox_status` must not each invent their own words.

        `paradox_table`'s gate used to be an inline `np.where` in the frame
        builder rather than a named function, which is exactly why it drifted
        out of this union once already — it was named in this docstring while
        being excluded from the set the test actually checked. SIMPLIFY extracted
        it to `paradox_status`, so all four now enter the union the same way and
        the asymmetry that allowed the drift is gone.
        """
        produced = {C.window_status(n) for n in range(0, 8)}
        produced |= {C.jackknife_status(u, 100) for u in (0, 2, 3, 70, 80, 95, 100)}
        produced |= {C.paradox_status(n) for n in (0, 1, 2, 3, 9, 10, 11)}
        produced |= {
            C.null_status(rho, v, 100)
            for rho in (-0.5, float("nan"))
            for v in (0, 50, 90, 100)
        }
        design = C.build_design(_coextensive_corpus(4), _TINY_ARM)
        paradox = C.paradox_table(
            design,
            _paradox_curve(design, [0.1, 0.2, 0.3, 0.4], [0.0] * 4,
                           [1.0, 3.0, 10.0, 12.0]),
            anchor=pd.Series(dtype=float),
        )
        assert set(paradox["ci_status"]) == {
            "suppressed_n_floor", "low_cluster_caution", "ok"
        }
        produced |= set(paradox["ci_status"])
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

    def test_dropping_a_president_removes_THEIR_content_from_every_window(self):
        """`drop` has to delete a president from the design, not merely from a
        row label. Three presidents talk only bin 0 and the fourth talks only
        bin 1, so the whole of the corpus's dispersion is that one president:
        deleting them takes it to exactly zero, and deleting nobody leaves it at
        0.5. Every eligible count falls by exactly one.
        """
        comp = _per_speech_corpus(lambda p, s, i: 1 if p == 3 else 0, 4, 3)
        design = C.build_design(comp, _TINY_ARM)
        assert design.presidents[3] == "P03"
        full = C.dispersion_curve(
            design, [8, 1], n_draws=8, with_floor=False, with_entropy_match=False
        )
        without = C.dispersion_curve(
            design, [8, 1], drop=3, n_draws=8,
            with_floor=False, with_entropy_match=False,
        )
        assert np.nanmean(full.dispersion) == pytest.approx(0.5, abs=0.05)
        assert np.nanmax(without.dispersion[without.used]) == pytest.approx(
            0.0, abs=1e-6
        )
        assert (without.n_eligible == full.n_eligible - 1).all()

    def test_every_jackknife_row_really_leaves_ITS_OWN_president_out(self):
        """Three coextensive presidents sit exactly on the `MIN_PRESIDENTS`
        floor, so a row that genuinely dropped its president has NO trend
        windows left while the full design has 16. A `jackknife` that passed
        `drop=None` — or a `dispersion_curve` that ignored `drop` — would report
        all 16 on every row and publish 45 identical copies of the full-corpus
        rho under 45 different president names.
        """
        design = C.build_design(_coextensive_corpus(3), _TINY_ARM)
        frame = C.jackknife(design, 0, np.linspace(-1, 1, 21), n_draws=4)
        assert len(frame) == 3
        assert frame["n_windows_full_design"].eq(C.n_windows_in_trend(design)).all()
        assert frame["n_windows_full_design"].gt(0).all()
        assert frame["n_windows_used"].eq(0).all()
        assert frame["ci_status"].eq("no_data").all()

    def test_leaving_out_different_presidents_gives_different_answers(self):
        """The counter to the fixture above, on a design where the floor never
        binds: the rows must still differ from each other, because they are
        different estimates and not one estimate copied 5 times."""
        design = C.build_design(
            _coextensive_corpus(5, distinct_agendas=True), _TINY_ARM
        )
        frame = C.jackknife(design, 0, np.linspace(-1, 1, 21), n_draws=8)
        rhos = frame["rho"].to_numpy()
        assert np.isfinite(rhos).all()
        assert rhos.std() > 0.01, "every leave-one-out row returned the same rho"

    def test_a_president_who_is_never_eligible_gets_no_jackknife_row(self):
        """A president outside the donor set never entered any window, so
        "leaving them out" is not an estimate — it is the full-design rho with a
        misleading name on it, scored against the full-design null."""
        span = list(range(1800, 1861, 2))
        comp = _hand_corpus([
            ("Pa", span, None), ("Pb", span, None), ("Pc", span, None),
            ("Pz_one_speech", [1830], None),
        ], 3)
        design = C.build_design(comp, _TINY_ARM)
        assert "Pz_one_speech" in design.presidents
        assert design.presidents.index("Pz_one_speech") not in (
            C.ever_eligible(design).tolist()
        )
        frame = C.jackknife(design, 0, np.linspace(-1, 1, 21), n_draws=4)
        assert set(frame["left_out_president"]) == {"Pa", "Pb", "Pc"}

    def test_the_gate_reads_RETENTION_the_right_way_round(self):
        """`jackknife_status(used, full)` is not symmetric, and the two
        arguments are both plain ints — so swapping them at the call site is
        invisible until a row that should be suppressed publishes as `ok`.

        A, B and C serve the whole 1800-1900 span; D_early only its first
        thirty years. Three eligible presidents is exactly the floor, so
        deleting any of A/B/C drops every window D_early does not cover:
        15 of 36 survive (retention 0.42, `suppressed_n_floor`), while the
        swapped call reads 36/15 = 2.4 and returns `ok` on all four rows.
        """
        span = list(range(1800, 1901, 2))
        comp = _hand_corpus([
            ("A", span, None), ("B", span, None), ("C", span, None),
            ("D_early", list(range(1800, 1831, 2)), None),
        ], 3)
        design = C.build_design(comp, _TINY_ARM)
        assert C.n_windows_in_trend(design) == 36
        frame = C.jackknife(
            design, 0, np.linspace(-1, 1, 21), n_draws=4
        ).set_index("left_out_president")
        for pivotal in ("A", "B", "C"):
            assert int(frame.loc[pivotal, "n_windows_used"]) == 15
            assert frame.loc[pivotal, "ci_status"] == "suppressed_n_floor"
        assert int(frame.loc["D_early", "n_windows_used"]) == 36
        assert frame.loc["D_early", "ci_status"] == "ok"

    def test_jackknife_frame_carries_the_gate_on_every_row(self):
        design = C.build_design(_coextensive_corpus(6), _TINY_ARM)
        frame = C.jackknife(design, 0, np.linspace(-1, 1, 21), n_draws=4)
        assert len(frame) == 6
        assert _trust_gate_violations({"jackknife": frame}) == []
        assert (frame["n_windows_full_design"] == C.n_windows_in_trend(design)).all()

    def test_a_jackknife_RISE_is_never_a_significant_DECLINE(self):
        """`jackknife` carries its OWN copy of the `significant_decline` rule
        (`rho < 0` AND `p <= ALPHA`); `null_rows` carries an identical one.

        The `null_rows` copy has two tests and this one had none — the same
        duplicated-rule class as `ends_2014`, and a duplicated rule is only as
        guarded as its least-guarded copy. Applied symmetrically here.

        A null whose every draw is +1.0 makes `p_one_sided` significant for
        EVERY finite rho, so the flag reduces to the sign conjunct alone: the
        rows that are flagged must be exactly the rows whose rho is negative.
        A rule missing `rho < 0` would flag all five; a rule hard-wired to False
        would flag none.
        """
        design = C.build_design(
            _coextensive_corpus(5, distinct_agendas=True), _TINY_ARM
        )
        frame = C.jackknife(design, 0, np.full(200, 1.0), n_draws=8)
        assert frame["p_one_sided"].le(C.ALPHA).all(), "the null did not make p small"
        # the fixture HAS the property: both signs are present, so neither a
        # flag-everything nor a flag-nothing rule can pass.
        assert frame["rho"].lt(0).any() and frame["rho"].gt(0).any()
        assert (
            frame["significant_decline"].tolist() == frame["rho"].lt(0).tolist()
        )

    def test_a_jackknife_rho_that_is_NOT_extreme_is_not_flagged(self):
        """The other half of the conjunct: a negative rho sitting in the middle
        of its null is not a significant decline. Without this the test above
        would also pass on a rule that read the sign and ignored the p."""
        design = C.build_design(
            _coextensive_corpus(5, distinct_agendas=True), _TINY_ARM
        )
        frame = C.jackknife(design, 0, np.linspace(-1.0, 1.0, 201), n_draws=8)
        negative = frame[frame["rho"] < 0]
        assert not negative.empty
        assert negative["p_one_sided"].gt(C.ALPHA).all()
        assert not frame["significant_decline"].any()

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

    def test_a_significant_RISE_is_never_a_significant_DECLINE(self):
        """`significant_decline` is what the decision table reads, and the
        hypothesis is one-sided. A rho sitting BELOW its whole null but on the
        positive side of zero is a significant result about a corpus getting
        MORE dispersed; calling it a decline would route an anti-convergence
        finding straight into the `convergence` cell.
        """
        rising = {("rolling", "all_windows", "dispersion"): 0.9}
        null = {("rolling", "all_windows", "dispersion"): np.linspace(0.95, 1.0, 100)}
        rows = pd.DataFrame(C.null_rows(C.ARMS[0], rising, null, 100))
        assert rows["p_one_sided"].iloc[0] <= C.ALPHA   # it IS significant
        assert rows["rho"].iloc[0] > 0
        assert not bool(rows["significant_decline"].iloc[0])

    def test_null_crit_05_is_the_LOWER_five_percent_of_the_null(self):
        """The published critical value, and the number this whole design
        exists to get right: the honest one-sided 5% cut on these 93%-overlapping
        windows is rho <= -0.41, not the -0.16 a closed-form p implies. A column
        reporting the UPPER tail instead would hand a reader a positive
        threshold to compare a negative rho against."""
        draws = np.linspace(-1.0, 1.0, 101)
        rows = pd.DataFrame(C.null_rows(
            C.ARMS[0],
            {("rolling", "all_windows", "dispersion"): -0.5},
            {("rolling", "all_windows", "dispersion"): draws}, 101,
        ))
        assert rows["null_crit_05"].iloc[0] == pytest.approx(-0.9)
        assert rows["null_crit_05"].iloc[0] < rows["null_mean"].iloc[0]

    def test_two_sided_p_centres_on_the_MEDIAN_not_the_mean(self):
        """A permutation null need not be symmetric, and the two-sided p asks
        how far the observed value sits from the null's CENTRE. The
        `[-2,-1,1,2]` fixture above cannot tell the two apart — its mean and
        median are both 0. On a right-skewed null they differ: median 1.0 gives
        p = 1.0 for an observed value sitting exactly on it, and a mean-centred
        version gives 0.833."""
        skewed = np.array([-1.0, 0.0, 1.0, 2.0, 100.0])
        assert C._two_sided_p(1.0, skewed) == pytest.approx(1.0)

    def test_a_significant_decline_is_still_flagged(self):
        """The control: without it the assertion above would also pass on a
        `significant_decline` hard-wired to False."""
        falling = {("rolling", "all_windows", "dispersion"): -0.9}
        null = {("rolling", "all_windows", "dispersion"): np.linspace(-0.2, 0.6, 100)}
        rows = pd.DataFrame(C.null_rows(C.ARMS[0], falling, null, 100))
        assert bool(rows["significant_decline"].iloc[0])

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

    def test_leg_a_is_TWO_sided_flat_means_flat_not_merely_not_declining(
        self, monkeypatch, cheap_selftest
    ):
        """Leg (a)'s claim is FLAT on the clustered null, and its check is
        `|rho| <= threshold` for that reason.

        An estimator that manufactured a strong POSITIVE trend out of a null
        corpus is exactly as broken as one that manufactured a negative one —
        it would read a real convergence as flat and this design's whole
        argument is about estimators that invent trends. A one-sided
        `rho <= threshold` would wave it through.

        Discriminated at a threshold of 0: the shrunken fixture's leg-(a) rho
        is -0.123, so `|rho| <= 0` fails (the gate raises) while `rho <= 0`
        passes. The sign is asserted first, or a future re-seed would make this
        test silently vacuous.
        """
        for name in ("SELFTEST_PARAGRAPH_BIAS_MAX_RHO", "SELFTEST_INJECTED_MAX_RHO",
                     "SELFTEST_MAX_SIZE", "SELFTEST_CLUSTER_FLAT_MAX_ABS_RHO"):
            monkeypatch.setattr(C, name, 1.0)
        baseline = C._selftest(verbose=False)
        rho = baseline["a_cluster_null_rho"]
        assert -1.0 < rho < 0.0, (
            f"fixture drifted: leg (a) rho is {rho}, which cannot separate "
            "`|rho| <= 0` from `rho <= 0`"
        )
        monkeypatch.setattr(C, "SELFTEST_CLUSTER_FLAT_MAX_ABS_RHO", 0.0)
        with pytest.raises(AssertionError, match="a_cluster_flat"):
            C._selftest(verbose=False)

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

    def test_the_published_window_rows_report_the_trend_membership_actually_used(
        self, tmp_path, tiny_pipeline
    ):
        """`used_in_trend` is the column a consumer filters on, and it must
        agree row-for-row with `ci_status`. The committed-artifact sweep cannot
        see a regression here — it reads a parquet built by an earlier commit —
        so the check has to run on a freshly built layer."""
        tables = C.build_convergence(
            out_dir=tmp_path / "convergence", n_permutations=2, progress=False
        )
        curves = tables["dispersion_curves"]
        blind = curves[curves["ci_status"].isin(["no_data", "suppressed_n_floor"])]
        assert len(blind) > 0, "fixture no longer produces a suppressed window"
        assert not blind["used_in_trend"].any()
        usable = curves[curves["ci_status"].isin(["ok", "low_cluster_caution"])]
        assert len(usable) > 0
        assert usable["used_in_trend"].all()

    def test_the_published_in_ends_2014_column_reads_the_window_END(self):
        """The same rule as `curve_statistics`' `ends_2014` mask, in a second
        place — so it needs its own guard. A window starting 1984 runs to 2013;
        one starting 1986 runs to 2015 and carries post-2014 content, which is
        precisely what the dating procedure excludes."""
        design = C.build_design(_coextensive_corpus(3), _TINY_ARM)
        curve = _hand_curve(
            np.linspace(0.9, 0.5, 20), start=np.arange(1970, 2010, 2)
        )
        rows = C._curve_rows(C.ARMS[0], design, curve)
        assert sum(r["in_ends_2014"] for r in rows) == 8
        assert all(
            r["in_ends_2014"] == (r["window_end"] <= C.DATING_END_YEAR) for r in rows
        )

    def test_ends_2014_does_not_exist_off_the_rolling_grid(self):
        """`ends_2014` is a SUBSET of the rolling grid (prereg 7.4), so on any
        other window kind the treatment must be absent, not silently different.

        SIMPLIFY created this branch when it collapsed two independent copies of
        the rule into `ends_2014_mask`; the all-False return replaced an inline
        `design.window_kind == "rolling"` conjunct that only one of the two
        copies carried. Nothing asserted it, so a mutant returning a live mask
        here would publish `in_ends_2014=True` on a grid where the prereg says
        the treatment does not exist. The windows below deliberately END well
        before 2014, so a rolling-grid mask would mark every one of them True —
        which is what makes the all-False assertion a fact about the BRANCH
        rather than about the dates.
        """
        curve = _hand_curve(
            np.linspace(0.9, 0.5, 8), start=np.arange(1800, 1816, 2)
        )
        assert (curve.end <= C.DATING_END_YEAR).all(), (
            "fixture must end before the cutoff, or all-False proves nothing"
        )
        assert not C.ends_2014_mask(curve, "nonoverlapping").any()
        assert C.ends_2014_mask(curve, "rolling").all()

        design = C.build_design(_coextensive_corpus(3), _TINY_ARM)
        assert design.window_kind == "rolling"
        nonoverlapping = design._replace(window_kind="nonoverlapping")
        rows = C._curve_rows(C.ARMS[0], nonoverlapping, curve)
        assert not any(r["in_ends_2014"] for r in rows)

    def test_the_jackknife_is_scored_against_the_ROLLING_grids_null(
        self, tmp_path, tiny_pipeline, monkeypatch
    ):
        """Every jackknife rho is a rolling-grid statistic and its p comes from
        the rolling-grid permutation null (`null_source` says so on every row).
        Scoring it against the 8-window non-overlapping grid's null would
        compare a 105-window rho against the spread of an 8-window one — the
        two differ by roughly an order of magnitude in width.

        Detected by making the two nulls trivially distinguishable: every
        rolling draw is -0.99 (so every observed rho sits above the whole null,
        p = 1.0) and every non-overlapping draw is +0.99 (p = 1/(R+1)).
        """
        real = C.permutation_null

        def tagged(designs, arm_index, **kwargs):
            observed, null = real(designs, arm_index, **kwargs)
            for key in null:
                null[key][:] = -0.99 if key[0] == "rolling" else 0.99
            return observed, null

        monkeypatch.setattr(C, "permutation_null", tagged)
        tables = C.build_convergence(
            out_dir=tmp_path / "convergence", n_permutations=4, progress=False
        )
        jack = tables["jackknife"]
        assert jack["rho"].notna().all()
        assert jack["rho"].between(-0.9, 0.9).all()
        assert jack["p_one_sided"].eq(1.0).all()

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


_LOOPS = (ast.For, ast.AsyncFor, ast.While)


def _loop_leaked_names(func: ast.FunctionDef) -> list[str]:
    """Names READ outside every loop in a function but bound ONLY inside one.

    That is the shape of the `rolling_curve` / `rolling_design` defect: correct
    today purely because `"rolling"` happens to be in `WINDOW_KINDS`, and a
    `NameError` at a distance — or, worse, a silent reuse of the PREVIOUS arm's
    design — the moment that stops being true.

    Loops are found ANYWHERE in the function, not only at its top statement
    level: the earlier version classified statements one level deep, so a `for`
    nested inside an `if`, a `with` or a `try` was invisible to it — and a loop
    guarded by an `if` is if anything MORE likely to leak, because the guard is
    another way for the loop body never to run.
    """
    inside_a_loop: set[int] = set()
    for loop in (n for n in ast.walk(func) if isinstance(n, _LOOPS)):
        inside_a_loop.update(id(n) for n in ast.walk(loop))

    bound_at_top = {a.arg for a in func.args.args} | {
        a.arg for a in func.args.kwonlyargs
    }
    bound_in_loop: set[str] = set()
    read_at_top: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Name):
            continue
        in_loop = id(node) in inside_a_loop
        if isinstance(node.ctx, ast.Store):
            (bound_in_loop if in_loop else bound_at_top).add(node.id)
        elif isinstance(node.ctx, ast.Load) and not in_loop:
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

    @pytest.mark.parametrize(
        "wrapper",
        ["    if flag:\n", "    with open(flag) as fh:\n", "    try:\n"],
    )
    def test_the_scan_sees_a_loop_nested_inside_another_statement(self, wrapper):
        """The scan used to classify only top-level statements, so a `for`
        inside an `if` / `with` / `try` was invisible — and a conditionally
        entered loop is exactly where a leak bites, because the guard is a
        second way for the body never to run."""
        tail = "        pass\n" if wrapper.startswith("    try") else ""
        suffix = "    except Exception:\n        pass\n" if wrapper.startswith(
            "    try"
        ) else ""
        defect = _function_defs(
            "def build(flag, items):\n"
            + wrapper
            + "        for x in items:\n"
            "            picked = x\n"
            + tail
            + suffix
            + "    return summarize(picked)\n"
        )["build"]
        assert _loop_leaked_names(defect) == ["picked"]

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


def _rising_no_topic_corpus(seed: int = 3) -> C.Composition:
    """Zero real convergence; only the UNLABELLED share drifts, 0 -> 0.9.

    President `p` owns bin `p` and shares no substantive bin with anyone, ever.
    What rises with time is the fraction of their paragraphs carrying no label
    at all — the corpus's real 1.49 -> 0.83 labels-per-paragraph drift, in its
    purest form.
    """
    rng = np.random.default_rng(seed)
    blank = {}

    def bin_of(p, s, i):
        u = 0.9 * p / (_RIVAL_PRESIDENTS - 1)
        if (p, s) not in blank:
            blank[(p, s)] = rng.random(10) < u
        return None if blank[(p, s)][i] else p

    return _per_speech_corpus(
        bin_of, _RIVAL_PRESIDENTS, _RIVAL_PRESIDENTS, paras_per_speech=10,
        speeches_per_president=10, president_step=6, year_step=1, year_cycle=6,
    )


def _rival_curve(comp: C.Composition) -> C.Curve:
    return C.dispersion_curve(
        C._selftest_design(comp), [5, 2, 3], with_floor=False
    )


def _rho(curve: C.Curve, values: np.ndarray) -> float:
    return C.spearman_rho(curve.center[curve.used], values[curve.used])


class TestRivalHypothesisSeparator:
    """The most load-bearing behavioural claim in the module: that
    `excess_ratio` reads "broader" and "more alike" differently."""

    # Every band below is a single-seed magnitude, so the MEASURED value is
    # recorded beside it: an assertion whose margin nobody wrote down is a
    # flakiness surface that only shows up as a red build months later. All
    # values below are from seed `_hand_corpus(seed=0)` x `_rival_curve`'s
    # `[5, 2, 3]`, 46 used windows.

    def test_the_broader_corpus_really_does_broaden(self):
        curve = _rival_curve(_broader_corpus())
        used = curve.used
        assert used.sum() >= 40                          # measured 46
        assert _rho(curve, curve.mean_entropy) >= 0.9    # measured +0.9988
        entropy = curve.mean_entropy[used]
        # measured 0.601 -> 3.940 bits, a 6.56x rise
        assert entropy[-1] > 4 * entropy[0]

    def test_broadening_alone_drives_dispersion_down(self):
        """Without this the flat-excess_ratio assertion would be vacuous: there
        would be no decline for the rival null to have to explain away."""
        curve = _rival_curve(_broader_corpus())
        assert _rho(curve, curve.dispersion) <= -0.9     # measured -0.9961
        disp = curve.dispersion[curve.used]
        # measured 0.988 -> 0.580, a ratio of 0.587
        assert disp[-1] < 0.65 * disp[0]

    def test_broadening_alone_leaves_excess_ratio_flat_at_one(self):
        """THE rival-null test: every window's observed dispersion lands within
        10% of what the presidents' own breadths already predict, so there is no
        excess for a shared agenda to explain.

        This is a statement about the STATISTIC, not about the decision path.
        It does not show that `score_decision_table` refuses this corpus — the
        production guard for that is the permutation null, which asks whether
        `excess_ratio`'s TREND is significant, and this test asserts nothing
        about that trend. The asserted band is [0.9, 1.1]; the measured range is
        [0.959, 1.069], so the margin is about 4 points of headroom on each side.
        """
        curve = _rival_curve(_broader_corpus())
        ratio = curve.excess_ratio[curve.used]
        assert np.isfinite(ratio).all()
        assert ratio.min() >= 0.9      # measured 0.9589
        assert ratio.max() <= 1.1      # measured 1.0693

    def test_the_alike_corpus_holds_breadth_fixed(self):
        curve = _rival_curve(_alike_corpus())
        entropy = curve.mean_entropy[curve.used]
        assert entropy.max() / entropy.min() <= 1.05     # measured 1.0232

    def test_real_alignment_drives_excess_ratio_down_too(self):
        """The counter that proves `excess_ratio` is not simply pinned at 1 by
        construction: on a corpus that DID converge, with breadth held fixed,
        the same statistic collapses."""
        curve = _rival_curve(_alike_corpus())
        assert _rho(curve, curve.dispersion) <= -0.9     # measured -0.9533
        assert _rho(curve, curve.excess_ratio) <= -0.8   # measured -0.9504
        assert curve.excess_ratio[curve.used][-1] <= 0.3         # measured 0.188

    def test_the_separator_actually_separates(self):
        """The two corpora are indistinguishable on `dispersion` — both decline
        hard — and are told apart only by `excess_ratio`. That is the whole
        reason the entropy-matched null is mandatory."""
        broader, alike = _rival_curve(_broader_corpus()), _rival_curve(_alike_corpus())
        assert _rho(broader, broader.dispersion) <= -0.9     # measured -0.9961
        assert _rho(alike, alike.dispersion) <= -0.9         # measured -0.9533
        # measured 0.9589 / 0.1880 = 5.10x, against an asserted 3x
        assert broader.excess_ratio[broader.used].min() > 3 * (
            alike.excess_ratio[alike.used].min()
        )

    def test_a_rising_no_topic_share_DOES_move_the_measure(self):
        """The boundary condition on the composition equation's immunity claim.

        Every paragraph contributing mass 1 makes the measure immune to the
        number of labels a labelled paragraph carries. It does NOT make it
        immune to paragraphs carrying NO label: the no-topic bin is a real bin,
        and presidents drifting into it drift toward each other in it.

        This corpus has zero real convergence — president `p` owns bin `p` and
        never shares a substantive bin with anybody — and only the UNLABELLED
        share drifts, 0 -> 0.9. Both `dispersion` and `excess_ratio` collapse
        anyway (measured rho -0.998 and -0.997; excess_ratio 1.026 -> 0.215),
        while mean entropy does not even rise monotonically (rho +0.247), so
        breadth is not what is driving it. `floor_corrected` is what largely
        absorbs this; `excess_ratio` does not, and prereg section 3's immunity
        sentence is narrower than it reads.
        """
        curve = _rival_curve(_rising_no_topic_corpus())
        assert curve.used.sum() >= 40
        assert _rho(curve, curve.dispersion) <= -0.9
        assert _rho(curve, curve.excess_ratio) <= -0.9
        ratio = curve.excess_ratio[curve.used]
        assert ratio[0] == pytest.approx(1.03, abs=0.1)
        assert ratio[-1] <= 0.3
        # The narrated control, now pinned: breadth is NOT what drives this. If
        # mean entropy fell alongside dispersion, "each agenda got narrower"
        # would be an equally good reading of the collapse and the docstring's
        # claim would be wrong. Measured +0.247.
        assert _rho(curve, curve.mean_entropy) == pytest.approx(0.247, abs=0.05)

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
