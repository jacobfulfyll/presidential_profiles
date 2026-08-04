"""`bands.py` — the two-surface confidence-band builder behind `data/bands.parquet`.

Five things are load-bearing here, and each gets its own class or two:

1. **The cluster floor and the bootstrap.** A period resting on one speech must
   publish NO interval rather than the zero-width one a single-cluster resample
   would return, and a period resting on two must publish a visibly enormous
   one. `_ci_status`'s boundaries decide which, so they are pinned exactly.

2. **Surface A is *exactly* sampling-only.** `lo` must EQUAL `lo_sampling` on
   every CorEx row — not "approximately", not "usually". That equality is the
   structural guarantee that the inapplicable annotator component never leaked
   into a chart whose labels have no annotator.

3. **Surface B widens and never narrows**, and its three-way
   `disagreement_status` names the ACTUAL failed precondition. Collapsing
   `no_interval_to_widen` into `unavailable_thin_paired_sample` would report a
   false cause; the precedence when both fail is deliberate and pinned so it
   cannot silently flip.

4. **The left-merge guard.** `validate="one_to_one"` on a LEFT merge cannot see
   diverging key SETS — length survives and every row silently loses its
   disagreement component. The guard must RAISE, and it must raise BEFORE the
   `available.astype(bool)` two dozen lines below, which reads NaN as True. A
   test that pins the raise is what protects that ordering.

5. **The published numbers.** Three bare counts in the module docstring (8,570
   paired paragraphs; The founding's 219; 262 paired speeches) and the
   degeneracy predicate's "12 rows, all 1785" are re-derived from the artifacts
   they are sourced to, because a bare number in prose is this repo's most
   reliably drifting object (CLAUDE.md, twice over).

Everything except the artifact-anchor classes runs on hand-authored frames of a
few dozen rows. The full-rebuild reproducibility test does drive the real
builder, but it is pure local compute over frozen parquets and costs $0 and
~2 seconds; `build_bands()` deliberately never touches `issues.build_issues()`
or any API.

--------------------------------------------------------------------------
Fixture convention — READ THIS BEFORE ADDING A TEST
--------------------------------------------------------------------------

Every helper below (`_labels`, `_sampling`, `_half_widths`, and `_band` /
`_llm_table` in `test_band_charts.py`) fills unspecified fields from
``r.get(key, <one scalar>)``. That keeps the hand-computed oracles in this
suite readable — a reader can see at a glance which field a test is actually
about — but it has one consequence, and it has already produced five separate
defects:

    **A field a test does not explicitly vary is IDENTICAL on every row.**

So the rule for any new test is:

    **No helper default may be shared by two rows the test distinguishes.**

If a test's claim is "era A is flagged and era B is not", "the primary read
more than the secondary", or "these two panels rank differently", then every
field carrying that distinction must be passed explicitly per row. A fixture
that is symmetric in the quantity under test passes just as happily when the
code is wrong — the assertion is comparing a value to itself.

Do NOT fix this by making the helpers emit per-row-varying defaults: the
oracles here are hand-computed against those flat defaults, and randomising
them would trade a known, documented sharp edge for an unreadable suite. Vary
explicitly, at the call site, and say in the test why.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import attention, bands as B
from presidential_profiles import llm_annotations as ann

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

ERA_A = "The founding"
ERA_B = "Expansion"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def real_annotations(monkeypatch):
    """Undo conftest's write-side redirect for the READ-ONLY annotation tables.

    `redirect_annotation_dirs` is autouse and repoints `ANNOTATIONS_DIR` at
    tmp_path so no test can write to the frozen paid artifacts.
    `load_paragraph_annotations` resolves that constant at call time, so the
    real tables become unreadable — which is right for every writer and wrong
    for the handful of tests here that must re-derive a published count from
    the artifact it was sourced to. Point it back, read-only.
    """
    real = attention.TAXONOMY_PATH.parent  # bound at import, never redirected
    monkeypatch.setattr(ann, "ANNOTATIONS_DIR", real)
    return real


@pytest.fixture(scope="module")
def shipped() -> pd.DataFrame:
    """`data/bands.parquet` read by ABSOLUTE worktree path.

    Not via `B.BANDS_PATH`: a worktree's module constants resolve off the
    imported package, and this suite must assert about THIS checkout's
    committed artifact (test-toolchain worktree gotcha).
    """
    return pd.read_parquet(DATA / "bands.parquet")


@pytest.fixture(scope="module")
def shipped_meta() -> dict:
    return json.loads((DATA / "bands_meta.json").read_text())


def _prose(text: str) -> str:
    """Docstring prose with its line wrapping collapsed, so a claim can be
    matched as the sentence it reads as rather than as the lines it happens to
    be broken into today."""
    return " ".join(text.split())


def _labels(specs: list[dict], issues: tuple[str, ...] = ("war", "trade")) -> pd.DataFrame:
    """A stand-in for `paragraph_issues.parquet`.

    Each spec is ``{"doc": name, "year": int, "n": paragraphs, "hits":
    {issue: how many of those paragraphs fire}}`` plus an optional
    ``president``. Boolean dtype matters: `corex_issue_columns` selects on it.

    Unspecified fields take one shared default across every row — see the
    module docstring's fixture convention before relying on one.
    """
    rows = []
    for spec in specs:
        hits = spec.get("hits", {})
        for i in range(spec["n"]):
            rows.append({
                "doc_name": spec["doc"],
                "para_idx": i,
                "year": spec["year"],
                "president": spec.get("president", "A President"),
                **{name: bool(i < hits.get(name, 0)) for name in issues},
            })
    out = pd.DataFrame(rows)
    for name in issues:
        out[name] = out[name].astype(bool)
    return out


def _sampling(rows: list[dict]) -> pd.DataFrame:
    """A stand-in for `attention.bootstrap_era_shares`'s output.

    Unspecified fields take one shared default across every row — see the
    module docstring's fixture convention before relying on one.
    """
    return pd.DataFrame([
        {
            "topic": r["topic"],
            "era": r["era"],
            "share": r.get("share", 0.20),
            "share_lo": r.get("lo", 0.10),
            "share_hi": r.get("hi", 0.30),
            "n_paragraphs": r.get("n_paragraphs", 500),
            "n_topic_paragraphs": r.get("n_topic_paragraphs", 100),
        }
        for r in rows
    ])


def _half_widths(rows: list[dict]) -> pd.DataFrame:
    """A stand-in for `disagreement_half_widths`'s output.

    The two share columns are DERIVED from `hw` rather than defaulted
    independently, so every row satisfies the invariant the real function
    guarantees::

        disagreement_half_width == |share_primary - share_secondary| / 2

    Defaulting all three to flat scalars (0.2, 0.2, 0.02) made that invariant
    false on every row. It was inert only because `llm_bands` never reads the
    share columns — the first test to assert the relationship against this
    fixture would have been asserting against a contradiction. The secondary is
    held at 0.20 and the primary carries the whole gap, which keeps both shares
    inside [0, 1] for every `hw` this suite uses.

    CONSEQUENCE, and it is a real limit: the derived gap is ONE-DIRECTIONAL —
    `share_primary >= share_secondary` on every default row — so this fixture
    can never model "the secondary annotator read MORE". That direction is not
    lost, it is just covered elsewhere:
    `TestDisagreementHalfWidths.test_half_width_is_symmetric_in_which_annotator_read_more`
    drives the real function over `_paired` fixtures at (40, 30) and (30, 40)
    and pins that `abs()` makes both give 0.05. A test that needs the negative
    direction HERE must pass `share_primary`/`share_secondary` explicitly.

    Other unspecified fields take one shared default across every row — see the
    module docstring's fixture convention before relying on one.
    """
    out = []
    for r in rows:
        hw = r.get("hw", 0.02)
        out.append({
            "topic": r["topic"],
            "era": r["era"],
            "share_primary": r.get("share_primary", 0.20 + 2 * hw),
            "share_secondary": r.get("share_secondary", 0.20),
            "disagreement_half_width": hw,
            "n_paired_paragraphs": r.get("n_paired", 500),
            "available": r.get("available", True),
        })
    return pd.DataFrame(out)


def _drive_llm_bands(monkeypatch, sampling: pd.DataFrame, hw: pd.DataFrame,
                     n_docs: int | dict[str, int] = 40) -> pd.DataFrame:
    """Run the real `llm_bands` composition over injected components.

    The sampling bootstrap and the paired-annotator half-widths are the two
    INPUTS; everything `llm_bands` itself does — the merge guard, the three-way
    status, the widen-never-narrow composition, the clip — runs for real.

    `n_docs` may be a per-era dict, so a test can tell whether each era's speech
    count reached its OWN row rather than merely reaching some row.
    """
    eras = sorted(set(sampling["era"]))
    counts = n_docs if isinstance(n_docs, dict) else {e: n_docs for e in eras}
    paragraphs = pd.DataFrame([
        {"doc_name": f"{era}-doc{k}", "era": era}
        for era in eras for k in range(counts[era])
    ])
    monkeypatch.setattr(
        attention, "load_inputs", lambda taxonomy=None: (paragraphs, pd.DataFrame())
    )
    monkeypatch.setattr(
        attention, "bootstrap_era_shares",
        lambda *a, **k: sampling.copy(),
    )
    monkeypatch.setattr(B, "disagreement_half_widths", lambda *a, **k: hw.copy())
    return B.llm_bands(taxonomy={"level2": [{"name": t} for t in sampling["topic"]]})


# --------------------------------------------------------------------------- #
# 1. the cluster floor
# --------------------------------------------------------------------------- #
class TestCiStatusFloor:
    def test_floor_constants_are_pinned_as_literals(self):
        """Anchored as literals, not read off the module: a test parametrized
        over the constant would follow it wherever it drifted. These three
        decide which periods the site is allowed to draw a band on, and
        `MIN_PERIOD_PARAGRAPHS = 40` is the charts' former hard mask."""
        assert B.MIN_CLUSTERS_FOR_CI == 2
        assert B.LOW_CLUSTER_CAUTION == 8
        assert B.MIN_PERIOD_PARAGRAPHS == 40
        assert B.PERIOD_YEARS == 5

    @pytest.mark.parametrize(
        "n_speeches,n_paragraphs,expected",
        [
            (0, 0, "suppressed_n_floor"),
            (1, 900, "suppressed_n_floor"),      # boundary: last suppressed
            (2, 900, "low_cluster_caution"),     # boundary: first interval
            (7, 900, "low_cluster_caution"),     # boundary: last thin-by-speech
            (8, 900, "ok"),                      # boundary: first unflagged
            (8, 40, "ok"),                       # boundary: paragraph floor met
            (8, 39, "low_cluster_caution"),      # boundary: paragraph floor missed
            (80, 39, "low_cluster_caution"),     # either floor alone flags
        ],
    )
    def test_status_boundaries(self, n_speeches, n_paragraphs, expected):
        assert B._ci_status(n_speeches, n_paragraphs) == expected


# --------------------------------------------------------------------------- #
# 2. the Surface A bootstrap
# --------------------------------------------------------------------------- #
class TestCorexBootstrap:
    def test_point_estimate_is_the_pooled_share_within_the_period(self):
        """Unweighted pooled share, not a mean of per-speech shares: 3 of 20
        paragraphs, regardless of how they split across the two speeches."""
        labels = _labels([
            {"doc": "a", "year": 1900, "n": 15, "hits": {"war": 3}},
            {"doc": "b", "year": 1902, "n": 5, "hits": {"war": 0}},
        ])
        out = B.bootstrap_corex_periods(labels, ["war", "trade"], n_draws=50)
        war = out[(out["series"] == "war") & (out["period"] == "1900")].iloc[0]
        assert war["point"] == pytest.approx(3 / 20)
        assert war["n_paragraphs"] == 20
        assert war["n_speeches"] == 2

    def test_periods_are_five_year_buckets_floored_to_the_period_start(self):
        labels = _labels([
            {"doc": "a", "year": 1904, "n": 10, "hits": {"war": 5}},
            {"doc": "b", "year": 1905, "n": 10, "hits": {"war": 5}},
        ])
        out = B.bootstrap_corex_periods(labels, ["war"], n_draws=20)
        assert sorted(out["period"]) == ["1900", "1905"]
        row = out[out["period"] == "1900"].iloc[0]
        assert (row["period_start"], row["period_end"], row["x"]) == (1900, 1904, 1900)

    def test_single_speech_period_publishes_no_interval_rather_than_a_zero_one(self):
        """A bootstrap over one cluster resamples the same speech every draw and
        returns a zero-width interval — a lie in exactly the wrong direction.
        The row is still published (the point estimate and its n survive); only
        the interval is withheld."""
        labels = _labels([
            {"doc": "solo", "year": 1785, "n": 14, "hits": {"war": 7}},
            {"doc": "a", "year": 1900, "n": 60, "hits": {"war": 6}},
            {"doc": "b", "year": 1901, "n": 60, "hits": {"war": 6}},
            {"doc": "c", "year": 1902, "n": 60, "hits": {"war": 6}},
        ])
        out = B.bootstrap_corex_periods(labels, ["war"], n_draws=50)
        solo = out[out["period"] == "1785"].iloc[0]
        assert solo["ci_status"] == "suppressed_n_floor"
        assert np.isnan(solo["lo_sampling"]) and np.isnan(solo["hi_sampling"])
        assert solo["point"] == pytest.approx(0.5)
        assert solo["n_speeches"] == 1

    def test_a_two_speech_period_produces_a_far_wider_band_than_a_dense_one(self):
        """The verification strategy's headline check: identical per-speech
        composition, two speeches vs thirty, and the interval must reflect it.
        If this stops holding the bootstrap is not clustering on speeches.

        The speeches deliberately DISAGREE with each other (alternating all-hit
        and no-hit). A fixture where every speech had the same share would make
        every resample identical and both widths zero — the test would pass its
        own premise vacuously while measuring nothing.
        """
        thin = [
            {"doc": "t0", "year": 1785, "n": 10, "hits": {"war": 10}},
            {"doc": "t1", "year": 1786, "n": 10, "hits": {"war": 0}},
        ]
        dense = [
            {"doc": f"d{k}", "year": 1900, "n": 10, "hits": {"war": 4 + k % 3}}
            for k in range(30)
        ]
        out = B.bootstrap_corex_periods(_labels(thin + dense), ["war"])
        width = out.set_index("period").eval("hi_sampling - lo_sampling")
        assert width["1900"] > 0, "dense fixture must not be degenerate either"
        assert width["1785"] > 5 * width["1900"]

    def test_identical_inputs_give_identical_intervals(self):
        labels = _labels([
            {"doc": f"d{k}", "year": 1900 + k, "n": 8, "hits": {"war": k % 4}}
            for k in range(10)
        ])
        first = B.bootstrap_corex_periods(labels, ["war", "trade"])
        second = B.bootstrap_corex_periods(labels, ["war", "trade"])
        pd.testing.assert_frame_equal(first, second)

    def test_adding_a_period_does_not_move_another_periods_interval(self):
        """The per-period seeding claim (`default_rng([seed, period])`) made
        testable: with a single global RNG consumed in period order, inserting a
        1700s period would shift every later period's draws. Here it must not
        move so much as a float."""
        base = [
            {"doc": f"d{k}", "year": 1900 + k, "n": 9, "hits": {"war": k % 3}}
            for k in range(6)
        ]
        extra = [{"doc": "early", "year": 1800, "n": 9, "hits": {"war": 4}},
                 {"doc": "early2", "year": 1801, "n": 9, "hits": {"war": 1}}]
        without = B.bootstrap_corex_periods(_labels(base), ["war"])
        with_extra = B.bootstrap_corex_periods(_labels(extra + base), ["war"])
        shared = sorted(set(without["period"]))
        pd.testing.assert_frame_equal(
            without[without["period"].isin(shared)].reset_index(drop=True),
            with_extra[with_extra["period"].isin(shared)].reset_index(drop=True),
        )

    def test_shares_are_fractions_not_percentages(self):
        """`bands_meta.json` states the units; the charts multiply by 100. A
        frame that had already been scaled would double-scale on the site."""
        # Four speeches, not two: at n=2 every speech agreeing at 1.0 is a
        # DEGENERATE interval and is now withdrawn to NaN, which would make
        # this units test assert on nothing. `MIN_CLUSTERS_FOR_RESOLVABLE_CI`
        # speeches keep a real published bound to check the scale of.
        labels = _labels([
            {"doc": d, "year": 1900 + i, "n": 50, "hits": {"war": 50}}
            for i, d in enumerate("abcd")
        ])
        out = B.bootstrap_corex_periods(labels, ["war"], n_draws=50)
        assert not out["interval_unresolvable"].any()
        assert out["point"].max() == pytest.approx(1.0)
        assert out["hi_sampling"].max() <= 1.0

    def test_only_boolean_columns_are_treated_as_issues(self):
        labels = _labels([{"doc": "a", "year": 1900, "n": 4, "hits": {"war": 2}}])
        labels["coherence"] = 0.5
        assert B.corex_issue_columns(labels) == ["war", "trade"]


# --------------------------------------------------------------------------- #
# 2b. `interval_unresolvable` — the per-cell degeneracy gate
# --------------------------------------------------------------------------- #
class TestResolvableFloorDerivation:
    """`MIN_CLUSTERS_FOR_RESOLVABLE_CI` is claimed to be DERIVED from `CI_LOW`.

    A single assertion that it equals 4 cannot tell a derivation from a
    hardcoded 4 that happens to agree — `_min_clusters_for_resolvable_ci` could
    `return 4` and pass. So the derivation is exercised at several percentiles,
    which is the only shape of test that distinguishes the two.
    """

    def test_the_module_constant_is_the_derivation_not_a_literal(self):
        assert B.MIN_CLUSTERS_FOR_RESOLVABLE_CI == B._min_clusters_for_resolvable_ci(
            B.CI_LOW)
        assert B.MIN_CLUSTERS_FOR_RESOLVABLE_CI == 4

    @pytest.mark.parametrize(
        "ci_low,expected",
        [
            # n**-n first drops below ci_low/100 at:
            (50.0, 2),   # 0.25 < 0.50 already at n=2
            # EXACT-EQUALITY BOUNDARY: 2**-2 == 25.0/100 to the bit. The loop
            # condition is strict (`>`), so a probability sitting exactly ON the
            # percentile counts as excludable and n=2 qualifies. Pinned because
            # `>` vs `>=` is invisible at every other percentile in this table.
            (25.0, 2),
            (5.0, 3),    # 0.25 > 0.05; 0.037 < 0.05
            (2.5, 4),    # 0.037 > 0.025; 0.0039 < 0.025   <- today's CI_LOW
            (0.1, 5),    # 0.0039 > 0.001; 0.00032 < 0.001
        ],
    )
    def test_the_floor_tracks_the_percentile_it_is_derived_from(self, ci_low, expected):
        """Four percentiles, four different floors. A function returning a
        constant — or one reading `CI_LOW` instead of its argument — fails on
        every row but the third."""
        assert B._min_clusters_for_resolvable_ci(ci_low=ci_low) == expected

    def test_the_derivation_matches_its_own_stated_premise(self):
        """The premise is `n**-n <= ci_low/100`. Asserted directly, so the
        docstring's arithmetic (0.25 / 0.037 / 0.0039) and the returned floor
        cannot drift apart."""
        n = B.MIN_CLUSTERS_FOR_RESOLVABLE_CI
        assert n ** (-n) <= B.CI_LOW / 100.0, "the floor itself must qualify"
        assert (n - 1) ** (-(n - 1)) > B.CI_LOW / 100.0, "and be the FIRST that does"

    def test_ci_low_is_positive(self):
        """The module constant, still pinned as the first line of defence.

        This used to be the ONLY guard: the derivation loop had no cap, so
        `ci_low = 0` never terminated (`n**-n > 0` is always true) and the test
        deliberately declined to call the function — a test that hangs is worse
        than the defect it documents. That reasoning was right, and the guard is
        kept, but the function now rejects a non-positive percentile itself
        (below), so the hang is no longer reachable to begin with.
        """
        assert B.CI_LOW > 0
        assert B.CI_HIGH > B.CI_LOW

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_a_non_positive_percentile_raises_instead_of_looping_forever(self, bad):
        """The guard that makes the hang unreachable — now safe to execute.

        NaN is included and is the more dangerous of the two shapes. Zero and
        negatives fail loudly (the old loop simply never exited); NaN fails
        SILENTLY, because every comparison against NaN is False, so
        `while n ** (-n) > nan` exits on its first test and the function returns
        `MIN_CLUSTERS_FOR_CI` — a plausible-looking floor derived from nothing
        at all. The `not (ci_low > 0)` spelling is what catches all three; a
        `ci_low <= 0` check would let NaN straight through.
        """
        with pytest.raises(ValueError) as err:
            B._min_clusters_for_resolvable_ci(bad)
        assert "positive percentile" in str(err.value)

    def test_the_iteration_cap_is_belt_and_braces_not_a_working_limit(self):
        """The cap must never bind for a percentile anyone could plausibly set.

        `n**-n` first evaluates to exactly 0.0 at n=149, and 0.0 is not greater
        than any positive threshold, so the search self-terminates far below
        `_MAX_CLUSTER_SEARCH` even at absurd percentiles. Asserted rather than
        assumed: a cap that silently became the thing deciding the floor would
        publish a derived-looking number that is really just the loop bound.

        The underflow point is RE-DERIVED here and pinned into the comment that
        cites it. Asserting only `< _MAX_CLUSTER_SEARCH` is what let that
        comment ship saying 178: the conclusion holds for any value under 1000,
        so a wrong n was invisible to the assertion while remaining the sole
        quantitative justification for the constant.
        """
        n = 2
        while n ** (-n) > 0.0:
            n += 1
        assert n == 149
        assert (n - 1) ** (-(n - 1)) > 0.0, "n=148 must still be nonzero"
        assert f"n={n}" in inspect.getsource(B), (
            f"the comment justifying _MAX_CLUSTER_SEARCH must cite n={n}")
        assert B._MAX_CLUSTER_SEARCH > 100
        for ci_low in (1e-10, 1e-100, 5e-300):
            assert B._min_clusters_for_resolvable_ci(ci_low) < B._MAX_CLUSTER_SEARCH

    @pytest.mark.parametrize(
        "n,expected", [(2, 3), (3, 10), (4, 35), (5, 126), (10, 92_378)])
    def test_distinct_resample_counts(self, n, expected):
        """`C(2n-1, n)`, the premise behind the whole gate: below a handful of
        clusters the estimator cannot express uncertainty even in principle."""
        assert B.distinct_resamples(n) == expected


class TestUnresolvableGate:
    """The conjunction — "no variation" AND "too few clusters" — from both sides.

    Every fixture states its own speech count explicitly rather than leaning on
    a helper default, because the two sides of this gate are distinguished by
    exactly that number.
    """

    @staticmethod
    def _run(specs, issues=("war", "trade")):
        return B.bootstrap_corex_periods(
            _labels(specs, issues=issues), list(issues), n_draws=200)

    @staticmethod
    def _cell(out, series, period):
        return out[(out["series"] == series) & (out["period"] == period)].iloc[0]

    @pytest.mark.parametrize("n_docs,flagged", [(3, True), (4, False)])
    def test_the_floor_binds_on_both_sides(self, n_docs, flagged):
        """n=3 is withdrawn, n=4 is published — the boundary
        `MIN_CLUSTERS_FOR_RESOLVABLE_CI` names. Both arms run the SAME all-zero
        data, so the cluster count is the only thing that differs."""
        out = self._run([
            {"doc": f"d{k}", "year": 1900 + k, "n": 10, "hits": {"war": 0}}
            for k in range(n_docs)
        ])
        row = self._cell(out, "war", "1900")
        assert row["n_speeches"] == n_docs
        assert bool(row["interval_unresolvable"]) is flagged
        if flagged:
            assert np.isnan(row["lo_sampling"]) and np.isnan(row["hi_sampling"])
        else:
            assert row["lo_sampling"] == 0.0 and row["hi_sampling"] == 0.0

    def test_a_high_n_all_zero_series_keeps_its_confident_zero(self):
        """SYNTHETIC control, independent of the corpus. Twenty speeches that
        genuinely never touch an issue are a finding, and the gate must not
        touch them. The in-corpus evidence (7 such cells at 13-14 speeches) says
        the same thing but is hostage to the data; this says it structurally."""
        out = self._run([
            {"doc": f"d{k}", "year": 1900, "n": 25, "hits": {"war": 0}}
            for k in range(20)
        ])
        row = self._cell(out, "war", "1900")
        assert row["n_speeches"] == 20
        assert not bool(row["interval_unresolvable"])
        assert row["lo_sampling"] == 0.0 and row["hi_sampling"] == 0.0
        assert row["point"] == 0.0

    def test_a_flagged_cell_may_have_a_NON_ZERO_point(self):
        """Every flagged cell in today's corpus sits at `point == 0`, which is
        an accident of the data, not a property of the gate: three speeches
        agreeing exactly at 50% are just as unresolvable as three agreeing at
        zero. The point estimate survives — the share was measured, only the
        interval was not.

        This also covers the empirical premise behind `unresolved_traces`'
        `cliponaxis=False` comment ("every flagged cell has point == 0"), which
        would silently stop holding here.
        """
        out = self._run([
            {"doc": f"d{k}", "year": 1900, "n": 10, "hits": {"war": 5}}
            for k in range(3)
        ])
        row = self._cell(out, "war", "1900")
        assert bool(row["interval_unresolvable"])
        assert row["point"] == pytest.approx(0.5)
        assert np.isnan(row["lo_sampling"]) and np.isnan(row["hi_sampling"])

    def test_variation_at_the_same_low_n_is_not_flagged(self):
        """The Religion & values control, synthetically: two speeches that
        DISAGREE resolve a real (enormous) width at n=2, and suppressing it
        would delete the loudest uncertainty signal the module can draw."""
        out = self._run([
            {"doc": "a", "year": 1900, "n": 10, "hits": {"war": 10}},
            {"doc": "b", "year": 1901, "n": 10, "hits": {"war": 0}},
        ])
        row = self._cell(out, "war", "1900")
        assert row["n_speeches"] == 2
        assert not bool(row["interval_unresolvable"])
        assert row["hi_sampling"] > row["lo_sampling"]

    def test_the_gate_is_per_cell_not_per_period(self):
        """One period, two series, one flagged and one not — the property
        `ci_status` structurally cannot express, and the reason this had to be a
        new column rather than a third `ci_status` value."""
        out = self._run([
            {"doc": "a", "year": 1900, "n": 10, "hits": {"war": 10, "trade": 0}},
            {"doc": "b", "year": 1901, "n": 10, "hits": {"war": 0, "trade": 0}},
        ])
        war, trade = self._cell(out, "war", "1900"), self._cell(out, "trade", "1900")
        assert war["ci_status"] == trade["ci_status"], "same period, same ci_status"
        assert not bool(war["interval_unresolvable"])
        assert bool(trade["interval_unresolvable"])

    def test_a_suppressed_n_floor_cell_is_not_ALSO_flagged(self):
        """Two columns must not claim the same suppression. Where no bootstrap
        ran at all, `ci_status` already names the cause and the bounds are
        already NaN; flagging it too would tell a consumer the bootstrap ran and
        resolved nothing, which is a different (and false) story."""
        out = self._run([
            {"doc": "solo", "year": 1785, "n": 14, "hits": {"war": 0}},
            {"doc": "a", "year": 1900, "n": 10, "hits": {"war": 0}},
            {"doc": "b", "year": 1901, "n": 10, "hits": {"war": 1}},
        ])
        row = self._cell(out, "war", "1785")
        assert row["ci_status"] == "suppressed_n_floor"
        assert not bool(row["interval_unresolvable"])
        assert np.isnan(row["lo_sampling"])

    def test_the_flag_never_moves_the_point_estimate(self):
        """Withdrawn interval, retained measurement — asserted against the
        pooled share computed by hand, not against the function's own output."""
        out = self._run([
            {"doc": f"d{k}", "year": 1900, "n": 8, "hits": {"war": 2}}
            for k in range(3)
        ])
        row = self._cell(out, "war", "1900")
        assert bool(row["interval_unresolvable"])
        assert row["point"] == pytest.approx(6 / 24)


class TestUnresolvableIntervalPredicate:
    """`_unresolvable_interval` directly, including the NaN case that keeps the
    two suppression columns from overlapping."""

    def test_below_the_floor_zero_width_cells_are_flagged_elementwise(self):
        lo = np.array([0.0, 0.1, 0.5])
        hi = np.array([0.0, 0.4, 0.5])
        mask = B._unresolvable_interval(lo, hi, n_clusters=2)
        assert list(mask) == [True, False, True]

    def test_at_or_above_the_floor_nothing_is_flagged(self):
        lo = hi = np.array([0.0, 0.5])
        for n in (B.MIN_CLUSTERS_FOR_RESOLVABLE_CI,
                  B.MIN_CLUSTERS_FOR_RESOLVABLE_CI + 40):
            assert not B._unresolvable_interval(lo, hi, n_clusters=n).any()

    def test_a_nan_bound_is_never_flagged(self):
        """`nan == nan` is False, so the `suppressed_n_floor` rows fall out
        naturally — but naturally is not the same as tested, and a future
        rewrite using `np.isclose` or a fillna would flag them all."""
        nan = np.array([np.nan, np.nan])
        assert not B._unresolvable_interval(nan, nan, n_clusters=2).any()

    def test_the_mask_is_boolean_and_shaped_like_its_input(self):
        mask = B._unresolvable_interval(np.zeros(5), np.zeros(5), n_clusters=99)
        assert mask.dtype == bool and mask.shape == (5,)


# --------------------------------------------------------------------------- #
# 3. Surface A is exactly sampling-only
# --------------------------------------------------------------------------- #
class TestSurfaceAIsSamplingOnly:
    @pytest.fixture
    def surface_a(self) -> pd.DataFrame:
        labels = _labels(
            [{"doc": "solo", "year": 1785, "n": 12, "hits": {"war": 6}}]
            + [{"doc": f"d{k}", "year": 1900 + (k % 5), "n": 20,
                "hits": {"war": k % 6, "trade": 1}} for k in range(12)]
        )
        return B.corex_bands(labels)

    def test_composed_interval_is_the_sampling_interval_column_for_column(self, surface_a):
        """`equals`, not `allclose`: the composed columns must be the sampling
        columns, NaNs and all. Any drift means an annotator component that does
        not apply to a deterministic topic model found its way in."""
        assert surface_a["lo"].equals(surface_a["lo_sampling"])
        assert surface_a["hi"].equals(surface_a["hi_sampling"])

    def test_provenance_says_not_applicable_not_missing(self, surface_a):
        assert (surface_a["ci_components"] == "sampling_only").all()
        assert (surface_a["disagreement_status"] == B.NOT_APPLICABLE).all()
        assert B.NOT_APPLICABLE == "not_applicable_no_annotator_in_pipeline"
        assert not surface_a["disagreement_band_applied"].any()
        assert surface_a["disagreement_half_width"].isna().all()
        assert surface_a["n_paired_paragraphs"].isna().all()
        assert (surface_a["agreement_source"] == B.COREX_AGREEMENT_SOURCE).all()
        assert "n/a" in B.COREX_AGREEMENT_SOURCE

    def test_agreement_v1_parquet_is_never_read(self, real_annotations, monkeypatch):
        """The architecture correction, asserted behaviourally rather than by
        scanning for a string (the meta sidecar names the file precisely to say
        it is NOT read). A jaccard on the ERA_SPAN sampling axis has no units
        convertible to percentage points of share, so widening a band with it
        would attach a measured magnitude to a quantity it does not describe.

        Every parquet the real builder opens is recorded; `agreement_v1` must
        not be among them.
        """
        seen: list[str] = []
        real_read = pd.read_parquet

        def recording(path, *a, **k):
            seen.append(str(path))
            return real_read(path, *a, **k)

        monkeypatch.setattr(pd, "read_parquet", recording)
        B.build_bands()
        assert seen, "recorder never fired — the builder read no parquet at all"
        assert not [p for p in seen if "agreement_v1" in p], seen
        # The tables it DOES read, named so a silent swap is visible.
        assert any("paragraph_issues" in p for p in seen)
        assert any("paragraph_annotations__opus4-8" in p for p in seen)


# --------------------------------------------------------------------------- #
# 4. the annotator half-width
# --------------------------------------------------------------------------- #
class TestDisagreementHalfWidths:
    TAXONOMY = {"level2": [{"name": "Trade"}, {"name": "War"}]}

    def _paired(self, era: str, n: int, primary_hits: int, secondary_hits: int):
        return pd.DataFrame([
            {
                "doc_name": f"d{i // 10}",
                "para_idx": i,
                "era": era,
                "topics_primary": frozenset({"Trade"}) if i < primary_hits else frozenset(),
                "topics_secondary": frozenset({"Trade"}) if i < secondary_hits else frozenset(),
            }
            for i in range(n)
        ])

    def test_half_width_is_half_the_gap_between_the_two_annotators(self):
        paired = self._paired(ERA_A, n=100, primary_hits=40, secondary_hits=30)
        hw = B.disagreement_half_widths(self.TAXONOMY, paired)
        row = hw[(hw["era"] == ERA_A) & (hw["topic"] == "Trade")].iloc[0]
        assert row["share_primary"] == pytest.approx(0.40)
        assert row["share_secondary"] == pytest.approx(0.30)
        assert row["disagreement_half_width"] == pytest.approx(0.05)
        assert row["n_paired_paragraphs"] == 100
        assert bool(row["available"])

    def test_perfect_agreement_gives_a_zero_half_width_not_a_missing_one(self):
        paired = self._paired(ERA_A, n=100, primary_hits=25, secondary_hits=25)
        hw = B.disagreement_half_widths(self.TAXONOMY, paired)
        row = hw[(hw["era"] == ERA_A) & (hw["topic"] == "Trade")].iloc[0]
        assert row["disagreement_half_width"] == 0.0
        assert bool(row["available"])

    def test_half_width_is_symmetric_in_which_annotator_read_more(self):
        """Fixture asymmetry on purpose: `abs()` removed would flip this sign."""
        more = B.disagreement_half_widths(
            self.TAXONOMY, self._paired(ERA_A, 100, 40, 30))
        less = B.disagreement_half_widths(
            self.TAXONOMY, self._paired(ERA_A, 100, 30, 40))
        pick = lambda f: f[(f["era"] == ERA_A) & (f["topic"] == "Trade")].iloc[0]
        assert pick(more)["disagreement_half_width"] == pytest.approx(0.05)
        assert pick(less)["disagreement_half_width"] == pytest.approx(0.05)

    @pytest.mark.parametrize("n,available", [(49, False), (50, True)])
    def test_thin_paired_sample_withholds_the_estimate_at_the_floor(self, n, available):
        """`MIN_PAIRED_PARAGRAPHS` is a live guard, not a formality: below it the
        half-width is NaN rather than an estimate off a handful of rows.

        Both arms assert on the VALUE, not merely on the flag. The withheld arm
        would otherwise be the only place in the suite where the NaN-ing itself
        is exercised: `TestThreeWayDisagreementStatus` reaches `available=False`
        through the fixture flag rather than through the floor, so it never runs
        this branch.

        Written as an explicit if/else rather than
        `np.isnan(...) is not available` — `np.isnan` returns `np.bool_`, which
        is never the `True`/`False` singleton, so that identity comparison is
        true for every combination of both operands and cannot fail.
        """
        assert B.MIN_PAIRED_PARAGRAPHS == 50
        paired = self._paired(ERA_A, n=n, primary_hits=n // 2, secondary_hits=0)
        hw = B.disagreement_half_widths(self.TAXONOMY, paired)
        row = hw[(hw["era"] == ERA_A) & (hw["topic"] == "Trade")].iloc[0]
        half_width = row["disagreement_half_width"]
        assert bool(row["available"]) is available
        if available:
            # 25 of 50 vs 0 of 50 -> |0.5 - 0.0| / 2
            assert half_width == pytest.approx(0.25)
        else:
            assert bool(np.isnan(half_width)), (
                f"below the floor the half-width must be withheld, got {half_width!r} "
                "— an estimate off 49 paragraphs published as if it were measured"
            )

    def test_every_era_gets_a_row_for_every_topic_even_with_no_paired_rows(self):
        """Sourced from the taxonomy and `trends.ERAS`, not from observed
        labels: a topic that vanished from an era must still produce a row, or
        the left merge in `llm_bands` would lose it and raise."""
        hw = B.disagreement_half_widths(self.TAXONOMY, self._paired(ERA_A, 100, 40, 30))
        assert len(hw) == 9 * 2
        empty = hw[(hw["era"] == ERA_B) & (hw["topic"] == "War")].iloc[0]
        assert empty["n_paired_paragraphs"] == 0
        assert not bool(empty["available"])
        assert np.isnan(empty["share_primary"])


class TestPairedAnnotationsGuards:
    """`paired_annotations` driven hermetically over four-row annotation tables.

    Three keyed merges and one era assignment stand between the two frozen
    annotator parquets and a per-era share, and every one of them fails
    SILENTLY if unguarded: an inner join drops non-matching rows, and a year
    outside `trends.ERAS` lands as NaN, which reads downstream as the
    whole-corpus row rather than as an error (`triangulate.assign_eras`'s
    documented trap).
    """

    TAXONOMY = {"level2": [{"name": "Trade & Tariffs"}, {"name": "War of 1812"}]}

    @pytest.fixture
    def install(self, monkeypatch, tmp_path):
        from presidential_profiles import issues

        def _install(primary: list[dict], secondary: list[dict],
                     years: list[dict]) -> None:
            tables = {
                B.PRIMARY_ANNOTATIONS: pd.DataFrame(primary),
                B.SECONDARY_ANNOTATIONS: pd.DataFrame(secondary),
            }
            monkeypatch.setattr(
                B, "load_paragraph_annotations", lambda name: tables[name].copy())
            path = tmp_path / "paragraph_issues.parquet"
            pd.DataFrame(years).to_parquet(path)
            monkeypatch.setattr(issues, "PARA_LABELS_PATH", path)

        return _install

    @staticmethod
    def _rows(keys, topics):
        return [{"doc_name": d, "para_idx": i, "topics": t}
                for (d, i), t in zip(keys, topics)]

    def test_the_paired_set_is_the_intersection_with_canonical_topic_sets(self, install):
        """Case drift ("trade & tariffs") is resolved to the canonical name, and
        the primary's extra full-corpus rows are dropped without complaint —
        only the SECONDARY side must survive whole.

        Both input frames arrive in SCRAMBLED key order, so the returned frame's
        `(doc_name, para_idx)` ordering has to come from the sort rather than
        from the inputs happening to line up — the positional-alignment trap
        this repo replaced a real bug over.
        """
        keys = [("a", 0), ("a", 1), ("b", 0)]
        install(
            primary=self._rows([("b", 0), ("c", 0), ("a", 0), ("a", 1)],
                               [["War Of 1812"], ["War of 1812"], ["trade & tariffs"], []]),
            secondary=self._rows([("b", 0), ("a", 1), ("a", 0)],
                                 [[], ["War of 1812"], ["Trade & Tariffs"]]),
            years=[{"doc_name": d, "para_idx": i, "year": y}
                   for (d, i), y in zip([("c", 0), ("a", 1), ("b", 0), ("a", 0)],
                                        [1900, 1800, 1850, 1800])],
        )
        paired = B.paired_annotations(self.TAXONOMY)
        assert len(paired) == 3
        assert paired.iloc[0]["topics_primary"] == frozenset({"Trade & Tariffs"})
        assert paired.iloc[0]["topics_secondary"] == frozenset({"Trade & Tariffs"})
        assert paired.iloc[2]["topics_primary"] == frozenset({"War of 1812"})
        assert list(paired["era"]) == [
            "The founding", "The founding", "Civil War & Reconstruction"]
        assert list(zip(paired["doc_name"], paired["para_idx"])) == keys

    def test_a_secondary_row_the_primary_never_labeled_raises(self, install):
        """The inner join would silently drop it and every downstream share
        would be computed over a smaller denominator than the one reported."""
        keys = [("a", 0), ("a", 1)]
        install(
            primary=self._rows([("a", 0)], [[]]),
            secondary=self._rows(keys, [[], []]),
            years=[{"doc_name": d, "para_idx": i, "year": 1800} for d, i in keys],
        )
        with pytest.raises(ValueError) as err:
            B.paired_annotations(self.TAXONOMY)
        assert "paired annotator tables" in str(err.value)

    def test_a_paired_paragraph_with_no_year_row_raises(self, install):
        keys = [("a", 0), ("a", 1)]
        install(
            primary=self._rows(keys, [[], []]),
            secondary=self._rows(keys, [[], []]),
            years=[{"doc_name": "a", "para_idx": 0, "year": 1800}],
        )
        with pytest.raises(ValueError) as err:
            B.paired_annotations(self.TAXONOMY)
        assert "paired x paragraph years" in str(err.value)

    def test_a_year_outside_the_named_eras_raises_instead_of_landing_as_nan(
        self, install
    ):
        """NaN is not "unknown era" downstream — on a per-era table it reads as
        the whole-corpus row. Refuse rather than let it through."""
        keys = [("a", 0), ("a", 1)]
        install(
            primary=self._rows(keys, [[], []]),
            secondary=self._rows(keys, [[], []]),
            years=[{"doc_name": "a", "para_idx": 0, "year": 1600},
                   {"doc_name": "a", "para_idx": 1, "year": 1800}],
        )
        with pytest.raises(ValueError) as err:
            B.paired_annotations(self.TAXONOMY)
        assert "1600" in str(err.value)
        assert "trends.ERAS" in str(err.value)

    def test_an_unmappable_topic_label_raises_rather_than_being_dropped(self, install):
        """The taxonomy and the annotations having diverged is a hard error, not
        a warning: a dropped label silently deflates that topic's share."""
        keys = [("a", 0)]
        install(
            primary=self._rows(keys, [["Cryptocurrency"]]),
            secondary=self._rows(keys, [[]]),
            years=[{"doc_name": "a", "para_idx": 0, "year": 1800}],
        )
        with pytest.raises(ValueError):
            B.paired_annotations(self.TAXONOMY)


# --------------------------------------------------------------------------- #
# 5. Surface B composition
# --------------------------------------------------------------------------- #
class TestSurfaceBComposition:
    def test_composition_widens_by_exactly_the_half_width(self, monkeypatch):
        out = _drive_llm_bands(
            monkeypatch,
            _sampling([{"topic": "Trade", "era": ERA_A, "lo": 0.10, "hi": 0.30}]),
            _half_widths([{"topic": "Trade", "era": ERA_A, "hw": 0.04}]),
        )
        row = out.iloc[0]
        assert row["lo"] == pytest.approx(0.06)
        assert row["hi"] == pytest.approx(0.34)
        assert row["lo_sampling"] == pytest.approx(0.10)
        assert row["hi_sampling"] == pytest.approx(0.30)
        assert row["ci_components"] == "sampling+annotator_disagreement"
        assert row["disagreement_status"] == B.MEASURED
        assert bool(row["disagreement_band_applied"])

    def test_composition_never_narrows_and_clips_to_the_unit_interval(self, monkeypatch):
        out = _drive_llm_bands(
            monkeypatch,
            _sampling([
                {"topic": "Trade", "era": ERA_A, "lo": 0.02, "hi": 0.98},
                {"topic": "War", "era": ERA_A, "lo": 0.40, "hi": 0.50},
                {"topic": "War", "era": ERA_B, "lo": 0.00, "hi": 1.00},
            ]),
            _half_widths([
                {"topic": "Trade", "era": ERA_A, "hw": 0.30},
                {"topic": "War", "era": ERA_A, "hw": 0.0},
                {"topic": "War", "era": ERA_B, "hw": 0.10},
            ]),
        )
        assert (out["lo"] <= out["lo_sampling"] + 1e-12).all()
        assert (out["hi"] >= out["hi_sampling"] - 1e-12).all()
        assert out["lo"].min() >= 0.0 and out["hi"].max() <= 1.0
        clipped = out[(out["series"] == "Trade")].iloc[0]
        assert clipped["lo"] == 0.0 and clipped["hi"] == 1.0

    def test_the_point_estimate_is_the_primary_annotators_share_untouched(self, monkeypatch):
        """The documented deflating caveat: the band is centred on the primary
        estimate, so composition must move the BOUNDS and never the point."""
        out = _drive_llm_bands(
            monkeypatch,
            _sampling([{"topic": "Trade", "era": ERA_A, "share": 0.23}]),
            _half_widths([{"topic": "Trade", "era": ERA_A, "hw": 0.05}]),
        )
        assert out.iloc[0]["point"] == pytest.approx(0.23)

    def test_the_trust_gate_reads_speeches_as_clusters_and_paragraphs_as_size(
        self, monkeypatch
    ):
        """The two n's are not interchangeable and the fixture is deliberately
        asymmetric on BOTH axes at once: era A has 40 speeches carrying 10
        paragraphs (thin by paragraphs), era B has 12 speeches carrying 400
        (ample). Swapping the two ARGUMENTS of `_ci_status` would publish A as
        `ok`, which the status assertions catch.

        Swapping the era -> speech-count MAPPING is a different defect, and the
        statuses cannot see it: `_ci_status(12, 10)` is still
        `low_cluster_caution` and `_ci_status(40, 400)` is still `ok`, so both
        rows would report exactly what they report today. What catches that one
        is the pair of explicit `n_speeches` assertions below — each era's own
        cluster count on its own row, not merely "some count reached some row".
        """
        out = _drive_llm_bands(
            monkeypatch,
            _sampling([{"topic": "Trade", "era": ERA_A, "n_paragraphs": 10},
                       {"topic": "Trade", "era": ERA_B, "n_paragraphs": 400}]),
            _half_widths([{"topic": "Trade", "era": ERA_A},
                          {"topic": "Trade", "era": ERA_B}]),
            n_docs={ERA_A: 40, ERA_B: 12},
        )
        by_era = out.set_index("period")
        assert by_era.loc[ERA_A, "ci_status"] == "low_cluster_caution"
        assert by_era.loc[ERA_B, "ci_status"] == "ok"
        # each era's own cluster count, not just "some count reached some row"
        assert by_era.loc[ERA_A, "n_speeches"] == 40
        assert by_era.loc[ERA_B, "n_speeches"] == 12

    def test_era_rows_carry_the_reporting_axis_bounds_and_order(self, monkeypatch):
        """`trends.ERAS`, not `taxonomy.ERA_SPAN` — and `x` is the era midpoint
        so Surface B shares the dashboard's year axis."""
        out = _drive_llm_bands(
            monkeypatch,
            _sampling([{"topic": "Trade", "era": ERA_A},
                       {"topic": "Trade", "era": ERA_B}]),
            _half_widths([{"topic": "Trade", "era": ERA_A},
                          {"topic": "Trade", "era": ERA_B}]),
        )
        from presidential_profiles.trends import ERAS

        bounds = {label: (lo, hi) for label, lo, hi in ERAS}
        for _, row in out.iterrows():
            lo, hi = bounds[row["period"]]
            assert (row["period_start"], row["period_end"]) == (lo, hi)
            assert row["x"] == pytest.approx((lo + hi) / 2)
        assert (out["period_kind"] == "era").all()
        first = out[out["period"] == ERA_A].iloc[0]
        second = out[out["period"] == ERA_B].iloc[0]
        assert first["period_order"] < second["period_order"]


class TestThreeWayDisagreementStatus:
    """Each precondition driven to fail INDEPENDENTLY, plus the precedence.

    `no_interval_to_widen` and `unavailable_thin_paired_sample` describe two
    genuinely different failures. Reporting "thin paired sample" for a cell
    whose paired sample was ample and whose *sampling* interval was the missing
    thing names the wrong cause — which is the whole reason a downstream
    consumer reads a status string instead of a boolean.
    """

    def _row(self, monkeypatch, *, available: bool, interval: bool) -> pd.Series:
        lo, hi = (0.1, 0.3) if interval else (np.nan, np.nan)
        out = _drive_llm_bands(
            monkeypatch,
            _sampling([{"topic": "Trade", "era": ERA_A, "lo": lo, "hi": hi}]),
            _half_widths([{"topic": "Trade", "era": ERA_A, "hw": 0.05,
                           "available": available}]),
        )
        return out.iloc[0]

    def test_both_preconditions_met_reports_measured(self, monkeypatch):
        row = self._row(monkeypatch, available=True, interval=True)
        assert row["disagreement_status"] == B.MEASURED == "measured_paired_annotators"

    def test_half_width_but_no_sampling_interval_reports_no_interval_to_widen(
        self, monkeypatch
    ):
        row = self._row(monkeypatch, available=True, interval=False)
        assert row["disagreement_status"] == B.NO_INTERVAL == "no_interval_to_widen"
        assert row["ci_components"] == "sampling_only"
        assert not bool(row["disagreement_band_applied"])
        assert np.isnan(row["disagreement_half_width"])

    def test_no_half_width_but_a_sampling_interval_reports_thin_paired_sample(
        self, monkeypatch
    ):
        row = self._row(monkeypatch, available=False, interval=True)
        assert row["disagreement_status"] == B.THIN_PAIRED
        assert B.THIN_PAIRED == "unavailable_thin_paired_sample"
        assert row["ci_components"] == "sampling_only"
        # The sampling interval survives untouched — a missing annotator
        # component must not delete the component that WAS measured.
        assert (row["lo"], row["hi"]) == (pytest.approx(0.1), pytest.approx(0.3))

    def test_one_missing_bound_is_no_interval_not_half_an_interval(self, monkeypatch):
        """`lo` present with `hi` absent is not a widenable interval. Under an
        `or` the row would be marked applied and `hi` would come back as
        NaN + half_width — a band with one live edge and one missing one, which
        `band_traces` then drops anyway, silently losing the row."""
        out = _drive_llm_bands(
            monkeypatch,
            _sampling([{"topic": "Trade", "era": ERA_A, "lo": 0.1, "hi": np.nan}]),
            _half_widths([{"topic": "Trade", "era": ERA_A, "hw": 0.05}]),
        )
        row = out.iloc[0]
        assert row["disagreement_status"] == B.NO_INTERVAL
        assert not bool(row["disagreement_band_applied"])
        assert row["ci_components"] == "sampling_only"

    def test_when_both_fail_the_thin_paired_sample_wins(self, monkeypatch):
        """Deliberate precedence: no half-width exists to apply whatever the
        sampling interval does, so the paired-sample cause is the more
        fundamental one. Pinned so it cannot silently flip to NO_INTERVAL."""
        row = self._row(monkeypatch, available=False, interval=False)
        assert row["disagreement_status"] == B.THIN_PAIRED

    def test_the_four_statuses_are_distinct_strings(self):
        statuses = {B.MEASURED, B.NO_INTERVAL, B.THIN_PAIRED, B.NOT_APPLICABLE}
        assert len(statuses) == 4

    def test_an_unresolvable_surface_b_cell_reports_both_the_root_and_the_proximate_cause(
        self, monkeypatch
    ):
        """DORMANT TODAY (Surface B's thinnest era carries dozens of clusters),
        and therefore worth pinning: no shipped row exercises this interaction.

        Nulling `lo_sampling` happens BEFORE `have_interval` is computed, so a
        flagged Surface B cell also reports `no_interval_to_widen`. That is
        coherent — `interval_unresolvable` is the root cause (the bootstrap
        resolved nothing) and `disagreement_status` the proximate one (there was
        no interval left to widen) — but it is exactly the kind of two-column
        interaction that silently inverts in a refactor, so both are asserted
        together with the half-width still present and unapplied.
        """
        out = _drive_llm_bands(
            monkeypatch,
            _sampling([{"topic": "Trade", "era": ERA_A, "lo": 0.25, "hi": 0.25}]),
            _half_widths([{"topic": "Trade", "era": ERA_A, "hw": 0.05,
                           "available": True}]),
            n_docs={ERA_A: B.MIN_CLUSTERS_FOR_RESOLVABLE_CI - 1},
        )
        row = out.iloc[0]
        assert bool(row["interval_unresolvable"])
        assert pd.isna(row["lo_sampling"]) and pd.isna(row["hi_sampling"])
        assert row["disagreement_status"] == B.NO_INTERVAL
        assert not bool(row["disagreement_band_applied"])
        assert row["ci_components"] == "sampling_only"
        # the point survives, and no half-width is smuggled onto a null bound
        assert row["point"] == pytest.approx(0.20)
        assert pd.isna(row["lo"]) and pd.isna(row["hi"])

    def test_a_resolvable_surface_b_cell_at_the_same_low_n_is_untouched(
        self, monkeypatch
    ):
        """The control for the test above: same two clusters, but the bootstrap
        resolved a real width, so nothing is withdrawn and the annotator
        component still applies."""
        out = _drive_llm_bands(
            monkeypatch,
            _sampling([{"topic": "Trade", "era": ERA_A, "lo": 0.20, "hi": 0.30}]),
            _half_widths([{"topic": "Trade", "era": ERA_A, "hw": 0.05}]),
            n_docs={ERA_A: B.MIN_CLUSTERS_FOR_RESOLVABLE_CI - 1},
        )
        row = out.iloc[0]
        assert not bool(row["interval_unresolvable"])
        assert row["disagreement_status"] == B.MEASURED
        assert row["lo"] == pytest.approx(0.15) and row["hi"] == pytest.approx(0.35)


class TestLeftMergeGuard:
    """`validate="one_to_one"` on a LEFT merge cannot see diverging key SETS.

    Length survives, `_require_full_merge` passes, and every row silently loses
    its disagreement component — published as `unavailable_thin_paired_sample`,
    a false cause. The guard must raise. It must also raise BEFORE
    `out["available"].astype(bool)` further down, which reads a NaN as True and
    would mark an unmatched row as HAVING a half-width. Pinning the raise is
    what protects that ordering.
    """

    def test_drifted_era_labels_raise_rather_than_publish_a_false_cause(self, monkeypatch):
        sampling = _sampling([{"topic": "Trade", "era": ERA_A},
                              {"topic": "War", "era": ERA_A}])
        hw = _half_widths([{"topic": "Trade", "era": ERA_A},
                           {"topic": "War", "era": ERA_A}])
        hw["era"] = hw["era"].str.title()  # "The founding" -> "The Founding"
        with pytest.raises(ValueError) as err:
            _drive_llm_bands(monkeypatch, sampling, hw)
        msg = str(err.value)
        assert "2 of 2" in msg
        assert "('Trade', 'The founding')" in msg
        assert "('War', 'The founding')" in msg
        assert "era" in msg

    def test_a_single_unmatched_cell_is_enough_to_raise(self, monkeypatch):
        """Not a threshold: one row silently reduced to a sampling-only interval
        and published as a thin paired sample is already a false cause.

        The half-width frame keeps its ROW COUNT here and drifts only one key,
        so the length assertion above it passes and this is the key-set guard
        being exercised, not `_require_full_merge` firing first.
        """
        sampling = _sampling([{"topic": "Trade", "era": ERA_A},
                              {"topic": "Trade", "era": ERA_B}])
        hw = _half_widths([{"topic": "Trade", "era": ERA_A},
                           {"topic": "Trade", "era": ERA_B}])
        hw.loc[hw["era"] == ERA_B, "era"] = "Expansion Era"
        with pytest.raises(ValueError) as err:
            _drive_llm_bands(monkeypatch, sampling, hw)
        assert "row count" not in str(err.value), "the length guard fired instead"
        assert "1 of 2" in str(err.value)
        assert f"('Trade', '{ERA_B}')" in str(err.value)

    def test_a_dropped_half_width_row_is_caught_by_the_length_guard_first(
        self, monkeypatch
    ):
        """The other half of the pair. A left merge preserves length against the
        LEFT frame, so a dropped right-hand row is invisible there —
        `_require_full_merge` compares against BOTH inputs, which is what makes
        it visible."""
        sampling = _sampling([{"topic": "Trade", "era": ERA_A},
                              {"topic": "Trade", "era": ERA_B}])
        hw = _half_widths([{"topic": "Trade", "era": ERA_A},
                           {"topic": "Trade", "era": ERA_B}])
        with pytest.raises(ValueError) as err:
            _drive_llm_bands(monkeypatch, sampling, hw[hw["era"] == ERA_A])
        assert "row count" in str(err.value)

    def test_duplicate_half_width_keys_are_refused_by_validate_one_to_one(
        self, monkeypatch
    ):
        """Narrowed to `MergeError` on purpose: a bare `Exception` here would be
        satisfied by an import error or a typo in the fixture, and would keep
        passing if `validate="one_to_one"` were dropped and something else
        happened to blow up two lines later."""
        sampling = _sampling([{"topic": "Trade", "era": ERA_A}])
        hw = _half_widths([{"topic": "Trade", "era": ERA_A},
                           {"topic": "Trade", "era": ERA_A}])
        with pytest.raises(pd.errors.MergeError) as err:
            _drive_llm_bands(monkeypatch, sampling, hw)
        assert "one-to-one" in str(err.value)

    def test_a_matched_frame_does_not_trip_the_guard(self, monkeypatch):
        """The guard's fail-safe direction, pinned: with keys that DO match it
        must stay silent, or the tests above would pass for the wrong reason."""
        out = _drive_llm_bands(
            monkeypatch,
            _sampling([{"topic": "Trade", "era": ERA_A}]),
            _half_widths([{"topic": "Trade", "era": ERA_A}]),
        )
        assert len(out) == 1


# --------------------------------------------------------------------------- #
# 6. assembly, schema and the seam
# --------------------------------------------------------------------------- #
class TestSchemaAndSeam:
    def test_meta_path_is_derived_from_the_parquet_path(self, tmp_path):
        """Redirecting `write_bands` must redirect BOTH files.
        `llm_annotations.write_manifest` hardcoding its own directory is the
        documented version of this bug."""
        assert B.meta_path_for(tmp_path / "bands.parquet") == tmp_path / "bands_meta.json"
        assert B.meta_path_for(Path("/x/y/other.parquet")) == Path("/x/y/other_meta.json")
        assert B.meta_path_for(B.BANDS_PATH) == B.BANDS_META_PATH

    def test_meta_counts_the_rows_that_actually_got_the_component(self, tmp_path):
        """On today's data all 450 LLM rows apply the disagreement component, so
        `rows_with_component_applied` and `rows_total` coincide and a check
        against the shipped artifact alone cannot tell them apart. Here they are
        deliberately made to differ."""
        table = pd.DataFrame({
            "surface": [B.LLM_SURFACE] * 3 + [B.COREX_SURFACE],
            "disagreement_band_applied": [True, True, False, False],
            # `write_bands` also derives the `interval_unresolvable` block from
            # the table it is handed, so the minimal frame has to carry those
            # columns too.
            "interval_unresolvable": [False] * 4,
            "period": ["e1", "e2", "e3", "1900"],
            "lo": [0.1] * 4,
            "hi": [0.2] * 4,
        })
        path = B.write_bands(table, tmp_path / "bands.parquet")
        meta = json.loads(B.meta_path_for(path).read_text())
        applied = meta["surfaces"][B.LLM_SURFACE]["annotator_disagreement"]
        assert applied["rows_total"] == 3
        assert applied["rows_with_component_applied"] == 2

    def test_load_bands_returns_none_when_the_artifact_is_absent(self, tmp_path):
        """A seam, not a stub: the charts render unbanded rather than inventing
        an interval."""
        assert B.load_bands(tmp_path / "nope.parquet") is None

    def test_load_bands_refuses_a_table_missing_band_columns(self, tmp_path):
        path = tmp_path / "bands.parquet"
        pd.DataFrame({"surface": ["corex_issues"], "series": ["war"]}).to_parquet(path)
        with pytest.raises(ValueError) as err:
            B.load_bands(path)
        assert "missing columns" in str(err.value)
        assert "'point'" in str(err.value)

    def test_load_bands_does_NOT_validate_dtypes_only_column_names(self, tmp_path):
        """PINNING A KNOWN GAP, NOT ENDORSING IT.

        Backlogged as `load-bands-validates-columns-not-dtypes`; do not "fix"
        it here.

        `load_bands`'s stated purpose is "refusing to guess at the band schema",
        but the check is column NAMES only. A table whose `lo`/`hi` are strings
        loads clean and fails much later, deep inside `band_traces`. This test
        documents the current behaviour so that tightening it is a deliberate,
        visible change rather than a silent one — and so the gap is discoverable
        from the test suite instead of only from the backlog.
        """
        path = tmp_path / "bands.parquet"
        bad = pd.DataFrame({c: ["not a number"] for c in B.BANDS_COLUMNS})
        bad.to_parquet(path)
        loaded = B.load_bands(path)
        assert loaded is not None, "column names alone satisfy today's guard"
        assert not pd.api.types.is_numeric_dtype(loaded["lo"]), (
            "the loaded band table's bounds are not even numeric and the guard "
            "passed anyway"
        )

    def test_series_band_returns_none_rather_than_an_empty_frame(self, shipped):
        """So a caller cannot draw a zero-length band and believe it drew
        something."""
        assert B.series_band(None, B.COREX_SURFACE, "war") is None
        assert B.series_band(shipped, B.COREX_SURFACE, "no such issue") is None
        assert B.series_band(shipped, B.LLM_SURFACE, shipped[
            shipped["surface"] == B.COREX_SURFACE]["series"].iloc[0]) is None

    def test_series_band_is_period_ordered(self, shipped):
        """The premise — that the input was actually reordered — is asserted
        before the conclusion. A fixed-seed `.sample(frac=1)` can hand back the
        original order, which would leave this passing while proving nothing
        about the sort."""
        name = shipped[shipped["surface"] == B.COREX_SURFACE]["series"].iloc[0]
        scrambled = shipped.sample(frac=1, random_state=3)
        rows = scrambled[(scrambled["surface"] == B.COREX_SURFACE)
                         & (scrambled["series"] == name)]
        assert not rows["period_order"].is_monotonic_increasing, (
            "the shuffle left the rows in period order — this test would be vacuous"
        )
        band = B.series_band(scrambled, B.COREX_SURFACE, name)
        assert band["period_order"].is_monotonic_increasing
        assert len(band) == 49

    def test_the_shipped_table_carries_exactly_the_declared_columns_in_order(self, shipped):
        """Two surfaces assembled by different code paths; a positional reader
        must not depend on which ran first."""
        assert list(shipped.columns) == B.BANDS_COLUMNS


# --------------------------------------------------------------------------- #
# 7. the shipped artifact
# --------------------------------------------------------------------------- #
class TestShippedArtifact:
    def test_surface_a_is_sampling_only_on_every_published_row(self, shipped):
        a = shipped[shipped["surface"] == B.COREX_SURFACE]
        assert len(a) == 1078
        assert a["lo"].equals(a["lo_sampling"])
        assert a["hi"].equals(a["hi_sampling"])
        assert set(a["ci_components"]) == {"sampling_only"}
        assert set(a["disagreement_status"]) == {B.NOT_APPLICABLE}

    def test_surface_b_widens_and_never_narrows_on_every_published_row(self, shipped):
        b = shipped[shipped["surface"] == B.LLM_SURFACE]
        assert len(b) == 450
        assert (b["lo"] <= b["lo_sampling"] + 1e-12).all()
        assert (b["hi"] >= b["hi_sampling"] - 1e-12).all()
        assert b["lo"].min() >= 0.0 and b["hi"].max() <= 1.0
        # Today every LLM row has both components; the guard is live, not
        # decorative, but a drift to sampling_only here would be a finding.
        assert set(b["ci_components"]) == {"sampling+annotator_disagreement"}

    def test_stripping_the_disagreement_component_leaves_surface_a_untouched(self, shipped):
        """The wiring test from the verification strategy: remove the annotator
        term and Surface B shrinks everywhere while Surface A does not move."""
        a = shipped[shipped["surface"] == B.COREX_SURFACE]
        b = shipped[shipped["surface"] == B.LLM_SURFACE]
        a_width = (a["hi"] - a["lo"])
        a_sampling_width = (a["hi_sampling"] - a["lo_sampling"])
        assert a_width.equals(a_sampling_width)
        b_width = (b["hi"] - b["lo"])
        b_sampling_width = (b["hi_sampling"] - b["lo_sampling"])
        assert (b_width >= b_sampling_width - 1e-12).all()
        assert (b_width > b_sampling_width + 1e-12).sum() > 0

    def test_the_degenerate_cells_are_flagged_and_carry_no_interval(
        self, shipped
    ):
        """A bootstrap over n clusters has only C(2n-1, n) distinct resamples —
        3 at n=2 — so a 1785 series with zero paragraphs in BOTH speeches
        returns the same replicate every draw and its interval collapses to
        lo == hi == 0. Those 12 cells now WITHDRAW the interval rather than
        publish a fake-precise zero, and are flagged per-cell. Recomputed from
        the shipped artifact."""
        a = shipped[shipped["surface"] == B.COREX_SURFACE]
        flagged = a[a["interval_unresolvable"]]
        assert len(flagged) == 12
        assert set(flagged["period"]) == {"1785"}
        for col in ("lo", "hi", "lo_sampling", "hi_sampling"):
            assert flagged[col].isna().all(), col
        # The point estimate survives: the observed share is a real
        # measurement, and only the interval was unresolvable.
        assert flagged["point"].notna().all()
        # And the OLD predicate now matches nothing, which is the whole change.
        assert len(a[(a["ci_status"] == "low_cluster_caution")
                     & (a["lo"] == a["hi"])]) == 0

    def test_a_legitimate_confident_zero_keeps_its_interval(self, shipped):
        """The gate is a conjunction, and this is the arm that proves it is not
        'zero width alone'. Seven Surface A cells are zero-width in `ok`
        periods, where a dozen-plus speeches genuinely never touched the issue
        — a finding, not an artifact — and they must still publish 0.0-0.0."""
        a = shipped[shipped["surface"] == B.COREX_SURFACE]
        kept = a[a["lo"].notna() & (a["lo"] == a["hi"])]
        assert len(kept) == 7
        assert not kept["interval_unresolvable"].any()
        assert (kept["n_speeches"] >= B.MIN_CLUSTERS_FOR_RESOLVABLE_CI).all()

    def test_the_widest_1785_band_is_untouched(self, shipped):
        """The other control: Religion & values' two speeches DISAGREE, so its
        bootstrap resolved a real (enormous) width at the same n=2. Suppressing
        it would delete the corpus's loudest uncertainty signal."""
        a = shipped[shipped["surface"] == B.COREX_SURFACE]
        row = a[(a["series"] == "Religion & values")
                & (a["period"] == "1785")].iloc[0]
        assert not row["interval_unresolvable"]
        assert row["lo"] == pytest.approx(0.0)
        assert row["hi"] == pytest.approx(2 / 3, abs=1e-3)

    def test_every_flagged_cell_in_THIS_corpus_sits_at_zero(self, shipped):
        """The empirical premise `unresolved_traces` cites to justify
        `cliponaxis=False`: every flagged ring lands exactly on the axis floor
        and would be drawn half-cropped without it.

        It is a fact about the data, not about the gate — a cell where all
        clusters agree at a non-zero share is equally unresolvable (pinned
        synthetically in `TestUnresolvableGate`). Recorded here so that if the
        corpus ever produces a non-zero flagged cell, the comment's premise
        fails visibly rather than the clip setting quietly becoming unjustified.
        """
        a = shipped[shipped["surface"] == B.COREX_SURFACE]
        flagged = a[a["interval_unresolvable"]]
        assert len(flagged) == 12
        assert (flagged["point"] == 0.0).all()

    def test_surface_b_has_nothing_to_flag_and_that_is_computed_not_assumed(
        self, shipped
    ):
        """Scope confirmation. Surface B's thinnest era carries far more than
        `MIN_CLUSTERS_FOR_RESOLVABLE_CI` clusters, so no row is flagged — but
        the gate still RUNS there, and the cluster margin is asserted rather
        than the zero count alone. A zero that comes from the gate never firing
        and a zero that comes from the gate being wired out look identical in
        the count.
        """
        b = shipped[shipped["surface"] == B.LLM_SURFACE]
        assert not b["interval_unresolvable"].any()
        assert b["n_speeches"].min() >= B.MIN_CLUSTERS_FOR_RESOLVABLE_CI
        assert b["lo_sampling"].notna().all()

    def test_the_docstrings_zero_width_census_is_LABELLED_as_a_pre_gate_count(
        self, shipped
    ):
        """19 is the PRE-gate census and both docstrings must say so.

        This test previously pinned the opposite: the module docstring read "the
        shipped table has 19 zero-width Surface A cells", which is false of the
        artifact — post-gate it holds 7, the other 12 being null rather than
        zero-width. The decomposition was always sound (7 kept + 12 withdrawn =
        19); only the tense was wrong, and both reviewers read past it because
        every number checked out. **That wording has been fixed. If this test
        fails, a docstring has regressed to quoting 19 as a property of the
        shipped table — it does not mean you broke the census.**

        Both places that quote the census are checked, because
        `_unresolvable_interval`'s docstring carried the same sentence and
        fixing only the module-level one would leave the claim alive two
        screens away.
        """
        a = shipped[shipped["surface"] == B.COREX_SURFACE]
        kept = a[a["lo"].notna() & (a["lo"] == a["hi"])]
        flagged = a[a["interval_unresolvable"]]
        assert len(kept) == 7, "zero-width cells actually present in the artifact"
        assert len(flagged) == 12
        assert len(kept) + len(flagged) == 19, "the pre-gate census"

        module_prose = _prose(B.__doc__)
        gate_prose = _prose(B._unresolvable_interval.__doc__)

        # The count is still stated — the fix was to label it, not to delete it.
        assert f"{len(kept) + len(flagged)} zero-width" in module_prose
        assert f"{len(kept) + len(flagged)} zero-width" in gate_prose
        # ... and each place says which side of the gate it is counting.
        assert "Before this gate runs" in module_prose
        assert f"the SHIPPED table contains {len(kept)} zero-width" in module_prose
        assert "PRE-gate census" in gate_prose
        assert f"the shipped table keeps {len(kept)}" in gate_prose
        # The exact false sentence, refused by name in both.
        assert "the shipped table has 19 zero-width" not in module_prose
        assert "the shipped table's 19 zero-width" not in gate_prose

        # the periods it names, re-derived
        assert sorted(kept["period"].unique()) == ["1790", "1795", "1800", "1810"]
        assert "(1790, 1795, 1800, 1810" in module_prose
        assert set(kept["ci_status"]) == {"ok"}, "the docstring calls them all ok"

        # ...and the SPEECH COUNT behind them, which is the conjunction's whole
        # second arm and was the one part of this census nothing pinned. Both
        # docstrings shipped a magnitude here instead of a measurement — one
        # said "sixty speeches", the other "dozens" — while the artifact says
        # 13-14. That is ~4x and >=24 respectively, on the number that decides
        # whether "enough speeches agreed for the agreement to mean something".
        lo_n, hi_n = int(kept["n_speeches"].min()), int(kept["n_speeches"].max())
        assert (lo_n, hi_n) == (13, 14)
        assert f"{lo_n}-{hi_n} speeches" in module_prose
        assert f"{lo_n}-{hi_n} speeches" in gate_prose
        # 14 is the ceiling over EVERY zero-width cell, kept or withdrawn — so
        # no larger count can honestly be quoted for this arm. The corpus's
        # per-period max is far higher (67), but that period has no zero-width
        # cell, which is exactly the trap the two wrong numbers fell into.
        zero_width = a[(a["lo"].notna() & (a["lo"] == a["hi"]))
                       | a["interval_unresolvable"]]
        assert int(zero_width["n_speeches"].max()) == hi_n
        assert int(a["n_speeches"].max()) > hi_n
        for overstatement in ("sixty speeches", "dozens of speeches"):
            assert overstatement not in module_prose
            assert overstatement not in gate_prose

    def test_the_recovered_period_is_1785_and_it_is_the_only_one(self, shipped):
        """The scope correction, pinned: the former `>= 40` hard mask dropped
        exactly ONE of the 49 five-year periods. It should not be described as
        a large recovery of hidden data."""
        a = shipped[shipped["surface"] == B.COREX_SURFACE]
        per_period = a.groupby("period")[["n_paragraphs", "n_speeches"]].first()
        assert len(per_period) == 49
        thin = per_period[per_period["n_paragraphs"] < B.MIN_PERIOD_PARAGRAPHS]
        assert list(thin.index) == ["1785"]
        assert thin.loc["1785"].tolist() == [14, 2]

    def test_suppressed_rows_carry_no_interval_and_cautioned_rows_do(self, shipped):
        """The trust gate's contract, stated as an invariant over the whole
        artifact: a consumer keying on `ci_status` must never find a published
        bound behind `suppressed_n_floor`, nor a missing one behind `ok`."""
        suppressed = shipped[shipped["ci_status"] == "suppressed_n_floor"]
        assert suppressed["lo"].isna().all() and suppressed["hi"].isna().all()
        for status in ("ok", "low_cluster_caution"):
            # `interval_unresolvable` is the ONE other way a bound goes missing,
            # and it is a per-CELL gate rather than the per-period `ci_status`.
            # Excluded explicitly rather than by loosening the invariant: a
            # missing bound behind `ok` with no flag on it is still a defect.
            live = shipped[(shipped["ci_status"] == status)
                           & ~shipped["interval_unresolvable"]]
            assert live["lo"].notna().all() and live["hi"].notna().all()
        assert set(shipped["ci_status"]) <= {"ok", "low_cluster_caution",
                                             "suppressed_n_floor"}

    def test_meta_carries_no_wall_clock_stamp(self, shipped_meta):
        """`data/combat/`'s rule: a dirty `git status data/bands.parquet` must
        mean the numbers moved, not that the clock did. Provenance identity is
        the corpus fingerprint; *when* it ran is git's job."""
        blob = json.dumps(shipped_meta)
        for banned in ("generated_at", "timestamp", "run_date", "built_at", "draw_date"):
            assert banned not in blob
        assert shipped_meta["api_calls"] == 0
        assert set(shipped_meta["corpus_fingerprint"]) == {
            "n_speeches", "n_paragraphs", "doc_name_sha256"}

    def test_meta_counts_agree_with_the_table_they_describe(self, shipped, shipped_meta):
        llm_meta = shipped_meta["surfaces"][B.LLM_SURFACE]["annotator_disagreement"]
        b = shipped[shipped["surface"] == B.LLM_SURFACE]
        assert llm_meta["rows_total"] == len(b)
        assert llm_meta["rows_with_component_applied"] == int(
            b["disagreement_band_applied"].sum())
        assert llm_meta["min_paired_paragraphs"] == B.MIN_PAIRED_PARAGRAPHS
        assert shipped_meta["bootstrap"]["seed"] == B.BOOTSTRAP_SEED
        assert shipped_meta["bootstrap"]["n_draws"] == B.BOOTSTRAP_DRAWS
        assert shipped_meta["ci_status"]["min_period_paragraphs"] == B.MIN_PERIOD_PARAGRAPHS

    def test_meta_unresolvable_counts_are_derived_from_the_table(
        self, shipped, shipped_meta
    ):
        """The computed half of the new meta block, re-derived from the table it
        describes. CLAUDE.md: the one prose block that shipped wrong on the
        previous task was the only one without a re-derivation test."""
        block = shipped_meta["interval_unresolvable"]
        for surface in (B.COREX_SURFACE, B.LLM_SURFACE):
            sub = shipped[shipped["surface"] == surface]
            assert block["rows_flagged"][surface] == int(
                sub["interval_unresolvable"].sum())
            assert block["rows_total"][surface] == len(sub)
            assert block["flagged_periods"][surface] == sorted(
                sub.loc[sub["interval_unresolvable"], "period"].unique().tolist())
        assert block["column"] == "interval_unresolvable"
        assert block["min_clusters_for_resolvable_ci"] == \
            B.MIN_CLUSTERS_FOR_RESOLVABLE_CI
        assert block["distinct_resamples"] == {
            str(n): B.distinct_resamples(n) for n in (2, 3, 4, 5, 10)}

    def test_meta_unresolvable_prose_numbers_are_all_re_derivable(
        self, shipped, shipped_meta
    ):
        """The TYPED half of the same block — and the reason this test exists.

        The block is written under a source comment claiming its values are
        "DERIVED from the table being written, never typed". Three of them are
        in fact typed into the f-strings and can drift from the artifact
        independently:

          * `0.0-66.7%`, Religion & values' 1785 band — a data-derived bound;
          * `n_speeches < 8 OR n_paragraphs < 40`, which are `LOW_CLUSTER_CAUTION`
            and `MIN_PERIOD_PARAGRAPHS` and are available as constants;
          * `CI_LOW/100 = 0.025` and the `n**-n` ladder in `floor_derivation`.

        Rather than narrow the claim (a source change, out of scope here), every
        typed number is re-derived so that a drift fails a test instead of
        shipping. See Discovered Issues for the overstated comment itself.
        """
        block = shipped_meta["interval_unresolvable"]
        a = shipped[shipped["surface"] == B.COREX_SURFACE]

        rv = a[(a["series"] == "Religion & values") & (a["period"] == "1785")].iloc[0]
        assert f"{rv['lo'] * 100:.1f}-{rv['hi'] * 100:.1f}%" in block[
            "why_both_conditions"], "the 0.0-66.7% band has moved"

        assert (f"n_speeches < {B.LOW_CLUSTER_CAUTION} OR n_paragraphs < "
                f"{B.MIN_PERIOD_PARAGRAPHS}") in block["why_not_ci_status"]
        assert f"same {int(a['interval_unresolvable'].sum())} cells" in block[
            "why_not_ci_status"]

        kept = int(len(a[a["lo"].notna() & (a["lo"] == a["hi"])]))
        flagged = int(a["interval_unresolvable"].sum())
        assert f"{kept} of the {kept + flagged} zero-width" in block[
            "why_both_conditions"]
        assert str(sorted(a.loc[a["lo"].notna() & (a["lo"] == a["hi"]),
                                "period"].unique().tolist())) in block[
            "why_both_conditions"]

        floor = B.MIN_CLUSTERS_FOR_RESOLVABLE_CI
        assert f"CI_LOW/100 = {B.CI_LOW / 100}" in block["floor_derivation"]
        assert f"n={floor} is the first n where it can" in block["floor_derivation"]
        # The n**-n ladder, each value matched TOGETHER WITH the n it belongs
        # to: "0.25" alone would match against the wrong row of the ladder.
        for n in (2, 3, 4):
            claim = f"{n ** (-n):.2g} at n={n}"
            assert claim in block["floor_derivation"], claim
        assert f"fewer than {floor} speech clusters" in block["predicate"]
        assert f"MIN_CLUSTERS_FOR_CI stays at {B.MIN_CLUSTERS_FOR_CI}" in block[
            "min_clusters_for_ci_unchanged"]

    def test_the_meta_floor_follows_the_constant_rather_than_agreeing_with_it(
        self, tmp_path, monkeypatch
    ):
        """`block["min_clusters_for_resolvable_ci"] == B.MIN_CLUSTERS_FOR_RESOLVABLE_CI`
        is belt-and-suspenders: both sides are 4, so a hardcoded `4` in the meta
        dict satisfies it. Rebuild the meta with the constant MOVED and check
        the file followed — the only assertion shape that tells a read from a
        coincidence. The predicate prose is checked too, since it interpolates
        the same constant separately.

        `CI_LOW` is retuned ALONGSIDE the constant to the percentile that
        actually derives 9 (`1e-6` puts the threshold between the n=8 and n=9
        rungs). `write_bands` now refuses to emit a `floor_derivation` block
        that argues for a different floor than the one in force, so the two can
        no longer be moved independently — see the divergence assertion at the
        end, which pins that refusal.
        """
        monkeypatch.setattr(B, "CI_LOW", 1e-6)
        monkeypatch.setattr(B, "MIN_CLUSTERS_FOR_RESOLVABLE_CI", 9)
        assert B._min_clusters_for_resolvable_ci(1e-6) == 9, (
            "the retuned percentile must actually derive the patched floor, or "
            "this test is asserting on an inconsistent pair")
        table = pd.DataFrame({
            "surface": [B.COREX_SURFACE, B.LLM_SURFACE],
            "disagreement_band_applied": [False, True],
            "interval_unresolvable": [False, False],
            "period": ["1900", "e1"],
            "lo": [0.1, 0.1],
            "hi": [0.2, 0.2],
        })
        path = B.write_bands(table, tmp_path / "bands.parquet")
        block = json.loads(B.meta_path_for(path).read_text())["interval_unresolvable"]
        assert block["min_clusters_for_resolvable_ci"] == 9
        assert "fewer than 9 speech clusters" in block["predicate"]
        assert "MIN_CLUSTERS_FOR_CI stays at 2" in block["min_clusters_for_ci_unchanged"]

        # The guard itself: move ONLY the constant and the derivation would
        # argue for 4 beside a published floor of 9 — a rationale defeating the
        # number it exists to support. `write_bands` must refuse BEFORE writing
        # anything, and both halves of the artifact pair are checked.
        #
        # Asserting only on the sidecar is what let the guard ship running after
        # `to_parquet`: it raised with the meta correctly absent and the PARQUET
        # already replaced, so the table moved while its provenance still
        # described the previous one — the one outcome worse than writing both
        # or writing neither, in a repo whose rule is that a dirty
        # `git status data/bands.parquet` means the numbers actually moved.
        monkeypatch.setattr(B, "CI_LOW", 2.5)
        with pytest.raises(ValueError) as err:
            B.write_bands(table, tmp_path / "divergent.parquet")
        assert "argues for a floor the artifact does not use" in str(err.value)
        assert not (tmp_path / "divergent_meta.json").exists(), (
            "the refusal must leave no partial meta behind")
        assert not (tmp_path / "divergent.parquet").exists(), (
            "the refusal must leave no partial artifact behind, sidecar or table")

    def test_floor_derivation_prose_follows_a_retuned_percentile_end_to_end(
        self, tmp_path, monkeypatch
    ):
        """The whole `floor_derivation` block is a function of `CI_LOW`.

        This test used to pin the OPPOSITE — that only `CI_LOW/100` was
        interpolated while the `n**-n` ladder and the conclusion were typed, so
        that at `CI_LOW = 5.0` the block reported a threshold of 0.05 beside
        "n=4 is the first n where it can", which its own n=3 rung (0.037)
        already clears. **That defect has been FIXED. If this test fails, the
        block has regressed to typed literals — it does not mean you broke the
        derivation.**

        Worth being precise about the defect class, because it changed what the
        fix had to do: this was typed-vs-derived, NOT the self-defeating
        rationale CLAUDE.md records from `topic-chart-upgrades`. That one shipped
        a premise that was false on the day it was written. This block was
        correct as published and would only have gone wrong after a retune, so
        the job was to keep it correct rather than to correct it.
        """
        monkeypatch.setattr(B, "CI_LOW", 5.0)
        # The gate constant moves WITH the percentile it is derived from:
        # `write_bands` refuses to publish a derivation arguing for a floor the
        # artifact does not apply, so a retune has to move both. Production
        # moves them together by construction (the constant is computed from
        # `CI_LOW` at import); only a monkeypatch can separate them.
        monkeypatch.setattr(B, "MIN_CLUSTERS_FOR_RESOLVABLE_CI", 3)
        table = pd.DataFrame({
            "surface": [B.COREX_SURFACE, B.LLM_SURFACE],
            "disagreement_band_applied": [False, True],
            "interval_unresolvable": [False, False],
            "period": ["1900", "e1"],
            "lo": [0.1, 0.1],
            "hi": [0.2, 0.2],
        })
        path = B.write_bands(table, tmp_path / "bands.parquet")
        block = json.loads(
            B.meta_path_for(path).read_text())["interval_unresolvable"]
        prose = block["floor_derivation"]

        retuned_floor = B._min_clusters_for_resolvable_ci(5.0)
        assert retuned_floor == 3
        # The derivation and the PUBLISHED floor agree after the retune — the
        # property the write-time guard exists to make unfalsifiable.
        assert block["min_clusters_for_resolvable_ci"] == retuned_floor
        assert f"fewer than {retuned_floor} speech clusters" in block["predicate"]
        # threshold, conclusion and ladder now move together ...
        assert "CI_LOW/100 = 0.05" in prose
        assert f"n={retuned_floor} is the first n where it can" in prose
        assert "0.037 at n=3" in prose
        # ... and the ladder STOPS at the floor rather than reciting a rung the
        # retuned percentile has made irrelevant. This is the assertion that
        # catches a re-typed ladder: a frozen ladder still contains the n=3 rung
        # asserted above, so checking only that would pass on the old string.
        assert "0.0039 at n=4" not in prose
        assert "n=4" not in prose

    def test_floor_derivation_block_is_self_consistent_at_the_shipped_percentile(
        self, shipped_meta
    ):
        """The committed artifact's own block, checked against the derivation
        rather than against a remembered string: the ladder's last rung must be
        the first one to clear the stated threshold, the rung below it must NOT
        clear it, and the conclusion must name that same n.

        The retune test above proves the block moves; this proves where it
        currently sits is right. Neither alone is enough — a block hardcoded to
        the correct answer passes this one, and a block that moves to the wrong
        answer passes that one.
        """
        block = shipped_meta["interval_unresolvable"]
        prose = block["floor_derivation"]
        floor = block["min_clusters_for_resolvable_ci"]

        assert floor == B.MIN_CLUSTERS_FOR_RESOLVABLE_CI
        assert f"CI_LOW/100 = {B.CI_LOW / 100}" in prose
        assert f"n={floor} is the first n where it can" in prose
        assert floor ** (-floor) <= B.CI_LOW / 100
        assert (floor - 1) ** (-(floor - 1)) > B.CI_LOW / 100
        assert f"{floor ** (-floor):.2g} at n={floor}" in prose
        assert f"{(floor - 1) ** (-(floor - 1)):.2g} at n={floor - 1}" in prose

    def test_the_superseded_predicate_prose_no_longer_claims_the_cells_are_published(
        self, shipped_meta
    ):
        """`degenerate_interval_predicate` used to end "Left published rather
        than suppressed". That sentence became false the moment the cells were
        withdrawn, and a provenance artifact that describes the opposite of what
        it contains is worse than one that says nothing."""
        superseded = shipped_meta["ci_status"]["degenerate_interval_predicate"]
        assert "Left published rather than suppressed" not in superseded
        assert "SUPERSEDED" in superseded
        assert "interval_unresolvable" in superseded

    def test_rebuilding_reproduces_the_committed_artifact_byte_for_byte(
        self, real_annotations, tmp_path
    ):
        """Determinism, end to end: `python -m presidential_profiles.bands` is
        pure local compute over frozen parquets ($0, zero API calls), so two
        builds must agree with each other AND with what is committed. Written
        to tmp_path — a test must never overwrite the artifact it verifies.

        The VALUE comparison runs before the byte comparison on purpose. A
        sha256 pin is the one piece of environment coupling in this suite: a
        pyarrow or pandas bump can change the encoding of an unchanged table.
        Ordering the assertions this way makes the failure self-diagnosing —
        values-differ means the numbers moved (a real regression), while
        values-match-but-bytes-differ means only the parquet writer changed, and
        the assertion message says so rather than leaving the next reader to
        guess from a pair of hex strings.
        """
        first, second = B.build_bands(), B.build_bands()
        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(first, pd.read_parquet(DATA / "bands.parquet"))

        p1 = B.write_bands(first, tmp_path / "bands.parquet")
        p2 = B.write_bands(second, tmp_path / "again.parquet")
        digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        assert digest(p1) == digest(p2), "two builds wrote different bytes"

        writer_note = (
            "the rebuilt table's VALUES match data/bands.parquet exactly (asserted "
            "above) but its bytes do not. That is a parquet-encoding change — a "
            "pandas/pyarrow bump — not a change in the numbers. Rewrite the "
            "artifact with `python -m presidential_profiles.bands` and commit the "
            "re-encoded file; do not chase it as a data regression."
        )
        assert digest(p1) == digest(DATA / "bands.parquet"), writer_note
        assert digest(B.meta_path_for(p1)) == digest(B.meta_path_for(p2))
        assert digest(B.meta_path_for(p1)) == digest(DATA / "bands_meta.json"), (
            "bands_meta.json is plain JSON, so this is a genuine content change"
        )


# --------------------------------------------------------------------------- #
# 8. the CLI entry point
# --------------------------------------------------------------------------- #
class TestCliAndChecks:
    """`main` / `print_checks` — previously unexercised.

    `main`'s own job is orchestration (build -> write -> report), so the build
    is stubbed with the shipped table and the write is redirected: `write_bands`
    binds its default `path=BANDS_PATH` at DEF time, so monkeypatching
    `B.BANDS_PATH` would NOT redirect it and a naive `main()` call would
    overwrite the committed artifact. Each test also digests the real file
    afterwards, so a future refactor that reintroduces the real write fails here
    rather than silently rewriting `data/bands.parquet`.

    Both entry points reach `print_checks`, which calls
    `pd.set_option("display.width", 200)` and never restores it — a source
    defect, backlogged as `print-checks-leaks-global-pandas-option` and
    deliberately NOT fixed here. This class is what makes the leak reachable
    inside the suite, though, so it contains it test-side: without the
    `option_context` below, every DataFrame repr for the rest of the session
    would be silently widened by a test that had already finished.
    """

    @pytest.fixture(autouse=True)
    def _contain_the_display_width_leak(self):
        with pd.option_context("display.width", pd.get_option("display.width")):
            yield

    @pytest.fixture
    def run_main(self, shipped, tmp_path, monkeypatch):
        out_path = tmp_path / "bands.parquet"

        def _run(argv):
            before = hashlib.sha256((DATA / "bands.parquet").read_bytes()).hexdigest()
            monkeypatch.setattr(B, "build_bands", lambda: shipped.copy())
            monkeypatch.setattr(
                B, "write_bands",
                lambda table, path=out_path: (
                    pd.DataFrame.to_parquet(table, path, index=False) or path),
            )
            B.main(argv)
            after = hashlib.sha256((DATA / "bands.parquet").read_bytes()).hexdigest()
            assert after == before, "main() wrote to the committed artifact"
            # `main` must actually PERSIST the table, not merely report on it —
            # asserting only on stdout leaves a main() that never writes green.
            assert out_path.exists(), "main() built the table but wrote nothing"
            assert len(pd.read_parquet(out_path)) == len(shipped)
            return out_path

        return _run

    def test_main_writes_the_table_and_reports_both_surfaces(self, run_main, capsys):
        run_main([])
        out = capsys.readouterr().out
        assert "wrote" in out and "1,528 rows" in out
        assert B.COREX_SURFACE in out and B.LLM_SURFACE in out
        # the trust gate and the components are both surfaced to the operator
        assert "ci_status" in out and "ci_components" in out
        # the thin-period flag reaches the operator's summary, not just the file
        assert "low_cluster_caution" in out
        assert "sampling+annotator_disagreement" in out

    def test_quiet_writes_without_the_checks(self, run_main, capsys):
        run_main(["--quiet"])
        out = capsys.readouterr().out
        assert "wrote" in out
        assert B.COREX_SURFACE not in out

    def test_print_checks_separates_the_flagged_cells_from_the_kept_zero_widths(
        self, shipped, capsys
    ):
        """The operator's view of this task's entire deliverable.

        `print_checks` is where a human sees the gate's effect, and its two new
        lines report the two halves of the conjunction: cells WITHDRAWN
        (`interval_unresolvable`) and zero-width cells KEPT because their period
        had enough clusters. The counts and the periods are re-derived from the
        table rather than typed, per the repo's provenance-prose doctrine.

        The load-bearing assertion is that the two lines DISAGREE. Both masks
        are one-line boolean expressions over the same frame, so a copy-paste of
        the flagged mask into the kept line — or vice versa — is the realistic
        regression, and it would print two plausible, self-consistent lines that
        nothing else in the suite reads.
        """
        B.print_checks(shipped)
        out = capsys.readouterr().out

        a = shipped[shipped["surface"] == B.COREX_SURFACE]
        flagged = a[a["interval_unresolvable"]]
        kept = a[a["lo"].notna() & (a["lo"] == a["hi"])]
        assert len(flagged) != len(kept), (
            "fixture premise: if the two counts coincided this test could not "
            "tell the two lines apart"
        )
        assert set(flagged["period"]).isdisjoint(kept["period"]), (
            "a period cannot be both, so the two period lists must not overlap"
        )

        assert (f"interval_unresolvable: {len(flagged)} "
                f"(periods {sorted(flagged['period'].unique().tolist())})") in out
        assert (f"zero-width intervals KEPT (legitimate): {len(kept)} "
                f"(periods {sorted(kept['period'].unique().tolist())})") in out

        # Surface B is computed, not assumed absent — the line must be printed
        # for it too, reporting zero rather than being skipped.
        assert out.count("interval_unresolvable:") == 2
        assert "interval_unresolvable: 0 (periods [])" in out

    def test_print_checks_reports_the_thinnest_period_per_issue(self, shipped, capsys):
        B.print_checks(shipped)
        out = capsys.readouterr().out
        assert "thinnest CorEx period per issue" in out
        assert "1785" in out
        assert "disagreement contribution to LLM band width" in out


# --------------------------------------------------------------------------- #
# 9. the docstring's bare numbers, re-derived from their named artifacts
# --------------------------------------------------------------------------- #
class TestPublishedCountsMatchTheirArtifacts:
    """Every bare number stated in a `bands.py` docstring, re-derived from the
    artifact it is sourced to.

    CLAUDE.md documents this drift class twice ("the drift concentrates in
    sentences containing a bare number"; "count periodization recovery in
    DISTINCT boundaries"). Reading the prose proves nothing — the only check
    that works is recomputing from the artifact, so that is what these do.

    The class covers the MODULE docstring's three counts (8,570 paired
    paragraphs; The founding's 219; 262 paired speeches) and `_ci_status`'s
    three era-grain counts. It is the pattern any new docstring number must be
    added to — including numbers added by a later refactor, which is how the
    `_ci_status` three arrived.
    """

    @pytest.fixture
    def paired(self, real_annotations) -> pd.DataFrame:
        return B.paired_annotations(attention.load_taxonomy())

    def test_the_ci_status_era_grain_justification_matches_the_artifact(self, shipped):
        """`_ci_status`'s docstring justifies reusing 5-year-period thresholds
        on the era grain with three bare numbers: the thinnest era's speech and
        paragraph counts, and that every LLM row publishes `ok`. Re-derived.

        Written to check EACH FLOOR'S OWN thinnest era rather than one "thinnest
        era", because the first draft of that docstring assumed a single one and
        was wrong: War & New Deal is thinnest by speeches (55) while The founding
        is thinnest by paragraphs (927), and the sentence had attached The
        founding's rank to War & New Deal's paragraph count. Exactly the class
        this class exists to catch — it was caught by writing the assertion, not
        by rereading the prose.

        The `ok` claim is the one nothing else in the suite covers.
        `test_surface_b_widens_and_never_narrows_on_every_published_row` pins
        `len(b) == 450` and the `ci_components` set but says nothing about
        `ci_status`, and `test_suppressed_rows_carry_no_interval_and_cautioned_rows_do`
        pins the status set over the WHOLE table — where `low_cluster_caution`
        is legitimately present on Surface A. So a Surface B row degrading to
        `low_cluster_caution` passes both of those while falsifying the
        docstring's "live-but-slack" reasoning outright.
        """
        b = shipped[shipped["surface"] == B.LLM_SURFACE]
        per_era = b.groupby("period")[["n_speeches", "n_paragraphs"]].first()
        prose = _prose(B._ci_status.__doc__)

        thin_speeches = per_era["n_speeches"].idxmin()
        thin_paragraphs = per_era["n_paragraphs"].idxmin()
        assert thin_speeches != thin_paragraphs, (
            "the two floors now bind on the SAME era, so the docstring's "
            "'the two floors bind on different eras' no longer holds"
        )
        assert (
            f"the thinnest era by speeches is {thin_speeches} at "
            f"{per_era.loc[thin_speeches, 'n_speeches']}"
        ) in prose
        assert (
            f"the thinnest by paragraphs is {thin_paragraphs} at "
            f"{per_era.loc[thin_paragraphs, 'n_paragraphs']}"
        ) in prose

        assert set(b["ci_status"]) == {"ok"}
        assert f"so all {len(b)} LLM rows publish `ok`" in prose
        # The floors it claims to clear are the module's own constants, so a
        # retune of either cannot leave the sentence quoting the old pair.
        assert (f"against floors of {B.LOW_CLUSTER_CAUTION} and "
                f"{B.MIN_PERIOD_PARAGRAPHS}") in prose
        assert per_era["n_speeches"].min() >= B.LOW_CLUSTER_CAUTION
        assert per_era["n_paragraphs"].min() >= B.MIN_PERIOD_PARAGRAPHS

    def test_the_paired_set_is_8570_paragraphs(self, paired):
        assert len(paired) == 8570
        assert "8,570 paragraphs" in _prose(B.__doc__)
        # And it is genuinely keyed, not positionally aligned.
        assert not paired.duplicated(["doc_name", "para_idx"]).any()

    def test_the_thinnest_era_is_the_founding_at_219_paragraphs(self, paired):
        counts = paired["era"].value_counts()
        assert counts.min() == 219
        assert counts.idxmin() == "The founding"
        assert "thinnest: The founding, 219 paragraphs" in _prose(B.__doc__)
        assert len(counts) == 9  # every era covered, so none falls back today
        assert counts.min() >= B.MIN_PAIRED_PARAGRAPHS

    def test_the_paired_set_covers_262_speeches_drawn_from_the_sample_file(
        self, paired, real_annotations, shipped_meta
    ):
        """A DOCUMENT-level sample, which is the whole point of the transfer
        caveat: the disagreement was measured on whole speeches and is assumed
        to transfer to a full-corpus era interval.

        262 is what the second annotator actually delivered. The sample FILE
        drew 266 — the persisted file, not the seed, being the source of truth
        for membership — and the four missing ones are the batch-cancel losses
        the annotation lessons record. So the claim is `262 <= 266`, not
        `262 == n_sampled`, and both halves are asserted."""
        docs = set(paired["doc_name"])
        assert len(docs) == 262
        assert "262 of them" in _prose(B.__doc__)
        assert "262 speeches" in json.dumps(shipped_meta)

        sample = json.loads(
            (real_annotations / "agreement_sample_v1.json").read_text())
        assert sample["n_sampled"] == 266
        assert docs <= set(sample["doc_names"])

    def test_every_paired_paragraph_lands_inside_a_named_era(self, paired):
        """`triangulate.assign_eras()`'s rule: an out-of-band year must raise
        rather than fall through to NaN and be read as a corpus-wide figure."""
        assert paired["era"].notna().all()
        assert set(paired["era"]) <= {label for label, _, _ in
                                      __import__(
                                          "presidential_profiles.trends",
                                          fromlist=["ERAS"]).ERAS}

    def test_paired_topic_sets_are_normalized_to_canonical_taxonomy_names(
        self, paired, real_annotations
    ):
        """Case drift ("War Of 1812") would fragment a real topic across two
        rows of the half-width table and quietly halve its measured share."""
        taxonomy = attention.load_taxonomy()
        canonical = set(attention.level_topics(taxonomy, "level2"))
        for col in ("topics_primary", "topics_secondary"):
            seen = set().union(*paired[col])
            assert seen <= canonical, sorted(seen - canonical)[:5]
