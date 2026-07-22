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
        labels = _labels([
            {"doc": "a", "year": 1900, "n": 50, "hits": {"war": 50}},
            {"doc": "b", "year": 1901, "n": 50, "hits": {"war": 50}},
        ])
        out = B.bootstrap_corex_periods(labels, ["war"], n_draws=50)
        assert out["point"].max() == pytest.approx(1.0)
        assert out["hi_sampling"].max() <= 1.0

    def test_only_boolean_columns_are_treated_as_issues(self):
        labels = _labels([{"doc": "a", "year": 1900, "n": 4, "hits": {"war": 2}}])
        labels["coherence"] = 0.5
        assert B.corex_issue_columns(labels) == ["war", "trade"]


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

    def test_the_degenerate_interval_predicate_matches_twelve_rows_all_in_1785(
        self, shipped
    ):
        """A bootstrap over n clusters has only C(2n-1, n) distinct resamples —
        3 at n=2 — so a 1785 series with zero paragraphs in BOTH speeches
        returns the same replicate every draw and its interval collapses to
        lo == hi == 0. That is NOT the same object as the many legitimate
        zero-width cells where sixty speeches genuinely never touched an issue,
        and the module docstring publishes the count. Recomputed here."""
        a = shipped[shipped["surface"] == B.COREX_SURFACE]
        degenerate = a[(a["ci_status"] == "low_cluster_caution") & (a["lo"] == a["hi"])]
        assert len(degenerate) == 12
        assert set(degenerate["period"]) == {"1785"}
        assert (degenerate["lo"] == 0).all()
        assert "12 of the 1,078 Surface A rows" in _prose(B.__doc__)

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
            live = shipped[shipped["ci_status"] == status]
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
