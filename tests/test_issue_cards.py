"""Tests for the issue-card scoring reworked in ``profiles.py``.

Background: an issue used to earn a president a card only on the ERA-RELATIVE
axis - ``rel_<n> >= 0.75 and share_<n> * n_paragraphs >= 4``. That silently
dropped issues that consumed a presidency but consumed its era equally: a
wartime president talking war at wartime rates scores ~0 rel and vanished from
his own profile. The fix judges an issue on EITHER of two normalised axes
(1.0 == exactly the bar):

  - ``rel_strength = rel_<n> / REL_DISTINCT_PP``            (era-distinctive)
  - ``raw_strength = (share_<n> / base[n]) / RAW_ELEVATED_MULT``  (above the
    corpus-wide base rate), 0 when ``base == 0``

An issue is eligible iff ``share_<n> * n_paras >= MIN_ISSUE_PARAS`` AND
``max(rel_strength, raw_strength) >= 1.0``; cards rank by ``-max(...)``. A card
is a ``topic_of_day`` iff ``raw_strength >= 1.0 and abs(rel) < REL_DISTINCT_PP``
(loud in raw terms, but no louder than its own era). Each president also carries
``n_speeches`` / ``n_paragraphs`` and ``low_confidence = n_speeches <
SPARSE_MIN_SPEECHES``.

What these tests pin:
  1. ``issue_base_rates`` is PARAGRAPH-WEIGHTED, not a mean of per-president
     shares - a thin president (Garfield, 29 paras) must not define "normal".
  2. The eligibility bug fix, its two axes, the MIN_ISSUE_PARAS floor, the
     ranking-by-stronger-axis, the topic_of_day predicate, and the base==0
     divide-by-zero guard.
  3. The low_confidence boundary at SPARSE_MIN_SPEECHES.

``issue_base_rates``, ``issue_strengths`` and ``is_topic_of_day`` are real
module-level functions, exercised directly here (the scalar adapters below only
marshal hand-computable values into them). The eligibility gate and the
``low_confidence`` flag remain INLINE inside ``issue_cards()`` (which reads the
real 36k-row parquet and is coupled to the full pipeline), so per CLAUDE.md
those two are re-expressed here against the real imported thresholds and pinned
on hand-computable synthetic values.

Because ``issue_strengths``/``is_topic_of_day`` are now importable, the strength
and topic-of-day formulas live in exactly one place: a change to a formula's
*shape* (not just a *constant*) now fails these tests. The eligibility
composition and the ``low_confidence`` boundary are additionally pinned
end-to-end by the real-function harness in ``TestRealIssueCardsRawAxis``.
"""

import pandas as pd
import pytest

from presidential_profiles import issues, profiles
from presidential_profiles.profiles import (
    MIN_ISSUE_PARAS,
    REL_DISTINCT_PP,
    SPARSE_MIN_SPEECHES,
    is_topic_of_day,
    issue_base_rates,
    issue_strengths,
)


# --------------------------------------------------------------------------- #
# Scalar adapters onto the real module-level functions. ``_strengths`` and
# ``_is_topic_of_day`` hold NO formula of their own - they build a one-issue
# ``prow``/``base`` and delegate to ``profiles.issue_strengths`` /
# ``profiles.is_topic_of_day``, so a change to a formula's *shape* (not just a
# constant) now fails these tests. ``_is_eligible`` composes the (still-inline)
# eligibility gate from the real strengths.
# --------------------------------------------------------------------------- #
def _strengths(rel: float, share: float, base: float) -> tuple[float, float]:
    """Scalar adapter onto the real ``profiles.issue_strengths``."""
    return issue_strengths({"rel_x": rel, "share_x": share}, "x", {"x": base})


def _is_eligible(rel: float, share: float, base: float, n_paras: float) -> bool:
    """Mirror of the (un-hoisted) eligibility comprehension in issue_cards,
    built from the real strengths."""
    return (share * n_paras >= MIN_ISSUE_PARAS
            and max(_strengths(rel, share, base)) >= 1.0)


def _is_topic_of_day(rel: float, share: float, base: float) -> bool:
    """Scalar adapter onto the real ``profiles.is_topic_of_day``."""
    _, raw_strength = _strengths(rel, share, base)
    return is_topic_of_day(raw_strength, rel)


def _old_gate_eligible(rel: float, share: float, n_paras: float) -> bool:
    """The PRE-FIX gate (era-relative only), kept so the bug-fix test can assert
    the exact case the old rule dropped: ``rel >= 0.75 and share*n_paras >= 4``."""
    return rel >= 0.75 and share * n_paras >= 4


# --------------------------------------------------------------------------- #
# 1. issue_base_rates: paragraph-weighted, not a mean of per-president shares
# --------------------------------------------------------------------------- #
class TestIssueBaseRates:
    def test_base_rate_is_paragraph_weighted_not_mean_of_shares(self):
        """A thin president with a very high share must NOT drag the base rate
        up: with a heavy 0.10-share president (1000 paras) and a thin 0.90-share
        president (100 paras), the paragraph-weighted base is
        (0.10*1000 + 0.90*100)/1100 = 0.1727 - near the heavy president, nowhere
        near the naive share-mean of 0.50. This is the whole reason base rates
        are weighted."""
        issue_df = pd.DataFrame(
            {
                "share_Trade": [0.10, 0.90],
                "n_paragraphs": [1000.0, 100.0],
            },
            index=pd.Index(["Heavy President", "Thin President"], name="president"),
        )

        base = issue_base_rates(issue_df, ["Trade"])

        weighted = base["Trade"]
        naive_mean_of_shares = issue_df["share_Trade"].mean()  # 0.50
        assert weighted == pytest.approx(190 / 1100)  # 0.172727...
        # The thin president's 0.90 barely moves the base off the heavy 0.10.
        assert weighted < 0.25
        assert weighted < naive_mean_of_shares
        assert naive_mean_of_shares == pytest.approx(0.50)

    def test_base_rates_returned_independently_per_issue(self):
        """Each name in ``display`` gets its own weighted base, computed only
        from its own share column, on a hand-computable frame:
          Trade = (0.5*10 + 0.1*30)/40 = 0.20
          War   = (0.2*10 + 0.4*30)/40 = 0.35"""
        issue_df = pd.DataFrame(
            {
                "share_Trade": [0.5, 0.1],
                "share_War": [0.2, 0.4],
                "share_Ignored": [0.9, 0.9],  # not in display -> must be absent
                "n_paragraphs": [10.0, 30.0],
            },
            index=pd.Index(["A", "B"], name="president"),
        )

        base = issue_base_rates(issue_df, ["Trade", "War"])

        assert set(base) == {"Trade", "War"}
        assert base["Trade"] == pytest.approx(0.20)
        assert base["War"] == pytest.approx(0.35)


# --------------------------------------------------------------------------- #
# 2. Eligibility, ranking, topic_of_day (mirrored on synthetic scalars)
# --------------------------------------------------------------------------- #
class TestEligibility:
    def test_era_equal_but_raw_elevated_issue_is_now_eligible(self):
        """THE BUG FIX. An issue at rel~0 (indistinguishable from its era) but
        share >= 1.5*base (loud in raw terms) is now eligible on the raw axis -
        the exact case the old rel-only gate dropped. base=0.10, share=0.20 ->
        raw_strength = (0.20/0.10)/1.5 = 1.333 >= 1.0."""
        rel, share, base, n_paras = 0.0, 0.20, 0.10, 30.0

        assert _is_eligible(rel, share, base, n_paras) is True
        # ...and it would have been excluded before, purely for being era-equal.
        assert _old_gate_eligible(rel, share, n_paras) is False
        # It qualifies on raw, not rel.
        rel_s, raw_s = _strengths(rel, share, base)
        assert raw_s >= 1.0
        assert rel_s < 1.0

    def test_era_distinctive_but_not_raw_issue_still_eligible(self):
        """The other axis: rel >= 0.75 (distinctive) even though share is below
        1.5*base stays eligible on the rel axis. base=0.10, rel=1.0 ->
        rel_strength=1.333; share=0.12 -> raw_strength=0.8."""
        rel, share, base, n_paras = 1.0, 0.12, 0.10, 50.0

        assert _is_eligible(rel, share, base, n_paras) is True
        rel_s, raw_s = _strengths(rel, share, base)
        assert rel_s >= 1.0
        assert raw_s < 1.0

    def test_issue_below_both_bars_is_excluded(self):
        """Below the era bar AND below the raw bar -> no card. rel=0.5
        (rel_strength 0.667) and share=0.12 (raw_strength 0.8); the floor is
        satisfied (0.12*50=6 >= 4) so it is the strength gate that excludes it."""
        rel, share, base, n_paras = 0.5, 0.12, 0.10, 50.0

        assert share * n_paras >= MIN_ISSUE_PARAS  # floor is NOT the reason
        assert max(_strengths(rel, share, base)) < 1.0
        assert _is_eligible(rel, share, base, n_paras) is False

    def test_min_issue_paras_floor_excludes_a_strong_but_tiny_issue(self):
        """The MIN_ISSUE_PARAS floor protects one-speech presidents from a stray
        metaphor: an issue blazing on BOTH axes is still excluded when
        share*n_paras < 4. rel=5.0, share=0.5 (both strengths huge) but
        n_paras=3 -> share*n_paras=1.5 < 4."""
        rel, share, base, n_paras = 5.0, 0.5, 0.10, 3.0

        assert max(_strengths(rel, share, base)) >= 1.0  # strength would pass
        assert share * n_paras < MIN_ISSUE_PARAS  # ...but the floor fails it
        assert _is_eligible(rel, share, base, n_paras) is False

    def test_base_zero_yields_raw_strength_zero_no_divide_by_zero(self):
        """The base==0 guard must return raw_strength 0 rather than raising.
        With base 0 an issue can only qualify on the rel axis."""
        rel_s, raw_s = _strengths(rel=1.0, share=0.5, base=0.0)
        assert raw_s == 0.0
        assert rel_s == pytest.approx(1.0 / REL_DISTINCT_PP)
        # rel carries eligibility when base is 0...
        assert _is_eligible(rel=1.0, share=0.5, base=0.0, n_paras=100.0) is True
        # ...and nothing rescues an era-equal issue when base is 0.
        assert _is_eligible(rel=0.0, share=0.5, base=0.0, n_paras=100.0) is False

    def test_cards_rank_by_their_stronger_axis(self):
        """Ranking is by ``-max(strengths)``, so a defining-but-ordinary issue
        (high raw, low rel) and a distinctive-but-small one (high rel, low raw)
        BOTH surface, each ordered by its stronger axis. Defining's max
        (raw 2.0) outranks Distinctive's max (rel 1.6)."""
        issues = {
            "Defining": dict(rel=0.1, share=0.30, base=0.10, n_paras=20.0),
            "Distinctive": dict(rel=1.2, share=0.11, base=0.10, n_paras=50.0),
        }
        eligible = [n for n, p in issues.items() if _is_eligible(**p)]
        eligible.sort(key=lambda n: -max(
            _strengths(issues[n]["rel"], issues[n]["share"], issues[n]["base"])))

        assert eligible == ["Defining", "Distinctive"]
        # Each leads on a different axis - the point of the two-axis design.
        d_rel, d_raw = _strengths(0.1, 0.30, 0.10)
        x_rel, x_raw = _strengths(1.2, 0.11, 0.10)
        assert d_raw > d_rel  # Defining is a raw-attention card
        assert x_rel > x_raw  # Distinctive is an era-relative card


class TestTopicOfDay:
    def test_true_when_raw_elevated_and_era_indistinct(self):
        """The positive case: loud in raw terms but no louder than its era.
        share=0.20/base=0.10 -> raw_strength 1.333 >= 1.0, and abs(rel)=0.2 <
        0.75."""
        assert _is_topic_of_day(rel=0.2, share=0.20, base=0.10) is True

    def test_false_when_raw_elevated_but_also_era_distinctive(self):
        """Not a topic-of-the-day if the president ALSO stands out from their
        era: raw_strength 2.0 >= 1.0 but abs(rel)=3.0 is not < 0.75, so the
        subject is genuinely theirs, not just in the air."""
        assert _is_topic_of_day(rel=3.0, share=0.30, base=0.10) is False

    def test_false_when_era_distinctive_but_not_raw_elevated(self):
        """The rel axis alone does not make a topic-of-the-day: raw_strength 0.8
        < 1.0 despite a large rel."""
        assert _is_topic_of_day(rel=2.0, share=0.12, base=0.10) is False

    def test_false_at_the_rel_boundary(self):
        """The era band is strict (``abs(rel) < REL_DISTINCT_PP``): an issue
        raw-elevated but sitting exactly at rel=0.75 is era-distinctive, not a
        topic-of-the-day."""
        assert (abs(0.75) < REL_DISTINCT_PP) is False  # documents the strict bar
        assert _is_topic_of_day(rel=REL_DISTINCT_PP, share=0.20, base=0.10) is False


# --------------------------------------------------------------------------- #
# 3. low_confidence boundary at SPARSE_MIN_SPEECHES
# --------------------------------------------------------------------------- #
def _low_confidence(n_speeches: int) -> bool:
    """Mirror of the per-president flag: ``n_speeches < SPARSE_MIN_SPEECHES``."""
    return bool(n_speeches < SPARSE_MIN_SPEECHES)


class TestLowConfidence:
    @pytest.mark.parametrize(
        "n_speeches, expected",
        [
            (1, True),   # Garfield-thin
            (4, True),   # just under the bar
            (5, False),  # exactly at the bar -> confident
            (6, False),
        ],
    )
    def test_low_confidence_boundary(self, n_speeches, expected):
        assert _low_confidence(n_speeches) is expected


# --------------------------------------------------------------------------- #
# 4. Real-function anchor: drive the ACTUAL profiles.issue_cards over synthetic
#    parquets (same monkeypatch harness as test_keyed_merge) so the two-axis
#    bug fix, the topic_of_day predicate, the empty-eligible data path and the
#    low_confidence flag are pinned against PRODUCTION code, not just the inline
#    mirrors above. The mirrors catch a drifting *constant* (they import the
#    thresholds); this test catches a drifting *formula shape* in the source,
#    which the mirrors by construction cannot. No 36k corpus / CorEx: issue_df
#    and the two-row parquet are all hand-computable.
# --------------------------------------------------------------------------- #
_REAL_WAR_SENTENCE = (
    "The war tested our army and navy as our soldiers and troops answered the "
    "solemn call of the national defense."
)
_REAL_CALM = (
    "A quiet paragraph about gardens and gentle weather with nothing at all "
    "notable happening on this particular afternoon."
)


class TestRealIssueCardsRawAxis:
    def _install(self, monkeypatch, tmp_path, paras, labels):
        p_path = tmp_path / "paragraphs.parquet"
        l_path = tmp_path / "paragraph_issues.parquet"
        paras.to_parquet(p_path, index=False)
        labels.to_parquet(l_path, index=False)
        monkeypatch.setattr(profiles, "PARAGRAPHS_PATH", p_path)
        monkeypatch.setattr(issues, "PARA_LABELS_PATH", l_path)

    def test_raw_axis_only_president_gets_topic_of_day_card_via_real_fn(
        self, monkeypatch, tmp_path
    ):
        """THE BUG FIX, against the real code. Raw President talks War at
        share 0.36 but at rel 0 (exactly their era's level); Quiet President
        (share 0.10) drags the base to (0.36*20 + 0.10*20)/40 = 0.23. Raw's
        raw_strength = (0.36/0.23)/1.5 = 1.04 >= 1.0 -> a card on the RAW axis
        alone (rel_strength 0). Under the pre-fix rel>=0.75 gate Raw scored
        nothing. Because rel is era-equal, the card is flagged topic_of_day.
        Quiet, only at the base rate, earns no card - the empty-eligible path."""
        paras = pd.DataFrame({
            "doc_name": ["raw-doc", "quiet-doc"],
            "para_idx": [0, 0],
            "text": [_REAL_WAR_SENTENCE, _REAL_CALM],
            "word_count": [len(_REAL_WAR_SENTENCE.split()), len(_REAL_CALM.split())],
        })
        labels = pd.DataFrame({
            "doc_name": ["raw-doc", "quiet-doc"],
            "para_idx": [0, 0],
            "president": ["Raw President", "Quiet President"],
            "year": [1900, 1950],
            "War & military": [True, False],
            "Discovered 5": [False, False],
        })
        issue_df = pd.DataFrame(
            {
                "rel_War & military": [0.0, 0.0],   # era-equal on BOTH
                "share_War & military": [0.36, 0.10],
                "rel_Discovered 5": [0.0, 0.0],
                "share_Discovered 5": [0.0, 0.0],
                "n_paragraphs": [20.0, 20.0],
            },
            index=pd.Index(["Raw President", "Quiet President"], name="president"),
        )
        titles = pd.DataFrame({
            "doc_name": ["raw-doc", "quiet-doc"],
            "president": ["Raw President", "Quiet President"],
            "title": ["Address: The War Years", "Address: A Quiet Peace"],
            "year": [1900, 1950],
        })
        distinctive = pd.DataFrame({"president": [], "term": [], "rank": [], "z": []})

        self._install(monkeypatch, tmp_path, paras, labels)
        out = profiles.issue_cards(
            df=titles,
            distinctive=distinctive,
            issue_df=issue_df,
            issue_meta={"issues": ["War & military"]},
        )

        war = [c for c in out["Raw President"]["cards"]
               if c["issue"] == "War & military"]
        assert len(war) == 1                     # eligible on the raw axis alone
        assert war[0]["rel"] == 0.0              # ...which the old gate dropped
        assert war[0]["topic_of_day"] is True    # loud in raw terms, era-equal
        assert out["Raw President"]["low_confidence"] is True   # 1 speech < 5
        # A president sitting at the base rate earns no card: empty-eligible path.
        assert out["Quiet President"]["cards"] == []
