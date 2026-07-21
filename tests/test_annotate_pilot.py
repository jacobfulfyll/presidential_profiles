"""Target #2 — the pilot draw is deterministic and cannot silently shift.

The ~$1 pilot is 20 era-stratified speeches; the whole auto-proceed gate hangs
off it. If the sample quietly moved (a pandas upgrade, a re-sort, a corpus
refresh), the pilot would validate a DIFFERENT 20 speeches than the ones a
reviewer signed off on. So the exact draw is pinned as a snapshot against the
real corpus, and the stratification logic is pinned order-independently against
a synthetic frame.
"""

from __future__ import annotations

import json

import pandas as pd

from presidential_profiles import annotate as A
from presidential_profiles.corpus import load

# The frozen pilot draw: exactly 20 speeches, chronological, Washington's
# 1796 Farewell -> Biden's 2021 Afghanistan remarks. Snapshotted so any shift
# in _pilot_speeches (or the corpus it draws from) trips this test loudly.
_EXPECTED_PILOT_DOC_NAMES = [
    "/the-presidency/presidential-speeches/september-19-1796-farewell-address",
    "/the-presidency/presidential-speeches/december-4-1827-third-annual-message",
    "/the-presidency/presidential-speeches/april-9-1841-address-upon-assuming-office-president-united",
    "/the-presidency/presidential-speeches/january-18-1854-proclamation",
    "/the-presidency/presidential-speeches/august-26-1863-public-letter-james-conkling",
    "/the-presidency/presidential-speeches/march-5-1877-inaugural-address",
    "/the-presidency/presidential-speeches/june-4-1889-statement-johnstown-flood",
    "/the-presidency/presidential-speeches/april-11-1898-message-regarding-cuban-civil-war",
    "/the-presidency/presidential-speeches/january-28-1915-veto-immigration-legislation",
    "/the-presidency/presidential-speeches/december-2-1930-second-state-union-address",
    "/the-presidency/presidential-speeches/january-15-1953-farewell-address",
    "/the-presidency/presidential-speeches/december-11-1961-remarks-un-delegation-women",
    "/the-presidency/presidential-speeches/october-6-1966-press-conference",
    "/the-presidency/presidential-speeches/february-2-1967-press-conference",
    "/the-presidency/presidential-speeches/april-18-1977-address-nation-energy",
    "/the-presidency/presidential-speeches/march-4-1987-address-nation-iran-contra",
    "/the-presidency/presidential-speeches/january-19-1999-state-union-address",
    "/the-presidency/presidential-speeches/december-1-2009-speech-strategy-afghanistan-and-pakistan",
    "/the-presidency/presidential-speeches/january-28-2014-2014-state-union-address",
    "august-16-2021-remarks-situation-afghanistan",
]


# ---------------------------------------------------------------------------
# real-corpus snapshot: the exact 20 doc_names must not move
# ---------------------------------------------------------------------------


def test_pilot_draws_exactly_the_pinned_twenty_doc_names_from_the_real_corpus():
    sample = A._pilot_speeches(load())
    assert len(sample) == A.PILOT_N == 20
    assert list(sample["doc_name"]) == _EXPECTED_PILOT_DOC_NAMES


def test_pilot_spans_washington_to_the_present_and_is_deterministic():
    a = A._pilot_speeches(load())
    b = A._pilot_speeches(load())
    # deterministic: same draw every call
    assert list(a["doc_name"]) == list(b["doc_name"])
    # era span: the sample reaches from the 18th century to the 21st
    assert a.iloc[0]["year"] == 1796
    assert a.iloc[-1]["year"] == 2021
    assert a["year"].is_monotonic_increasing  # returned in chronological order


# ---------------------------------------------------------------------------
# synthetic frame: stratification is one-per-era and order-independent
# ---------------------------------------------------------------------------


def _synthetic_chronological(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "doc_name": [f"doc{i:03d}" for i in range(n_rows)],
            "date": [pd.Timestamp("1800-01-01") + pd.Timedelta(days=i) for i in range(n_rows)],
        }
    )


def test_pilot_picks_exactly_one_speech_from_each_equal_frequency_stratum():
    base = _synthetic_chronological(40)
    picked = A._pilot_speeches(base, n=5)
    assert len(picked) == 5

    # Each pick must land in a distinct equal-frequency chronological stratum
    # ((position * n) // total) — i.e. one speech per era block, not five from
    # one corner of the timeline.
    positions = [int(d[3:]) for d in picked["doc_name"]]
    strata = [(p * 5) // 40 for p in positions]
    assert sorted(strata) == [0, 1, 2, 3, 4]


def test_pilot_is_independent_of_the_frames_incoming_row_order():
    base = _synthetic_chronological(40)
    shuffled = base.sample(frac=1, random_state=7).reset_index(drop=True)
    # Prove the shuffle actually reordered — otherwise "order-independent" is
    # vacuously true and the test proves nothing.
    assert list(shuffled["doc_name"]) != list(base["doc_name"])

    from_sorted = A._pilot_speeches(base, n=5)
    from_shuffled = A._pilot_speeches(shuffled, n=5)
    # Same 20-analogue draw regardless of incoming order — the sort-by-(date,
    # doc_name) inside the sampler is what makes the pilot reproducible.
    assert list(from_sorted["doc_name"]) == list(from_shuffled["doc_name"])


def test_pilot_returns_all_speeches_when_corpus_is_smaller_than_the_sample():
    tiny = _synthetic_chronological(3)
    assert list(A._pilot_speeches(tiny, n=20)["doc_name"]) == list(tiny["doc_name"])


# ---------------------------------------------------------------------------
# CLI wiring: `dry-run --pilot` actually narrows to exactly the pilot 20
# ---------------------------------------------------------------------------


def test_dry_run_pilot_flag_selects_exactly_the_twenty_pilot_speeches(args, redirect_annotation_dirs):
    """The helper snapshot proves the sampler; this proves cmd_dry_run's --pilot
    branch feeds that sample through (one request per pilot speech), offline."""
    A.cmd_dry_run(args(run_id="pilotdry", spec=["placeholder"], pilot=True))

    index = json.loads((A.ann.RUNS_DIR / "pilotdry" / "requests_index.json").read_text())
    doc_names = {meta["doc_name"] for meta in index.values()}
    assert doc_names == set(_EXPECTED_PILOT_DOC_NAMES)
    assert len(index) == 20  # one placeholder request per pilot speech
