"""The three genre treatments in `register.py`.

The 19th-century annual message was a departmental catalog that forced every
president to touch every issue, and annual messages fall from ~40% of early
speeches to ~15% of modern ones. So the genre mix is a first-order rival
explanation for every trend the module reports, and each trend is computed
`raw`, `sotu_only`, and `genre_standardized`.

Standardization is direct: each (era x reference genre) cell keeps the era's
total paragraph mass but has it redistributed in the reference proportions, and
a cell thinner than `MIN_CELL_SPEECHES` is dropped with the remaining reference
weights renormalized. The renormalization is where a silent error would live —
weights that no longer sum to 1 would rescale an era's whole estimate — so it is
checked as a closed-form arithmetic identity rather than a smoke test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import register as R

SOTU = R.SOTU_TYPE
INAUGURAL = "inaugural_address"
SPECIAL = "special_message_to_congress"
VETO = "veto_or_signing_statement"
REMARKS = "public_remarks_or_address"   # a modern-only genre, not in the reference mix


def _speeches(rows):
    """rows: (era, speech_type) pairs."""
    return pd.DataFrame(
        [{"era": era, "speech_type": genre} for era, genre in rows]
    )


def _all_four(era, per_genre=1):
    return [
        (era, genre)
        for genre in (INAUGURAL, SPECIAL, SOTU, VETO)
        for _ in range(per_genre)
    ]


# =========================================================================== #
class TestReferenceGenreMix:
    def test_weights_sum_to_one_over_the_reference_genres_in_order(self):
        speeches = _speeches(_all_four(1800) + _all_four(1830))
        mix = R.reference_genre_mix(speeches, np.ones(len(speeches)))

        assert list(mix.index) == list(R.REFERENCE_GENRES)
        assert mix.sum() == pytest.approx(1.0)

    def test_weights_are_pooled_paragraph_shares(self):
        """Hand-computable: paragraph mass 60/20/10/10 across the four durable
        genres gives weights 0.6/0.2/0.1/0.1."""
        speeches = _speeches(_all_four(1800) + _all_four(1830))
        # Two eras x four genres, ordered inaugural, special, sotu, veto.
        counts = np.array([10.0, 5.0, 30.0, 5.0, 10.0, 5.0, 30.0, 5.0])

        mix = R.reference_genre_mix(speeches, counts)

        assert mix[SOTU] == pytest.approx(0.6)
        assert mix[INAUGURAL] == pytest.approx(0.2)
        assert mix[SPECIAL] == pytest.approx(0.1)
        assert mix[VETO] == pytest.approx(0.1)

    def test_mix_is_by_paragraph_mass_not_speech_count(self):
        """The reweighting redistributes PARAGRAPH mass, so the reference has to
        be measured in paragraphs. A speech-count mix would score a one-line
        signing statement as heavily as a 400-paragraph annual message."""
        rows = [(1800, SOTU)] + [(1800, INAUGURAL)] * 9 + [(1800, SPECIAL), (1800, VETO)]
        rows += [(1830, genre) for genre in (SOTU, INAUGURAL, SPECIAL, VETO)]
        speeches = _speeches(rows)
        counts = np.array([400.0] + [1.0] * 9 + [1.0, 1.0] + [1.0, 1.0, 1.0, 1.0])

        mix = R.reference_genre_mix(speeches, counts)

        assert mix[SOTU] > 0.9                 # 401 of 417 paragraphs
        assert mix[SOTU] > mix[INAUGURAL]      # despite 2 speeches vs 10

    def test_a_modern_only_genre_does_not_enter_the_mix(self):
        """Six genres do not exist before the 20th century. Standardizing to a
        mix that includes them would require inventing counterfactual cells,
        which is the confound rather than a control for it."""
        speeches = _speeches(_all_four(1800) + _all_four(1830) + [(1830, REMARKS)])
        counts = np.ones(len(speeches))

        mix = R.reference_genre_mix(speeches, counts)

        assert REMARKS not in mix.index
        assert mix.sum() == pytest.approx(1.0)

    def test_a_new_genre_spanning_every_era_raises_rather_than_redefining_the_mix(self):
        """The reference set is pinned, not derived, so a corpus change surfaces
        as an exception instead of silently changing what "genre-standardized"
        means between runs."""
        speeches = _speeches(
            _all_four(1800) + _all_four(1830) + [(1800, REMARKS), (1830, REMARKS)]
        )

        with pytest.raises(ValueError, match="present in every era has changed"):
            R.reference_genre_mix(speeches, np.ones(len(speeches)))

    def test_a_reference_genre_missing_from_one_era_raises(self):
        speeches = _speeches(_all_four(1800) + [(1830, g) for g in (INAUGURAL, SPECIAL, SOTU)])

        with pytest.raises(ValueError, match="present in every era has changed"):
            R.reference_genre_mix(speeches, np.ones(len(speeches)))

    def test_reference_genres_are_the_four_durable_genres(self):
        """Spec anchor, decoupled from the constant every other test reads. The
        four genres named here are the ones present in all nine 30-year eras;
        every test above indexes `R.REFERENCE_GENRES`, so a regression that
        swapped one for a modern genre would pass them all."""
        assert set(R.REFERENCE_GENRES) == {
            "inaugural_address",
            "special_message_to_congress",
            "state_of_the_union_or_annual_message",
            "veto_or_signing_statement",
        }
        assert R.SOTU_TYPE == "state_of_the_union_or_annual_message"
        assert R.SOTU_TYPE in R.REFERENCE_GENRES
        assert R.GENRE_TREATMENTS == ("raw", "sotu_only", "genre_standardized")
        assert R.YEAR_TREATMENTS == ("raw", "sotu_only")


# =========================================================================== #
class TestGenreCells:
    def _speeches_with_year(self, rows):
        frame = _speeches([(era, genre) for era, genre, _ in rows])
        frame["year"] = [year for _, _, year in rows]
        return frame

    def test_marks_a_cell_included_at_exactly_the_minimum(self):
        """The boundary. `MIN_CELL_SPEECHES` is a `>=` threshold, so a cell with
        exactly three speeches is kept — one fewer is dropped."""
        speeches = _speeches(
            [(1800, SOTU)] * 3 + [(1800, INAUGURAL)] * 2
        )

        cells = R.genre_cells(speeches).set_index("speech_type")

        assert cells.loc[SOTU, "n_speeches"] == 3
        assert bool(cells.loc[SOTU, "included"]) is True
        assert cells.loc[INAUGURAL, "n_speeches"] == 2
        assert bool(cells.loc[INAUGURAL, "included"]) is False

    def test_reports_only_reference_genres(self):
        """The published table exists so a reader can see WHICH cells the
        standardized estimate rests on; a modern-only genre is not one of them."""
        speeches = _speeches([(1800, SOTU)] * 3 + [(1980, REMARKS)] * 20)

        cells = R.genre_cells(speeches)

        assert REMARKS not in set(cells["speech_type"])
        assert set(cells["speech_type"]) == {SOTU}

    def test_a_genre_absent_from_an_era_simply_has_no_row(self):
        """Absent is not the same as thin: an era that never produced a veto
        message gets no veto cell at all, rather than a zero-count included one."""
        speeches = _speeches([(1800, SOTU)] * 3 + [(1830, SOTU)] * 3 + [(1830, VETO)] * 3)

        cells = R.genre_cells(speeches)
        era_1800 = cells[cells["era"] == 1800]

        assert set(era_1800["speech_type"]) == {SOTU}
        assert len(cells) == 3

    def test_can_group_by_year_instead_of_era(self):
        speeches = self._speeches_with_year(
            [(1800, SOTU, 1805)] * 3 + [(1800, SOTU, 1806)] * 2
        )

        cells = R.genre_cells(speeches, group_col="year").set_index("year")

        assert bool(cells.loc[1805, "included"]) is True
        assert bool(cells.loc[1806, "included"]) is False

    def test_thresholds_match_the_documented_values(self):
        """Spec anchor. Every behavioural test above stays on the same side of
        these lines whichever way they move, so the values themselves are pinned
        as literals."""
        assert R.MIN_CELL_SPEECHES == 3
        assert R.MIN_YEAR_PARAGRAPHS == 30


# =========================================================================== #
def _mix(**weights):
    full = {INAUGURAL: 0.0, SPECIAL: 0.0, SOTU: 0.0, VETO: 0.0}
    full.update(weights)
    return pd.Series(full).reindex(list(R.REFERENCE_GENRES))


class TestCellPlan:
    def _panel(self, register_panel, rows):
        """rows: (speech_type, n_paragraphs) — all in one era."""
        return register_panel([
            {
                "doc_name": f"d{i}",
                "year": 1805,
                "speech_type": genre,
                "n_paragraphs": float(paras),
                "para_words": float(paras * 100),
            }
            for i, (genre, paras) in enumerate(rows)
        ])

    def test_raw_is_a_single_unweighted_cell_over_every_speech(self, register_panel):
        panel = self._panel(register_panel, [(SOTU, 10), (REMARKS, 5), (VETO, 3)])
        positions = np.arange(3)

        cells = R._cell_plan(panel, positions, "raw", _mix(**{SOTU: 1.0}))

        assert len(cells) == 1
        members, weight = cells[0]
        assert members.tolist() == [0, 1, 2]
        assert weight == 1.0

    def test_sotu_only_keeps_the_annual_messages_and_nothing_else(self, register_panel):
        """The one genre spanning all 240 years — a trend that survives inside it
        is about presidents, not about the changing form of presidential speech."""
        panel = self._panel(
            register_panel, [(SOTU, 10), (REMARKS, 5), (SOTU, 8), (VETO, 3)]
        )

        cells = R._cell_plan(panel, np.arange(4), "sotu_only", _mix(**{SOTU: 1.0}))

        assert len(cells) == 1
        members, weight = cells[0]
        assert members.tolist() == [0, 2]
        assert weight == 1.0

    def test_sotu_only_returns_no_cell_when_the_group_has_no_annual_message(
        self, register_panel
    ):
        """A modern year of nothing but press conferences produces no
        `sotu_only` estimate at all — not a zero, and not an estimate quietly
        computed over the other genres."""
        panel = self._panel(register_panel, [(REMARKS, 5), (VETO, 3)])

        assert R._cell_plan(panel, np.arange(2), "sotu_only", _mix()) == []

    def test_raw_returns_no_cell_for_an_empty_group(self, register_panel):
        panel = self._panel(register_panel, [(SOTU, 10)])

        assert R._cell_plan(panel, np.array([], dtype=int), "raw", _mix()) == []

    def test_unknown_treatment_raises(self, register_panel):
        panel = self._panel(register_panel, [(SOTU, 10)])

        with pytest.raises(ValueError, match="unknown genre treatment"):
            R._cell_plan(panel, np.arange(1), "standardised", _mix())

    # ------------------------------------------------------------------ #
    # genre_standardized
    # ------------------------------------------------------------------ #
    def test_standardized_weights_hold_paragraph_mass_and_split_it_by_the_mix(
        self, register_panel
    ):
        """Direct standardization, checked as arithmetic. Four thick cells with
        reference weights 0.1/0.2/0.6/0.1 and 60 paragraphs of mass: each cell's
        weighted mass must be its reference SHARE of the group's total, and the
        shares must add back to the original mass."""
        panel = self._panel(
            register_panel,
            [(SOTU, 10)] * 3 + [(INAUGURAL, 5)] * 3 + [(SPECIAL, 2)] * 3
            + [(VETO, 1)] * 3,
        )
        mix = _mix(**{INAUGURAL: 0.1, SPECIAL: 0.2, SOTU: 0.6, VETO: 0.1})
        total_paras = 3 * 10 + 3 * 5 + 3 * 2 + 3 * 1

        cells = R._cell_plan(panel, np.arange(12), "genre_standardized", mix)
        paras = panel.scalars["n_paragraphs"].to_numpy()
        mass = {
            panel.speeches["speech_type"].to_numpy()[members[0]]:
                weight * paras[members].sum()
            for members, weight in cells
        }

        assert len(cells) == 4
        assert sum(mass.values()) == pytest.approx(total_paras)
        assert mass[SOTU] == pytest.approx(0.6 * total_paras)
        assert mass[INAUGURAL] == pytest.approx(0.1 * total_paras)
        assert mass[SPECIAL] == pytest.approx(0.2 * total_paras)
        assert mass[VETO] == pytest.approx(0.1 * total_paras)

    def test_dropping_a_thin_cell_renormalizes_the_remaining_weights_to_one(
        self, register_panel
    ):
        """The 1770-era case: two of the four reference genres are too thin, so
        the standardized estimate rests on a renormalized two-genre mix. The
        surviving weights 0.6 and 0.1 must be rescaled to 6/7 and 1/7 — leaving
        them at 0.6 and 0.1 would silently shrink the era's estimate to 70% of
        the group's paragraph mass."""
        panel = self._panel(
            register_panel,
            [(SOTU, 10)] * 4 + [(INAUGURAL, 5)] * 4 + [(SPECIAL, 10)] * 2,
        )
        mix = _mix(**{INAUGURAL: 0.1, SPECIAL: 0.2, SOTU: 0.6, VETO: 0.1})

        cells = R._cell_plan(panel, np.arange(10), "genre_standardized", mix)
        paras = panel.scalars["n_paragraphs"].to_numpy()
        genres = panel.speeches["speech_type"].to_numpy()
        mass = {genres[members[0]]: weight * paras[members].sum() for members, weight in cells}
        included_mass = 4 * 10 + 4 * 5   # the two-speech SPECIAL cell is dropped

        assert set(mass) == {SOTU, INAUGURAL}
        assert sum(mass.values()) == pytest.approx(included_mass)
        assert mass[SOTU] / included_mass == pytest.approx(0.6 / 0.7)
        assert mass[INAUGURAL] / included_mass == pytest.approx(0.1 / 0.7)

    def test_a_genre_absent_from_the_group_is_simply_not_a_cell(self, register_panel):
        """Same renormalization path reached the other way: an era that never
        produced a veto message has no veto cell, and the remaining weights carry
        the whole mass."""
        panel = self._panel(register_panel, [(SOTU, 10)] * 3 + [(INAUGURAL, 10)] * 3)
        mix = _mix(**{INAUGURAL: 0.25, SPECIAL: 0.25, SOTU: 0.25, VETO: 0.25})

        cells = R._cell_plan(panel, np.arange(6), "genre_standardized", mix)
        paras = panel.scalars["n_paragraphs"].to_numpy()
        mass = [weight * paras[members].sum() for members, weight in cells]

        assert len(cells) == 2
        assert sum(mass) == pytest.approx(60.0)
        assert mass[0] == pytest.approx(30.0)   # renormalized 0.25 -> 0.5 each
        assert mass[1] == pytest.approx(30.0)

    def test_non_reference_genres_are_excluded_from_the_standardized_estimate(
        self, register_panel
    ):
        """`raw` sees the press conferences, `genre_standardized` must not — that
        difference is the whole point of the treatment."""
        panel = self._panel(
            register_panel, [(SOTU, 10)] * 3 + [(INAUGURAL, 10)] * 3 + [(REMARKS, 10)] * 9
        )
        mix = _mix(**{INAUGURAL: 0.5, SOTU: 0.5})

        cells = R._cell_plan(panel, np.arange(15), "genre_standardized", mix)
        covered = np.concatenate([members for members, _ in cells])

        assert set(panel.speeches["speech_type"].to_numpy()[covered]) == {SOTU, INAUGURAL}
        assert len(covered) == 6

    def test_no_thick_cell_at_all_yields_no_estimate(self, register_panel):
        """Rather than standardizing on one two-speech cell, the group gets no
        standardized estimate — which surfaces downstream as nan, not as a
        confident number resting on two speeches."""
        panel = self._panel(register_panel, [(SOTU, 10)] * 2 + [(INAUGURAL, 10)] * 2)
        mix = _mix(**{INAUGURAL: 0.5, SOTU: 0.5})

        assert R._cell_plan(panel, np.arange(4), "genre_standardized", mix) == []

    def test_a_thick_cell_carrying_no_paragraphs_is_skipped(self, register_panel):
        """Defensive: a zero-paragraph cell would divide by zero in the weight.
        It is dropped and the rest renormalize over it."""
        panel = self._panel(register_panel, [(SOTU, 10)] * 3 + [(INAUGURAL, 0)] * 3)
        mix = _mix(**{INAUGURAL: 0.5, SOTU: 0.5})

        cells = R._cell_plan(panel, np.arange(6), "genre_standardized", mix)
        paras = panel.scalars["n_paragraphs"].to_numpy()

        assert len(cells) == 1
        members, weight = cells[0]
        assert weight * paras[members].sum() == pytest.approx(30.0)

    def test_zero_reference_weight_on_every_present_genre_yields_no_estimate(
        self, register_panel
    ):
        """`total_weight <= 0` guard: a mix that puts all its weight on a genre
        the group lacks cannot standardize it."""
        panel = self._panel(register_panel, [(SOTU, 10)] * 3)
        mix = _mix(**{VETO: 1.0})

        assert R._cell_plan(panel, np.arange(3), "genre_standardized", mix) == []
