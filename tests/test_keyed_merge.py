"""Tests for the (doc_name, para_idx) keyed join between paragraphs and their
issue labels.

Background: ``paragraph_issues.parquet`` used to be aligned to
``paragraphs.parquet`` purely by row order, guarded only by a length check that
could not detect reordering - a silent data-corruption risk. The fix gives the
label table a real ``(doc_name, para_idx)`` key and replaces the positional
alignment with an explicit
``paras.merge(labels, on=["doc_name", "para_idx"], how="inner",
validate="one_to_one")`` in both consumers:

  - ``profiles.issue_cards``            (profiles.py)
  - ``issues_site.write_issue_pages``   (issues_site.py)

and writes ``para_idx`` into the label frame in ``issues.build_issues``
(issues.py).

These tests exercise the real consumer functions where practical, and pin down
the exact merge semantics the fix relies on with small synthetic frames (no
36k-row corpus, no CorEx retrain). The core guarantees under test:

  1. Reordering the label rows is now SELF-HEALING, not silently corrupting
     (keyed merge, left-order preserved).
  2. A non-unique ``(doc_name, para_idx)`` raises ``pd.errors.MergeError``
     instead of silently corrupting - the "loud error" this task is about.
  3. ``build_issues`` writes a ``para_idx`` column keeping the key unique.
  4. The raw ``how="inner"`` merge itself DROPS an unmatched row rather than
     raising (a pandas contract, pinned directly in ``TestKeyedMergeContract``)
     - but both real consumers wrap that merge with a post-merge length check
     (``len(merged) != len(paras) or len(merged) != len(labels)``) that turns a
     dropped row back into a loud ``RuntimeError``, restoring the old
     length-guard's failure mode without reintroducing positional coupling.
"""

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import issues, profiles
from presidential_profiles.issues_site import _issue_quotes


# A clearly quotable, war-anchored sentence: single sentence, 70-300 chars, and
# it trips several "War & military" anchors so _pick_sentence selects it.
WAR_SENTENCE = (
    "The war tested our army and navy as our soldiers and troops answered "
    "the solemn call of the national defense."
)
CALM_GARDENS = (
    "A quiet paragraph about gardens and gentle weather with nothing at all "
    "notable happening on this particular afternoon."
)
CALM_RIVERS = (
    "Another calm passage concerning rivers and harvests drifting across the "
    "wide countryside through the turning of the season."
)

WAR_ANCHORS = ["war", "army", "navy", "troops", "defense", "soldiers", "battle"]


# --------------------------------------------------------------------------- #
# Synthetic fixture builders (schema mirrors the real parquet files)
# --------------------------------------------------------------------------- #
def _paragraphs_frame() -> pd.DataFrame:
    """Mirrors data/paragraphs.parquet: doc_name, para_idx, text, word_count.

    The war paragraph is deliberately NOT first, so a positional (buggy) join
    against reordered labels would mislabel it.
    """
    rows = [
        ("doc-a", 0, CALM_GARDENS),
        ("doc-a", 1, WAR_SENTENCE),
        ("doc-b", 0, CALM_RIVERS),
    ]
    return pd.DataFrame(
        {
            "doc_name": [r[0] for r in rows],
            "para_idx": [r[1] for r in rows],
            "text": [r[2] for r in rows],
            "word_count": [len(r[2].split()) for r in rows],
        }
    )


def _labels_frame() -> pd.DataFrame:
    """Mirrors data/paragraph_issues.parquet (trimmed to the columns the tests
    touch): key + president/year + the two issue-bool columns used as `display`.

    Row order is intentionally NOT the same as _paragraphs_frame(): the war
    label sits first here while the war text sits second in paragraphs. Only a
    keyed join pairs them correctly.
    """
    rows = [
        # (doc_name, para_idx, president,          year, War,  Disc5)
        ("doc-a", 1, "Alpha President", 1900, True, False),
        ("doc-a", 0, "Alpha President", 1901, False, False),
        ("doc-b", 0, "Beta President", 1990, False, False),
    ]
    return pd.DataFrame(
        {
            "doc_name": [r[0] for r in rows],
            "para_idx": [r[1] for r in rows],
            "president": [r[2] for r in rows],
            "year": [r[3] for r in rows],
            "War & military": [r[4] for r in rows],
            "Discovered 5": [r[5] for r in rows],
        }
    )


def _issue_df() -> pd.DataFrame:
    """issue_df indexed by president (as build_profile_data passes it).

    Alpha President is eligible for "War & military" (rel >= 0.75 and
    share * n_paragraphs >= 4); nothing else is eligible.
    """
    return pd.DataFrame(
        {
            "rel_War & military": [5.0, 0.0],
            "share_War & military": [0.5, 0.0],
            "rel_Discovered 5": [0.0, 0.0],
            "share_Discovered 5": [0.0, 0.0],
            "n_paragraphs": [8.0, 8.0],
        },
        index=pd.Index(["Alpha President", "Beta President"], name="president"),
    )


def _titles_df() -> pd.DataFrame:
    """The `df` issue_cards receives: the speeches frame (one row per doc_name),
    used for titles/years AND, since the issue-card rework, for the per-president
    speech count (``df.groupby("president").size()`` -> n_speeches /
    low_confidence). Each doc is attributed to the president who owns its
    paragraphs in _labels_frame(), so the count is well-defined."""
    return pd.DataFrame(
        {
            "doc_name": ["doc-a", "doc-b"],
            "president": ["Alpha President", "Beta President"],
            "title": ["Address: The War Years", "Address: A Quiet Peace"],
            "year": [1900, 1990],
        }
    )


def _empty_distinctive() -> pd.DataFrame:
    return pd.DataFrame({"president": [], "term": [], "rank": [], "z": []})


def _install_parquets(monkeypatch, tmp_path, paras, labels):
    """Write the synthetic frames to parquet and point the module-level paths
    the real functions read from at them."""
    p_path = tmp_path / "paragraphs.parquet"
    l_path = tmp_path / "paragraph_issues.parquet"
    paras.to_parquet(p_path, index=False)
    labels.to_parquet(l_path, index=False)
    monkeypatch.setattr(profiles, "PARAGRAPHS_PATH", p_path)
    monkeypatch.setattr(issues, "PARA_LABELS_PATH", l_path)


def _run_issue_cards() -> dict:
    return profiles.issue_cards(
        df=_titles_df(),
        distinctive=_empty_distinctive(),
        issue_df=_issue_df(),
        issue_meta={"issues": ["War & military"]},
    )


# --------------------------------------------------------------------------- #
# Real-function tests: profiles.issue_cards drives the production keyed merge
# --------------------------------------------------------------------------- #
class TestIssueCardsKeyedMerge:
    def test_output_is_identical_regardless_of_label_row_order(
        self, monkeypatch, tmp_path
    ):
        """Reordering paragraph_issues rows must not change the result: the
        keyed merge re-pairs text with its label by key, so a shuffle is
        self-healing. This is the core anti-corruption guarantee."""
        paras = _paragraphs_frame()

        _install_parquets(monkeypatch, tmp_path, paras, _labels_frame())
        in_order = _run_issue_cards()

        shuffled = _labels_frame().sample(frac=1, random_state=7).reset_index(
            drop=True
        )
        _install_parquets(monkeypatch, tmp_path, paras, shuffled)
        reordered = _run_issue_cards()

        assert reordered == in_order

    def test_war_card_quotes_the_correct_paragraph_text(
        self, monkeypatch, tmp_path
    ):
        """Beyond invariance: the war card's quote must come from the paragraph
        actually keyed to War=True (the war sentence), not whichever row shared
        its position. Proves the pairing is correct, not merely stable."""
        _install_parquets(
            monkeypatch, tmp_path, _paragraphs_frame(), _labels_frame()
        )
        out = _run_issue_cards()

        cards = out["Alpha President"]["cards"]
        war_cards = [c for c in cards if c["issue"] == "War & military"]
        assert len(war_cards) == 1
        # The quote is html-escaped; the sentence has no escapable chars.
        assert war_cards[0]["quote"] == WAR_SENTENCE
        # The calm paragraphs must never surface as the war quote.
        assert "gardens" not in war_cards[0]["quote"]
        assert "rivers" not in war_cards[0]["quote"]

    def test_duplicate_key_raises_merge_error(self, monkeypatch, tmp_path):
        """A non-unique (doc_name, para_idx) in the label table - the shape real
        corruption would take - must raise loudly via validate='one_to_one'
        rather than silently producing a fan-out join."""
        labels = _labels_frame()
        dup = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
        _install_parquets(monkeypatch, tmp_path, _paragraphs_frame(), dup)

        with pytest.raises(pd.errors.MergeError):
            _run_issue_cards()

    def test_missing_label_row_raises_via_length_guard(self, monkeypatch, tmp_path):
        """A paragraph absent from the label table would otherwise be silently
        dropped by how='inner' (see TestKeyedMergeContract). issue_cards wraps
        the merge with a post-merge length check, so a diverging key set - the
        one silent-failure mode an inner join could reintroduce - raises loudly
        instead, same as the old length-only assert did."""
        labels = _labels_frame()
        # Drop the war paragraph's label row (doc-a, para_idx=1).
        labels = labels[~((labels["doc_name"] == "doc-a") & (labels["para_idx"] == 1))]
        _install_parquets(monkeypatch, tmp_path, _paragraphs_frame(), labels)

        with pytest.raises(RuntimeError, match="key sets diverge"):
            _run_issue_cards()


# --------------------------------------------------------------------------- #
# Real-function test: issues_site._issue_quotes consumes the pre-merged frame
# --------------------------------------------------------------------------- #
class TestIssueQuotesReadsFromMergedFrame:
    def test_quote_and_citation_come_from_the_same_row(self):
        """_issue_quotes now takes a single merged frame; text and its
        president/year/doc must be read from the same row. A quote from one
        paragraph paired with another paragraph's citation would be exactly the
        corruption the key join prevents."""
        merged = pd.DataFrame(
            {
                "doc_name": ["doc-a", "doc-b", "doc-c"],
                "para_idx": [0, 0, 0],
                "year": [1900, 1950, 1990],
                "text": [WAR_SENTENCE, CALM_RIVERS, CALM_GARDENS],
                "War & military": [True, False, False],
            }
        )
        titles = pd.DataFrame(
            {
                "president": ["Alpha President", "Beta", "Gamma"],
                "title": ["Address: War", "Address: Rivers", "Address: Gardens"],
                "year": [1900, 1950, 1990],
            },
            index=pd.Index(["doc-a", "doc-b", "doc-c"], name="doc_name"),
        )

        quotes = _issue_quotes(merged, "War & military", WAR_ANCHORS, titles)

        assert len(quotes) == 1
        q = quotes[0]
        assert q["quote"] == WAR_SENTENCE
        # Citation must resolve through doc-a (the war row), not doc-b/doc-c.
        assert "Alpha President" in q["cite"]
        assert "1900" in q["cite"]
        assert q["url"].endswith("doc-a")


# --------------------------------------------------------------------------- #
# Contract tests: the exact merge call the fix relies on, on synthetic frames.
# These pin the pandas semantics profiles.py / issues_site.py depend on so a
# future refactor that weakens the join (drops the key, changes validate/how)
# is caught here directly.
# --------------------------------------------------------------------------- #
def _keyed_merge(paras: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """The identical call used in profiles.issue_cards and
    issues_site.write_issue_pages."""
    return paras.merge(
        labels, on=["doc_name", "para_idx"], how="inner", validate="one_to_one"
    ).reset_index(drop=True)


class TestKeyedMergeContract:
    def test_reorder_realigns_text_to_correct_label(self):
        """The label rows are in a different order than the paragraph rows; a
        positional join would mislabel the war paragraph. The keyed merge must
        pair each paragraph's text with the label that shares its key."""
        merged = _keyed_merge(_paragraphs_frame(), _labels_frame())

        by_text = merged.set_index("text")["War & military"]
        assert by_text[WAR_SENTENCE] == True  # noqa: E712  (want the label, not truthiness)
        assert by_text[CALM_GARDENS] == False
        assert by_text[CALM_RIVERS] == False

    def test_result_follows_left_frame_order(self):
        """Why the shuffle is self-healing: an inner merge preserves the left
        (paragraphs) key order regardless of the right (labels) order."""
        paras = _paragraphs_frame()
        shuffled_labels = _labels_frame().sample(frac=1, random_state=7)

        merged = _keyed_merge(paras, shuffled_labels)

        assert list(zip(merged["doc_name"], merged["para_idx"])) == list(
            zip(paras["doc_name"], paras["para_idx"])
        )

    def test_duplicate_key_raises_merge_error(self):
        """validate='one_to_one' turns a duplicated key into a loud
        MergeError - the guarantee this whole task delivers."""
        labels = _labels_frame()
        dup = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)

        with pytest.raises(pd.errors.MergeError):
            _keyed_merge(_paragraphs_frame(), dup)

    def test_unmatched_row_is_dropped_silently(self):
        """Pandas contract, not consumer behavior: the bare how='inner' merge
        drops a paragraph that has no label row (and vice versa) without
        raising. Both real consumers wrap this call with a post-merge length
        check (see test_missing_label_row_raises_via_length_guard) so this
        silent drop never reaches a caller - it is caught one layer up."""
        paras = _paragraphs_frame()
        labels = _labels_frame()
        labels = labels[~((labels["doc_name"] == "doc-b") & (labels["para_idx"] == 0))]

        merged = _keyed_merge(paras, labels)  # must not raise

        assert len(merged) == len(paras) - 1
        remaining_keys = set(zip(merged["doc_name"], merged["para_idx"]))
        assert ("doc-b", 0) not in remaining_keys


# --------------------------------------------------------------------------- #
# para_idx insert: mirrors the label-frame assembly in issues.build_issues.
# build_issues fits a CorEx model over the full corpus, so we exercise only the
# key-assembly logic (issues.py: para_labels.insert of doc_name/para_idx/...)
# on a synthetic label matrix rather than retraining the model.
# --------------------------------------------------------------------------- #
class TestParaIdxInsert:
    def _paras(self) -> pd.DataFrame:
        """Stand-in for the paragraphs+speeches frame build_issues builds:
        carries the (doc_name, para_idx) key plus president/year."""
        return pd.DataFrame(
            {
                "doc_name": ["doc-a", "doc-a", "doc-b"],
                "para_idx": [0, 1, 0],
                "president": ["Alpha", "Alpha", "Beta"],
                "year": [1900, 1900, 1990],
                "text": [WAR_SENTENCE, CALM_GARDENS, CALM_RIVERS],
            }
        )

    def _assemble_labels(self, paras: pd.DataFrame) -> pd.DataFrame:
        """Replicates issues.build_issues label-frame assembly: a boolean topic
        matrix with the key columns inserted in front."""
        all_names = ["War & military", "Discovered 1"]
        matrix = np.array([[True, False], [False, True], [True, True]])
        para_labels = pd.DataFrame(matrix, columns=all_names)
        para_labels.insert(0, "doc_name", paras["doc_name"].values)
        para_labels.insert(1, "para_idx", paras["para_idx"].values)
        para_labels.insert(2, "president", paras["president"].values)
        para_labels.insert(3, "year", paras["year"].values)
        return para_labels

    def test_para_idx_present_right_after_doc_name_with_unique_key(self):
        paras = self._paras()
        labels = self._assemble_labels(paras)

        assert list(labels.columns[:2]) == ["doc_name", "para_idx"]
        assert labels["para_idx"].tolist() == paras["para_idx"].tolist()
        # The key the consumers merge on must be unique.
        assert not labels.duplicated(["doc_name", "para_idx"]).any()

    def test_assembled_labels_merge_one_to_one_with_paragraphs(self):
        """Closes the loop: the frame build_issues writes is compatible with the
        one_to_one keyed merge its consumers perform, and pairs the right text
        with the right label."""
        paras = self._paras()
        labels = self._assemble_labels(paras)
        paragraphs_only = paras[["doc_name", "para_idx", "text"]]

        merged = paragraphs_only.merge(
            labels[["doc_name", "para_idx", "War & military"]],
            on=["doc_name", "para_idx"],
            how="inner",
            validate="one_to_one",
        )

        assert len(merged) == len(paras)
        war_row = merged.loc[merged["text"] == WAR_SENTENCE].iloc[0]
        assert war_row["War & military"] == True  # noqa: E712
