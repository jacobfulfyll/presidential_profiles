import numpy as np
import pandas as pd

from presidential_profiles import coverage_pressure as cp
from presidential_profiles import inference


def test_effective_topics_examples():
    assert cp.effective_topics([1, 1, 1, 1]) == 4
    assert cp.effective_topics([8, 0]) == 1
    assert cp.effective_topics([0, 0]) == 0
    assert 1 < cp.effective_topics([9, 1]) < 2


def test_multilabel_exposure_conserves_words():
    split = cp.split_topic_words(101, ["a", "b", "a"])
    assert sum(split.values()) == 101
    assert split == {"a": 50.5, "b": 50.5}


def test_episode_construction_tracks_paragraphs_and_words():
    rows = pd.DataFrame({
        "para_idx": [0, 1, 3, 0], "topic": ["a", "a", "a", "b"],
        "exposure_words": [10, 20, 5, 9],
    })
    episodes = cp.topic_episodes(rows)
    assert episodes[episodes.topic == "a"].run_words.tolist() == [30, 5]
    assert episodes[episodes.topic == "a"].run_paragraphs.tolist() == [2, 1]


def test_holm_and_joint_decision():
    assert inference.holm_adjust([.01, .04]).tolist() == [.02, .04]
    assert cp.joint_decision(.02, .03, 1.0, -2.0) == "supported"
    assert cp.joint_decision(.02, .03, -1.0, -2.0) == "mixed"
    assert cp.joint_decision(.08, .03, 1.0, -2.0) == "inconclusive"


def test_p_value_floor():
    p, floor = inference.p_value_from_draws(np.ones(19), null=0, alternative="greater")
    assert p == floor == .05

