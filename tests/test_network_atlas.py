import pandas as pd

from presidential_profiles import networks


def test_topic_edge_math_uses_paragraph_support():
    assignments = pd.DataFrame([
        ("d1", 0, "a"), ("d1", 0, "b"),
        ("d1", 1, "a"), ("d2", 0, "a"), ("d2", 0, "b"),
    ], columns=["doc_name", "para_idx", "topic"])
    speeches = pd.DataFrame({"doc_name": ["d1", "d2"]})
    edge = networks.topic_edges(assignments, speeches).iloc[0]
    assert edge.shared_paragraphs == 2
    assert edge.shared_speeches == 2
    assert edge.jaccard == 2 / 3
    assert edge.lift == 1


def test_layout_is_stable():
    edges = pd.DataFrame({"source": ["a", "b"], "target": ["b", "c"],
                          "shared_speeches": [2, 3]})
    assert networks.stable_layout(edges) == networks.stable_layout(edges)

