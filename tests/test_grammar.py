from presidential_profiles import grammar


def test_declared_eras_cover_story_endpoints():
    assert grammar._era(1789) == "The founding"
    assert grammar._era(1933) == "War & New Deal"
    assert grammar._era(2026) == "The present era"


def test_grammar_artifact_has_stable_coarse_tag_schema():
    table = grammar.build()
    assert table.doc_name.is_unique
    assert len(table) == 1057
    assert {"pos_noun", "pos_verb", "pos_pron", "n_tagged_tokens"} <= set(table)
    assert (table.n_tagged_tokens > 0).all()
