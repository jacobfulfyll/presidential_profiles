import json

import pandas as pd
import pytest

from presidential_profiles import grammar


def test_declared_eras_cover_story_endpoints():
    assert grammar._era(1789) == "Establishing the republic"
    assert grammar._era(1933) == "New Deal, world war & settlement"
    assert grammar._era(2026) == "Platform-era intensification"


def test_grammar_artifact_has_stable_coarse_tag_schema():
    table = grammar.build()
    assert table.doc_name.is_unique
    assert len(table) == 1057
    assert {"pos_noun", "pos_verb", "pos_pron", "n_tagged_tokens"} <= set(table)
    assert (table.n_tagged_tokens > 0).all()


def test_grammar_manifest_pins_exact_runtime_and_source_identity():
    _, manifest = grammar.validate_cached_artifacts()
    assert manifest["schema_version"] == "grammar-pos-v2"
    assert manifest["model"]["library"] == "spacy"
    assert manifest["model"]["library_version"]
    assert manifest["model"]["package"] == "en_core_web_sm"
    assert manifest["model"]["package_version"]
    assert manifest["source_identity"]["corpus_fingerprint"]["n_speeches"] == 1057
    assert manifest["source_identity"]["corpus_fingerprint"]["n_paragraphs"] == 36229
    assert set(manifest["artifacts"]) == {"speech_pos.parquet", "era_pos.csv"}
    assert manifest["generation"]["paid_api_calls"] is False


def _tiny_cached_bundle(tmp_path, monkeypatch):
    speech_path = tmp_path / "speech_pos.parquet"
    era_path = tmp_path / "era_pos.csv"
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr(grammar, "SPEECH_POS_PATH", speech_path)
    monkeypatch.setattr(grammar, "ERA_POS_PATH", era_path)
    monkeypatch.setattr(grammar, "MANIFEST_PATH", manifest_path)

    row = {
        "doc_name": "a", "president": "President A", "year": 1900,
        "title": "Example", "n_tagged_tokens": 12, "era_key": "example",
        "era": "Example era", "era_order": 0,
        **{f"pos_{label.lower()}": 1 for label in grammar.POS_LABELS},
    }
    table = pd.DataFrame([row])
    table.to_parquet(speech_path, index=False)
    pd.DataFrame([
        {"era_key": "example", "era": "Example era", "era_order": 0,
         "part_of_speech": label,
         "share_percent": 100 / len(grammar.POS_LABELS), "n_tagged_tokens": 12}
        for label in grammar.POS_LABELS
    ]).to_csv(era_path, index=False)
    source = {"corpus_fingerprint": {"n_speeches": 1, "n_paragraphs": 1}}
    model = {"package": "example", "package_version": "1"}
    monkeypatch.setattr(grammar, "_source_identity", lambda speeches=None: source)
    monkeypatch.setattr(grammar, "_runtime_identity", lambda: model)
    manifest = {
        "schema_version": grammar.SCHEMA_VERSION,
        "labels": list(grammar.POS_LABELS),
        "era_contract": grammar._era_contract(),
        "model": model,
        "source_identity": source,
        "n_speeches": 1,
        "artifacts": {
            "speech_pos.parquet": grammar._sha256(speech_path),
            "era_pos.csv": grammar._sha256(era_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest))
    return table, manifest_path, speech_path


def test_grammar_cache_refuses_stale_source_identity(tmp_path, monkeypatch):
    _, manifest_path, _ = _tiny_cached_bundle(tmp_path, monkeypatch)
    manifest = json.loads(manifest_path.read_text())
    manifest["source_identity"] = {"corpus_fingerprint": {"n_speeches": 2}}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(grammar.GrammarArtifactError, match="source corpus identity drift"):
        grammar.validate_cached_artifacts()


def test_grammar_cache_refuses_content_hash_drift(tmp_path, monkeypatch):
    _, _, speech_path = _tiny_cached_bundle(tmp_path, monkeypatch)
    table = pd.read_parquet(speech_path)
    table.loc[0, "pos_noun"] = 999
    table.to_parquet(speech_path, index=False)
    with pytest.raises(grammar.GrammarArtifactError, match="content hash drift"):
        grammar.validate_cached_artifacts()


def test_grammar_cache_refuses_partial_bundle(tmp_path, monkeypatch):
    _tiny_cached_bundle(tmp_path, monkeypatch)
    grammar.ERA_POS_PATH.unlink()
    with pytest.raises(grammar.GrammarArtifactError, match="bundle is incomplete"):
        grammar.build()
