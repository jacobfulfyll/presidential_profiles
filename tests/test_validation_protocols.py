"""Acceptance tests for execution-blocked Data Trust validation protocols."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest

from presidential_profiles import era_profiles, validation_protocols as protocols
from presidential_profiles.prompts import annotation_v1 as prompt_v1


DATA = Path(__file__).parents[1] / "data"
REPO = DATA.parent


@pytest.fixture(scope="module")
def inputs() -> protocols.ProtocolInputs:
    return protocols.load_inputs()


@pytest.fixture(scope="module")
def bundle(inputs) -> protocols.ValidationProtocolBundle:
    return protocols.build_protocol_bundle(inputs=inputs)


def test_paragraph_and_factual_samples_have_exact_governed_quotas(bundle):
    paragraph = bundle.paragraph_sample
    speech = bundle.speech_sample
    era_keys = [spec.key for spec in era_profiles.ERA_PROFILE_SPECS]
    assert len(paragraph) == 360
    assert len(speech) == 90
    assert list(paragraph.sort_values("story_era_order").story_era_key.drop_duplicates()) == era_keys
    assert list(speech.sort_values("story_era_order").story_era_key.drop_duplicates()) == era_keys
    assert paragraph.groupby("story_era_key").size().to_dict() == {
        key: 40 for key in era_keys
    }
    assert speech.groupby("story_era_key").size().to_dict() == {
        key: 10 for key in era_keys
    }
    counts = paragraph.groupby(["story_era_key", "selection_arm"]).size()
    for key in era_keys:
        assert counts[key].to_dict() == {
            "genre-balanced-base": 20,
            "high-disagreement": 10,
            "rare-positive": 10,
        }


def test_selection_keys_are_unique_eligible_and_arms_are_mutually_exclusive(bundle, inputs):
    selected = bundle.paragraph_sample
    eligible = inputs.paragraph_view.loc[inputs.paragraph_view.analysis_eligible]
    assert not selected.duplicated(protocols.PARAGRAPH_KEY).any()
    assert protocols._key_set(selected) <= protocols._key_set(eligible)
    assert set(selected.selection_arm) == set(protocols.ARM_ORDER)
    assert not bundle.speech_sample.doc_name.duplicated().any()
    assert set(bundle.speech_sample.doc_name) <= set(inputs.source_speeches.doc_name)
    assert len(inputs.source_speeches) == protocols.EXPECTED_FACTUAL_SOURCE_SPEECHES
    assert bundle.manifest["populations"]["factual_candidate_source_documents"] == 1_057
    first_pass_cells = bundle.speech_cells.loc[
        bundle.speech_cells.allocation_cell.ne("fallback")
    ]
    assert int(first_pass_cells.candidate_documents.sum()) == 1_057


def test_high_disagreement_arm_is_the_preregistered_top_ten_per_era(inputs, bundle):
    _, paired = protocols._prepare_population(inputs)
    observed = bundle.paragraph_sample.loc[
        bundle.paragraph_sample.selection_arm.eq("high-disagreement")
    ]
    for spec in era_profiles.ERA_PROFILE_SPECS:
        candidates = paired.loc[paired.story_era_key.eq(spec.key)].copy()
        candidates["tie"] = [
            protocols._rank_hash(
                "high-disagreement", spec.key, row.doc_name, int(row.para_idx)
            )
            for row in candidates.itertuples(index=False)
        ]
        expected = candidates.sort_values(
            ["selection_score", "tie", "doc_name", "para_idx"],
            ascending=[False, True, True, True],
            kind="stable",
        ).head(10)
        assert protocols._key_set(
            observed.loc[observed.story_era_key.eq(spec.key)]
        ) == protocols._key_set(expected)


def test_genre_first_pass_and_fallback_are_explicit(bundle):
    base = bundle.paragraph_sample.loc[
        bundle.paragraph_sample.selection_arm.eq("genre-balanced-base")
    ]
    first = base.loc[base.selection_stage.eq("unique-document-first-pass")]
    assert not first.duplicated(["story_era_key", "doc_name"]).any()
    cells = bundle.paragraph_cells
    fallback = cells.loc[cells.allocation_cell.eq("fallback")].set_index("story_era_key")
    assert fallback.loc["founding", "selected_total"] == 5
    assert fallback.loc["expansion", "selected_total"] == 8
    assert fallback.loc["civil-war-reconstruction", "selected_total"] == 4
    assert fallback.loc["gilded-age", "selected_total"] == 3
    assert int(fallback.loc[
        ["progressives-depression", "war-new-deal", "cold-war", "post-cold-war", "present"],
        "selected_total",
    ].sum()) == 0


def test_factual_allocation_uses_sentinel_and_document_fallback(bundle):
    sample = bundle.speech_sample
    for row in sample.itertuples(index=False):
        assert row.tie_break_sha256 == protocols._rank_hash(
            "factual-speech", row.story_era_key, row.doc_name, -1
        )
    fallback = bundle.speech_cells.loc[
        bundle.speech_cells.allocation_cell.eq("fallback")
    ].set_index("story_era_key")
    assert fallback.loc["founding", "selected_total"] == 2
    assert fallback.loc["expansion", "selected_total"] == 3
    assert int(fallback.drop(index=["founding", "expansion"]).selected_total.sum()) == 0


def test_selected_sample_hashes_are_exact_regressions(bundle):
    selection = bundle.manifest["selection"]
    assert selection["paragraph_sample_sha256"] == (
        "sha256:8a402b8b29efd3217e58047586e96f314cb0dbe20c29e3da49055ff8263f8440"
    )
    assert selection["paragraph_key_set_sha256"] == (
        "sha256:e8c86719bbbb110c54fa076c13295a24d3db85728c56c692bccf1fd4d2720aad"
    )
    assert selection["factual_speech_key_set_sha256"] == (
        "sha256:8cb151a5b96ce2e533814b46ab94c98a9283abc95f82a307d56d66e5a259e8cb"
    )


def test_factual_sample_refuses_an_incomplete_source_annotation_population(inputs):
    incomplete = replace(
        inputs, speech_annotations=inputs.speech_annotations.iloc[:-1].copy()
    )
    with pytest.raises(
        protocols.ValidationProtocolError,
        match="must contain all 1,057 source speeches",
    ):
        protocols._select_speeches(pd.DataFrame(), incomplete)


def test_weights_exist_only_for_probability_sampled_first_pass_rows(bundle):
    for frame in (bundle.paragraph_sample, bundle.speech_sample.assign(selection_score=0.0)):
        protocols._validate_probabilities(frame, "test frame")
        weighted = frame.weight_status.eq("design_weight_available")
        prohibited = frame.weight_status.isin({
            "not_applicable_purposive_enrichment",
            "not_available_sequential_fallback",
        })
        assert frame.loc[prohibited, ["selection_probability", "sampling_weight"]].isna().all().all()
        assert frame.loc[weighted, "selection_probability"].between(
            0, 1, inclusive="both"
        ).all()
        assert all(
            math.isclose(weight, 1 / probability, rel_tol=0, abs_tol=1e-10)
            for weight, probability in zip(
                frame.loc[weighted, "sampling_weight"],
                frame.loc[weighted, "selection_probability"],
                strict=True,
            )
        )
    paragraph_status = bundle.paragraph_sample.weight_status.value_counts().to_dict()
    assert paragraph_status == {
        "not_applicable_purposive_enrichment": 180,
        "design_weight_available": 160,
        "not_available_sequential_fallback": 20,
    }
    speech_status = bundle.speech_sample.weight_status.value_counts().to_dict()
    assert speech_status == {
        "design_weight_available": 85,
        "not_available_sequential_fallback": 5,
    }


def test_blinded_coder_files_expose_only_prespecified_evidence(bundle):
    assert {key for row in bundle.paragraph_tasks for key in row} == (
        protocols._BLINDED_PARAGRAPH_FIELDS
    )
    assert {key for row in bundle.speech_tasks for key in row} == (
        protocols._BLINDED_SPEECH_FIELDS
    )
    assert not any(
        protocols._PROHIBITED_BLINDED_FIELDS & set(row)
        for row in [*bundle.paragraph_tasks, *bundle.speech_tasks]
    )
    assert len({row["item_id"] for row in bundle.paragraph_tasks}) == 360
    assert len({row["item_id"] for row in bundle.speech_tasks}) == 90
    assert all(1 <= len(row["opening_paragraphs"]) <= 5 for row in bundle.speech_tasks)


def test_current_prompt_is_production_exact_and_hidden_diff_is_allowlisted(bundle):
    row = bundle.paragraph_tasks[0]
    master = bundle.paragraph_sample.set_index("study_item_id").loc[row["item_id"]]
    decade = int(row["decade"][:-1])
    para_idx = int(master.para_idx)
    frame = pd.DataFrame([{"para_idx": para_idx, "text": row["paragraph_text"]}])
    production = prompt_v1.JUDGMENT_INSTRUCTION + "\n\n" + prompt_v1.judgment_context(
        SimpleNamespace(decade=decade), frame
    )
    assert protocols.render_current_user_prompt(
        paragraph_text=row["paragraph_text"], decade=decade, para_idx=para_idx
    ) == production
    assert bundle.hidden_prompt == protocols._hide_decade_prompt(bundle.current_prompt)
    assert bundle.current_prompt["output_schema"] == bundle.hidden_prompt["output_schema"]
    assert "<DECADE>" in bundle.current_prompt["context_template"]
    assert "<DECADE>" not in bundle.hidden_prompt["context_template"]


def test_invariance_request_plan_reuses_same_keys_without_assigning_model(bundle):
    plan = bundle.request_plan
    assert len(plan) == 360
    assert plan.presentation_order.tolist() == list(range(1, 361))
    assert set(plan.study_item_id) == set(bundle.paragraph_sample.study_item_id)
    assert set(plan.runtime_model) == {"UNSET"}
    assert set(plan.items_per_request) == {1}
    assert set(plan.planned_evaluations) == {2}
    study = bundle.manifest["measurement_invariance_study"]
    assert study["runtime_model"] is None
    assert study["decoding_configuration"] is None
    assert study["retry_policy"] is None
    assert study["hard_max_cost_usd"] is None
    assert study["approver"] is None
    assert study["cost_estimation"]["estimated_cost_usd"] is None
    assert study["cost_estimation"]["status"].startswith("blocked_")


def test_cost_formula_is_machine_checkable_but_never_fabricates_missing_inputs():
    blocked = protocols.estimate_cost(
        input_tokens=None,
        output_tokens=None,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
        input_usd_per_million=None,
        output_usd_per_million=None,
        cache_write_multiplier=None,
        cache_read_multiplier=None,
        batch_discount_multiplier=None,
    )
    assert blocked["estimated_cost_usd"] is None
    ready = protocols.estimate_cost(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_creation_input_tokens=10_000,
        cache_read_input_tokens=50_000,
        input_usd_per_million=3.0,
        output_usd_per_million=15.0,
        cache_write_multiplier=1.25,
        cache_read_multiplier=0.1,
        batch_discount_multiplier=0.5,
    )
    assert ready["estimated_cost_usd"] == pytest.approx(2.27625)
    assert ready["status"] == "estimate_ready_requires_separate_approval"
    with pytest.raises(protocols.ValidationProtocolError, match="finite and non-negative"):
        protocols.estimate_cost(
            input_tokens=-1,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            input_usd_per_million=3,
            output_usd_per_million=15,
            cache_write_multiplier=1.25,
            cache_read_multiplier=0.1,
            batch_discount_multiplier=0.5,
        )


def test_duplicate_nonfinite_and_prompt_drift_mutations_fail_closed(bundle):
    duplicate = pd.concat(
        [bundle.paragraph_sample, bundle.paragraph_sample.iloc[[0]]], ignore_index=True
    )
    with pytest.raises(protocols.ValidationProtocolError, match="duplicate semantic keys"):
        protocols._require_unique(duplicate, protocols.PARAGRAPH_KEY, "mutant")
    nonfinite = bundle.paragraph_sample.copy()
    weighted_index = nonfinite.index[
        nonfinite.weight_status.eq("design_weight_available")
    ][0]
    nonfinite.loc[weighted_index, "selection_probability"] = float("inf")
    with pytest.raises(protocols.ValidationProtocolError, match="invalid design"):
        protocols._validate_probabilities(nonfinite, "mutant")
    hidden = dict(bundle.hidden_prompt)
    hidden["user_instruction"] += " extra change"
    with pytest.raises(protocols.ValidationProtocolError, match="outside the allowlisted"):
        protocols._validate_prompt_pair(bundle.current_prompt, hidden)


def test_source_hash_validation_refuses_stale_input(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"governed")
    receipt = protocols._source_receipt(source, REPO)
    manifest = {"source_inventory": [receipt]}
    protocols.validate_source_hashes(manifest)
    source.write_bytes(b"drifted")
    with pytest.raises(protocols.ValidationProtocolError, match="source hash drift"):
        protocols.validate_source_hashes(manifest)


def test_source_inventory_refuses_a_stale_protocol_builder(tmp_path):
    builder = (
        tmp_path / "src" / "presidential_profiles" / "validation_protocols.py"
    )
    builder.parent.mkdir(parents=True)
    builder.write_bytes(Path(protocols.__file__).read_bytes())
    receipt = protocols._source_receipt(builder, tmp_path)
    assert receipt["path"] == "src/presidential_profiles/validation_protocols.py"
    manifest = {"source_inventory": [receipt]}
    protocols.validate_source_hashes(manifest, repo_root=tmp_path)
    builder.write_bytes(builder.read_bytes() + b"\n# stale mutation\n")
    with pytest.raises(protocols.ValidationProtocolError, match="source hash drift"):
        protocols.validate_source_hashes(manifest, repo_root=tmp_path)


def test_public_inventory_is_disjoint_and_cross_artifact_blinding_fails_closed(bundle):
    assert set(protocols.PUBLIC_FILES).isdisjoint(protocols.RESTRICTED_FILES)
    assert set(protocols.GOVERNED_FILES) == (
        set(protocols.PUBLIC_FILES) | set(protocols.RESTRICTED_FILES)
    )
    protocols._validate_cross_artifact_blinding(
        bundle.files, bundle.paragraph_sample, bundle.speech_sample
    )
    leaked = dict(bundle.files)
    selected_doc = str(bundle.paragraph_sample.iloc[0].doc_name)
    leaked[protocols.PARAGRAPH_CELLS_NAME] += selected_doc.encode("utf-8")
    with pytest.raises(protocols.ValidationProtocolError, match="leaks a selected identifier"):
        protocols._validate_cross_artifact_blinding(
            leaked, bundle.paragraph_sample, bundle.speech_sample
        )


def test_restricted_protocol_tree_is_git_ignored_untracked_and_not_public():
    restricted_relative = Path("data/validation_protocols/v1/restricted")
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(restricted_relative)],
        cwd=REPO,
        check=False,
    )
    assert ignored.returncode == 0
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=REPO, text=True
    ).splitlines()
    assert not any(
        path == str(restricted_relative) or path.startswith(f"{restricted_relative}/")
        for path in tracked
    )
    public_root = REPO / "docs" / "data" / "validation-protocols-v1"
    assert not any((public_root / name).exists() for name in protocols.RESTRICTED_FILES)


def test_build_is_byte_identical_and_publication_validates(bundle, inputs, tmp_path):
    repeat = protocols.build_protocol_bundle(inputs=inputs)
    assert bundle.files == repeat.files
    output = tmp_path / "validation_protocols" / "v1"
    paths = protocols.write_publication(bundle, output)
    assert {str(path.relative_to(output)) for path in paths} == set(
        protocols.GOVERNED_FILES
    )
    manifest = protocols.validate_publication(output)
    assert manifest["metadata_sha256"] == protocols._metadata_hash(manifest)
    assert manifest["execution_authorized"] is False
    for filename, receipt in manifest["artifacts"].items():
        payload = (output / filename).read_bytes()
        assert receipt["sha256"] == protocols._sha256_bytes(payload)

    public_output = tmp_path / "public" / "validation-protocols-v1"
    public_paths = protocols.write_public_projection(
        public_output, governed_dir=output
    )
    assert {str(path.relative_to(public_output)) for path in public_paths} == set(
        protocols.PUBLIC_FILES
    )
    assert {
        str(path.relative_to(public_output))
        for path in public_output.rglob("*")
        if path.is_file()
    } == set(protocols.PUBLIC_FILES)
    protocols.validate_public_projection(public_output)
    assert not any((public_output / name).exists() for name in protocols.RESTRICTED_FILES)


def test_directory_publication_restores_previous_bundle_on_failure(tmp_path):
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "sentinel.txt").write_text("previous", encoding="utf-8")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "sentinel.txt").write_text("candidate", encoding="utf-8")

    def fail(_path):
        raise RuntimeError("post-swap validation failed")

    with pytest.raises(RuntimeError, match="post-swap"):
        protocols._publish_candidate(candidate, destination, post_validate=fail)
    assert (destination / "sentinel.txt").read_text(encoding="utf-8") == "previous"


def test_module_has_no_network_or_provider_imports():
    tree = ast.parse(Path(protocols.__file__).read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert not imported_roots & {
        "anthropic", "openai", "requests", "httpx", "urllib", "socket", "aiohttp"
    }


def test_manifest_public_language_is_planned_not_results(bundle):
    manifest = bundle.manifest
    assert manifest["generation_status"] == "deterministic_local_no_api_calls"
    assert manifest["publication_status"] == "planned_validation_only_no_results"
    assert manifest["human_validity_study"]["status"] == "planned_not_run"
    assert manifest["measurement_invariance_study"]["status"] == (
        "dry_run_only_blocked_not_run"
    )
    assert manifest["human_validity_study"]["coders"]["named_coders"] == []
    assert manifest["human_validity_study"]["reporting"][
        "enrichment_population_weighting_prohibited"
    ] is True
    assert "never use high-disagreement" in manifest["selection"][
        "population_inference_gate"
    ]
    assert all(
        status.startswith("blocked_")
        for gate, status in manifest["human_validity_study"]["execution_gates"].items()
        if gate != "key_and_blinding_checks"
    )
