"""Deterministic contracts for the public era-boundary analysis."""

from __future__ import annotations

import itertools
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import (
    era_boundaries,
    era_boundaries_site,
    era_profiles,
    eras,
)
from presidential_profiles.expansion_site import NAV


def _fingerprints() -> pd.DataFrame:
    rows = []
    for i, unit in enumerate(range(1784, 1880, 8)):
        for measure, group, value in [
            ("topic__Economy", "topic", float(i < 6)),
            ("marker_boosters", "marker", float(i >= 6)),
        ]:
            rows.append(
                {
                    "grain": "bin8",
                    "unit": str(float(unit)),
                    "measure": measure,
                    "axis_group": group,
                    "value_raw": value,
                    "n_paragraphs": 100,
                    "n_speeches": 10,
                }
            )
    return pd.DataFrame(rows)


def _markers(docs: list[str]) -> pd.DataFrame:
    rows = []
    for doc in docs:
        year = int(doc[1:])
        row = {"doc_name": doc, "n_words": 1000.0}
        for column in eras.MARKER_COLUMNS:
            row[column] = float(column == "boosters" and year >= 2008)
        rows.append(row)
    return pd.DataFrame(rows)


def _stats(docs: list[str]) -> pd.DataFrame:
    rows = []
    for doc in docs:
        row = {
            "doc_name": doc,
            "n_tokens": 1000.0,
            "fk_grade": 10.0,
            "words_per_sentence": 20.0,
            "i_count": 5.0,
            "we_count": 5.0,
        }
        row.update({column: 0.0 for column in eras.MODAL_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def _master() -> pd.DataFrame:
    rows = []
    for year in range(2000, 2016):
        rows.append(
            {
                "doc_name": f"d{year}",
                "para_idx": 0,
                "year": year,
                "era": "window",
                "president": "P",
                "bin8": (year // 8) * 8,
                "center_year": float(year),
                "speech_type": "annual_message",
                "topic__Economy": year >= 2008,
                "party_attack": False,
                "enemy_naming": False,
                "zero_sum": False,
                "proposal_values": "neither",
            }
        )
    return pd.DataFrame(rows)


def _two_axis_reference() -> pd.DataFrame:
    rows = []
    for unit, topic, marker in [("2000.0", 0.0, 0.0), ("2008.0", 1.0, 1.0)]:
        for measure, group, value in [
            ("topic__Economy", "topic", topic),
            ("marker_boosters", "marker", marker),
        ]:
            rows.append(
                {
                    "grain": "bin8",
                    "unit": unit,
                    "measure": measure,
                    "axis_group": group,
                    "value_raw": value,
                    "n_paragraphs": 8,
                    "n_speeches": 8,
                }
            )
    return pd.DataFrame(rows)


def test_cluster_stability_uses_only_observed_bin_starts():
    out = era_boundaries.build_cluster_stability(_fingerprints())
    observed = set(range(1784, 1880, 8))
    assert set(out["boundary_year"]).issubset(observed)
    assert set(out["condition"]) == {
        "full",
        "drop_topic",
        "drop_combat",
        "drop_register",
        "drop_marker",
        "drop_stat",
        "drop_genre",
        "drop_opponents",
    }
    assert out[(out["condition"] == "full") & (out["k"] == 9)].shape[0] == 8


def test_sliding_score_tests_the_only_possible_calendar_boundary():
    master = _master()
    docs = master["doc_name"].tolist()
    out = era_boundaries.sliding_transition_scores(
        master=master,
        fingerprints=_two_axis_reference(),
        markers=_markers(docs),
        stats=_stats(docs),
        taxonomy={"level2": [{"name": "Topic", "level1": "Economy"}]},
        window_years=8,
    )
    assert out["boundary_year"].tolist() == [2008]
    assert out.loc[0, "left_start"] == 2000
    assert out.loc[0, "right_end"] == 2015
    # Each raw difference is 1 and each two-bin population SD is .5.
    assert out.loc[0, "score"] == pytest.approx(np.sqrt(2 * 2.0**2))
    assert out.loc[0, "rank"] == 1


def test_consensus_combines_window_specific_rank_percentiles():
    sliding = pd.DataFrame(
        [
            {"boundary_year": year, "window_years": window, "score": score, "rank": rank}
            for window, values in (
                (4, [(1800, 9.0, 1), (1801, 5.0, 2), (1802, 1.0, 3)]),
                (8, [(1800, 1.0, 3), (1801, 5.0, 2), (1802, 9.0, 1)]),
            )
            for year, score, rank in values
        ]
    )
    out = era_boundaries.consensus_transition_scores(sliding)
    assert out["consensus"].tolist() == pytest.approx([0.5, 0.5, 0.5])
    assert out["consensus_rank"].tolist() == [1, 1, 1]


def test_persistent_regime_artifact_records_reviewed_near_optimum():
    analysis = era_boundaries.load()
    full = analysis["segmentation"]
    full = full[full["condition"] == "full"]
    assert full["segment_start"].tolist() == [
        1789, 1805, 1849, 1869, 1913, 1937, 1953, 1981, 2017
    ]
    assert analysis["meta"]["reviewed_grid_starts"] == [
        1789, 1809, 1849, 1869, 1913, 1933, 1953, 1981, 2017
    ]
    assert analysis["meta"]["reviewed_excess_objective_pct"] == pytest.approx(
        0.5121370210498188
    )


def test_public_page_explains_grid_artifact_and_presidential_calls():
    page = era_boundaries_site.render_era_boundaries(era_boundaries.load())
    assert "How we chose the eras" in page
    assert "Why did 1872 appear before?" in page
    assert "the four-year primary partition starts at 1869" in page
    assert "1809 · Madison takes office" in page
    assert "Accession-year ownership" in page
    assert "Current Story scheme · reviewed and live" in page
    assert "Recommended scheme · live story ranges not yet migrated" not in page
    assert "Mexican-American War ends in 1848" in page
    assert "Territorial acquisition ends in 1848" not in page
    assert "accounts for only about 5%" not in page
    assert "right-censored" in page
    assert "data/era-boundary-sliding-scores.csv" in page
    assert "data/era-boundary-segmentation.csv" in page
    assert "data/era-boundary-sensitivity.csv" in page
    assert "data/era-boundary-cluster-stability.csv" in page
    for start in (1809, 1850, 1869, 1913, 1933, 1953, 1981, 2017):
        assert f'id="boundary-{start}"' in page
    for artifact_range in (
        "1805–1810", "1849–1853", "1868–1871", "1912–1915",
        "1929–1937", "1951–1954", "1979–1982", "2013–2018",
    ):
        assert f"Segmentation sensitivity range: {artifact_range}" in page


def test_public_analytical_counts_are_rendered_from_supplied_artifact_meta():
    analysis = era_boundaries.load()
    mutated = {
        **analysis,
        "meta": {
            **analysis["meta"],
            "n_axes": 64,
            "cluster_count_range": [3, 11],
        },
    }
    page = era_boundaries_site.render_era_boundaries(mutated)
    assert "Governed 64-axis fingerprints" in page
    assert "Download k=3–11 cluster stability" in page


def test_public_era_choices_are_derived_from_story_specs():
    choices = era_boundaries_site._choices()
    assert [row["key"] for row in choices] == [
        spec.key for spec in era_profiles.ERA_PROFILE_SPECS
    ]
    assert [
        (row["start"], row["end"], row["title"], row["section_key"])
        for row in choices
    ] == [
        (spec.start_year, spec.end_year, spec.title, spec.section_key)
        for spec in era_profiles.ERA_PROFILE_SPECS
    ]
    page = era_boundaries_site.render_era_boundaries(era_boundaries.load())
    for spec in era_profiles.ERA_PROFILE_SPECS:
        assert f'index.html#{spec.section_key}' in page
        assert html.escape(spec.title) in page


def test_every_public_figure_has_semantic_evidence_contract():
    page = era_boundaries_site.render_era_boundaries(era_boundaries.load())
    figures = re.findall(r"<figure[^>]*data-substantive-chart[^>]*>", page)
    assert len(figures) == 2
    assert {
        re.search(r'data-metric="([^"]+)"', figure).group(1)
        for figure in figures
    } == {
        "era_annual_abruptness",
        "era_segmentation_sensitivity",
    }
    assert all('data-evidence="' in figure for figure in figures)
    assert all('aria-labelledby="' in figure for figure in figures)
    assert "Accessible table: all candidate-year scores" in page
    assert page.count("<tr><th scope=\"row\">") >= len(
        era_boundaries.load()["consensus_scores"]
    )
    assert "Accessible table: every condition" in page
    assert re.search(r'id="sliding-chart"[^>]*\shidden>', page)
    assert re.search(r'id="sensitivity-chart"[^>]*\shidden>', page)
    assert 'el.removeAttribute("hidden")' in page
    assert "pending.catch(()=>{el.hidden=true;})" in page
    assert "catch(error){el.hidden=true;}" in page
    assert "top:var(--global-nav-height,0)" in page
    assert 'matchMedia("(max-width: 800px)")' in page
    assert "height:narrow?680:760" in page
    assert "size:narrow?11:12" in page
    assert page.count('class="table-wrap" role="region"') >= 3
    assert page.count('tabindex="0"') >= 3
    assert "Scroll table horizontally →" in page
    assert "scroll-margin-top:190px" in page
    assert 'href="metrics.html#era_annual_abruptness"' in page
    assert 'href="metrics.html#era_segmentation_sensitivity"' in page
    for label in (
        "Claim", "Population", "Method", "Uncertainty", "Artifact", "Status",
        "Limitation", "Downstream use",
    ):
        assert f"<dt>{label}</dt>" in page


def test_page_is_a_primary_navigation_destination():
    data_children = dict(NAV)["Data"]
    assert ("Era Choices", "era-boundaries.html") in data_children


def test_optimal_partition_matches_brute_force_oracle():
    raw = np.array([0.0, 0.2, 0.1, 4.0, 4.1, 3.9, 9.0, 9.2, 8.9])
    costs = np.full((len(raw), len(raw)), np.inf)
    for left in range(len(raw)):
        for right in range(left, len(raw)):
            values = raw[left:right + 1]
            costs[left, right] = float(((values - values.mean()) ** 2).sum())
    starts, objective = era_boundaries._optimal_partition(costs, k=3, min_units=2)

    candidates = []
    for first, second in itertools.combinations(range(2, len(raw) - 1), 2):
        if second - first < 2 or len(raw) - second < 2:
            continue
        value = costs[0, first - 1] + costs[first, second - 1] + costs[second, len(raw) - 1]
        candidates.append(([0, first, second], value))
    expected_starts, expected_objective = min(candidates, key=lambda item: item[1])
    assert starts == expected_starts == [0, 3, 6]
    assert objective == pytest.approx(expected_objective)


@pytest.mark.parametrize(
    ("k", "min_units"),
    [(1, 2), (3, 0), (4, 3)],
)
def test_optimal_partition_rejects_invalid_or_infeasible_requests(k, min_units):
    with pytest.raises(ValueError, match="infeasible"):
        era_boundaries._optimal_partition(np.zeros((10, 10)), k=k, min_units=min_units)


def test_cycle_grid_offset_and_terminal_support_are_explicit():
    master = _master()
    docs = master["doc_name"].tolist()
    out = era_boundaries.build_cycle_fingerprints(
        master=master,
        fingerprints=_two_axis_reference(),
        markers=_markers(docs),
        stats=_stats(docs),
        taxonomy={"level2": [{"name": "Topic", "level1": "Economy"}]},
        grid_offset=1,
        anchor_year=2000,
    )
    assert out["cycle_start"].tolist() == [2001, 2005, 2009, 2013]
    assert out["cycle_end"].tolist() == [2004, 2008, 2012, 2015]
    assert out["grid_offset"].eq(1).all()
    assert out["is_complete_cycle"].tolist() == [True, True, True, False]
    with pytest.raises(ValueError, match="grid_offset"):
        era_boundaries.build_cycle_fingerprints(
            master=master,
            fingerprints=_two_axis_reference(),
            markers=_markers(docs),
            stats=_stats(docs),
            taxonomy={"level2": [{"name": "Topic", "level1": "Economy"}]},
            grid_offset=4,
        )


def test_segmentation_costs_reject_bad_axes_and_apply_family_weighting():
    cycle = pd.DataFrame({
        "topic__one": [0.0, 0.0, 1.0, 1.0],
        "marker_a": [0.0, 1.0, 0.0, 1.0],
        "marker_b": [0.0, 1.0, 0.0, 1.0],
        "marker_c": [0.0, 1.0, 0.0, 1.0],
    })
    axes = list(cycle.columns)
    per_axis = era_boundaries._segmentation_costs(cycle, axes, "per_axis")
    per_family = era_boundaries._segmentation_costs(
        cycle, axes, "per_axis_family"
    )
    assert not np.allclose(per_axis[np.isfinite(per_axis)], per_family[np.isfinite(per_family)])
    with pytest.raises(ValueError, match="weighting"):
        era_boundaries._segmentation_costs(cycle, axes, "invented")
    with pytest.raises(ValueError, match="missing axes"):
        era_boundaries._segmentation_costs(cycle, ["topic__missing"])
    broken = cycle.copy()
    broken.loc[0, "topic__one"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        era_boundaries._segmentation_costs(broken, axes)


def test_v3_artifacts_cover_every_declared_sensitivity_and_support_contract():
    analysis = era_boundaries.load()
    sensitivity = analysis["sensitivity"]
    expected = {row["condition"] for row in era_boundaries.sensitivity_specifications()}
    assert set(sensitivity["condition"]) == expected
    assert analysis["meta"]["schema_version"] == "era-boundaries-v3"
    assert analysis["meta"]["n_sensitivity_conditions"] == 25
    assert analysis["meta"]["n_estimable_sensitivity_conditions"] == 24
    assert analysis["meta"]["n_unavailable_sensitivity_conditions"] == 1
    assert set(sensitivity.loc[sensitivity["status"] == "computed", "k"]) == set(range(2, 13))
    assert set(sensitivity["grid_offset"]) == {0, 1, 2, 3}
    assert set(sensitivity["min_units"]) == {3, 4}
    assert set(sensitivity["weighting"]) == {"per_axis", "per_axis_family"}

    unavailable = sensitivity[sensitivity["condition"] == "sotu_annual_only"]
    assert unavailable["status"].tolist() == ["not_estimable"]
    assert "1933-1936" in unavailable.iloc[0]["status_reason"]

    primary = sensitivity[sensitivity["condition"] == "primary"]
    assert primary["segment_start"].astype(int).tolist() == [
        1789, 1805, 1849, 1869, 1913, 1937, 1953, 1981, 2017
    ]
    assert primary.iloc[-1]["right_censored"]
    assert not primary.iloc[-1]["terminal_cycle_complete"]
    assert primary["n_paragraphs"].sum() == 36_229
    assert primary["n_speeches"].sum() == 1_057


def test_v3_receipt_hashes_rows_and_story_contract():
    analysis = era_boundaries.load()
    meta = analysis["meta"]
    expected_story = [
        {
            "key": spec.key,
            "label": spec.label,
            "start_year": spec.start_year,
            "end_year": spec.end_year,
            "section_key": spec.section_key,
        }
        for spec in era_profiles.ERA_PROFILE_SPECS
    ]
    assert meta["story_eras"] == expected_story
    for filename, expected_hash in meta["artifact_sha256"].items():
        path = era_boundaries.ANALYSIS_DIR / filename
        assert era_boundaries._sha256(path) == expected_hash
        key = next(
            key for key, value in {
                "sliding_scores": era_boundaries.SLIDING_SCORES_PATH.name,
                "cluster_stability": era_boundaries.CLUSTER_STABILITY_PATH.name,
                "consensus_scores": era_boundaries.CONSENSUS_SCORES_PATH.name,
                "cycle_fingerprints": era_boundaries.CYCLE_FINGERPRINTS_PATH.name,
                "segmentation": era_boundaries.SEGMENTATION_PATH.name,
                "sensitivity": era_boundaries.SENSITIVITY_PATH.name,
                "boundary_evidence": era_boundaries.BOUNDARY_EVIDENCE_PATH.name,
                "boundary_drivers": era_boundaries.BOUNDARY_DRIVERS_PATH.name,
            }.items() if value == filename
        )
        assert meta["artifact_rows"][filename] == len(analysis[key])


def test_boundary_evidence_and_driver_denominators_are_complete():
    analysis = era_boundaries.load()
    evidence = analysis["boundary_evidence"]
    drivers = analysis["boundary_drivers"]
    assert evidence["editorial_start"].tolist() == [
        spec.start_year for spec in era_profiles.ERA_PROFILE_SPECS[1:]
    ]
    assert evidence.iloc[-1]["right_censored"]
    assert set(evidence["n_sensitivity_conditions"]) == {25}
    assert set(evidence["n_estimable_conditions"]) == {24}
    assert set(evidence["n_unavailable_conditions"]) == {1}
    assert evidence.loc[
        evidence["editorial_start"] == 1809, "penalty_pct_of_optimum"
    ].iloc[0] == pytest.approx(0.058089, abs=1e-6)
    assert evidence.loc[
        evidence["editorial_start"] == 1933, "penalty_pct_of_optimum"
    ].iloc[0] == pytest.approx(0.454048, abs=1e-6)
    assert drivers.groupby("editorial_start").size().eq(5).all()
    assert {
        "left_equal_cycle_value", "right_equal_cycle_value",
        "left_story_pooled_value", "right_story_pooled_value",
        "n_left_paragraphs", "n_right_paragraphs",
    }.issubset(drivers.columns)


def test_write_public_projection_includes_v3_and_compatibility_exports(tmp_path):
    analysis = era_boundaries.load()
    era_boundaries_site.write_era_boundaries(tmp_path, analysis)
    expected = {
        "era-boundary-sliding-scores.csv",
        "era-boundary-cluster-stability.csv",
        "era-boundary-consensus-scores.csv",
        "era-boundary-cycle4-fingerprints.csv",
        "era-boundary-segmentation.csv",
        "era-boundary-sensitivity.csv",
        "era-boundary-evidence.csv",
        "era-boundary-drivers.csv",
        "era-boundaries-meta.json",
    }
    assert expected.issubset({path.name for path in (tmp_path / "data").iterdir()})
    receipt = json.loads((tmp_path / "data" / "era-boundaries-meta.json").read_text())
    assert receipt["schema_version"] == "era-boundaries-v3"
    assert receipt["public_manifest_schema"] == "era-boundaries-public-manifest-v3"
    assert receipt["generation_status"] == "deterministic_local_zero_api_calls"
    canonical = tmp_path / "data" / "era-boundaries-v3"
    assert (canonical / "manifest.json").read_bytes() == (
        tmp_path / "data" / "era-boundaries-meta.json"
    ).read_bytes()
    assert set(path.name for path in canonical.iterdir()) == {
        *(expected - {"era-boundaries-meta.json"}), "manifest.json"
    }
    for filename, record in receipt["public_projection"].items():
        canonical_payload = (canonical / filename).read_bytes()
        assert canonical_payload == (tmp_path / "data" / filename).read_bytes()
        assert record["sha256"] == "sha256:" + hashlib.sha256(
            canonical_payload
        ).hexdigest()
        assert record["bytes"] == len(canonical_payload)
    assert era_boundaries_site.validate_public_projection(tmp_path) == receipt


def test_load_rejects_a_mutated_committed_artifact(tmp_path, monkeypatch):
    path_constants = {
        "SLIDING_SCORES_PATH": era_boundaries.SLIDING_SCORES_PATH,
        "CLUSTER_STABILITY_PATH": era_boundaries.CLUSTER_STABILITY_PATH,
        "CONSENSUS_SCORES_PATH": era_boundaries.CONSENSUS_SCORES_PATH,
        "CYCLE_FINGERPRINTS_PATH": era_boundaries.CYCLE_FINGERPRINTS_PATH,
        "SEGMENTATION_PATH": era_boundaries.SEGMENTATION_PATH,
        "SENSITIVITY_PATH": era_boundaries.SENSITIVITY_PATH,
        "BOUNDARY_EVIDENCE_PATH": era_boundaries.BOUNDARY_EVIDENCE_PATH,
        "BOUNDARY_DRIVERS_PATH": era_boundaries.BOUNDARY_DRIVERS_PATH,
        "META_PATH": era_boundaries.META_PATH,
    }
    for constant, source in path_constants.items():
        destination = tmp_path / source.name
        shutil.copyfile(source, destination)
        monkeypatch.setattr(era_boundaries, constant, destination)
    assert era_boundaries.load()["meta"]["schema_version"] == "era-boundaries-v3"
    with (tmp_path / era_boundaries.SENSITIVITY_PATH.name).open("ab") as handle:
        handle.write(b"mutation")
    with pytest.raises(ValueError, match="hashes do not match"):
        era_boundaries.load()


def test_bundle_publication_restores_previous_directory_on_swap_failure(
    tmp_path, monkeypatch
):
    destination = tmp_path / "era_boundaries"
    destination.mkdir()
    (destination / "old.txt").write_text("governed-old")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "new.txt").write_text("candidate-new")
    real_replace = era_boundaries.os.replace

    def fail_candidate_swap(source, target):
        if Path(source) == candidate and Path(target) == destination:
            raise OSError("simulated swap failure")
        return real_replace(source, target)

    monkeypatch.setattr(era_boundaries.os, "replace", fail_candidate_swap)
    with pytest.raises(OSError, match="simulated"):
        era_boundaries._publish_bundle(candidate, destination)
    assert (destination / "old.txt").read_text() == "governed-old"
    assert not (destination / "new.txt").exists()


def test_public_transaction_restores_bundle_and_aliases_on_failure(
    tmp_path, monkeypatch
):
    versioned = tmp_path / "data" / "era-boundaries-v3"
    versioned.mkdir(parents=True)
    (versioned / "old.csv").write_bytes(b"old bundle\n")
    alias = tmp_path / "data" / "era-boundary-sensitivity.csv"
    alias.write_bytes(b"old alias\n")
    page = tmp_path / "era-boundaries.html"
    page.write_bytes(b"old page\n")
    real_replace = era_boundaries_site.os.replace

    def fail_page_install(source, target):
        if Path(target) == page and Path(source).parent.name == "staged":
            raise OSError("simulated public failure")
        return real_replace(source, target)

    monkeypatch.setattr(era_boundaries_site.os, "replace", fail_page_install)
    with pytest.raises(OSError, match="simulated public failure"):
        era_boundaries_site._publish_public_transaction(
            tmp_path,
            {"new.csv": b"new bundle\n", "manifest.json": b"{}\n"},
            {alias: b"new alias\n", page: b"new page\n"},
        )
    assert (versioned / "old.csv").read_bytes() == b"old bundle\n"
    assert not (versioned / "new.csv").exists()
    assert alias.read_bytes() == b"old alias\n"
    assert page.read_bytes() == b"old page\n"
