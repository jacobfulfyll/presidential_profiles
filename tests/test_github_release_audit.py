from pathlib import Path
import runpy


AUDIT = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "audit_github_release.py")
)


def test_release_allowlist_accepts_portable_project_paths():
    allowed = AUDIT["allowed"]
    assert allowed(".gitattributes")
    assert allowed("README.md")
    assert allowed(".codex/knowledge/README.md")
    assert allowed("data/networks/network_atlas.json")
    assert allowed("docs/index.html")
    assert allowed("src/presidential_profiles/site.py")


def test_release_boundary_rejects_control_plane_and_telemetry():
    forbidden = AUDIT["forbidden"]
    assert forbidden("data/annotation_ledger/materialized/current")
    assert forbidden(".codex/operators/arcv1_codex_runner.py")
    assert forbidden(".claude/stats.json")
    assert not forbidden("data/networks/network_atlas.json")


def test_release_boundary_allows_removing_but_never_publishing_telemetry():
    policy = AUDIT["release_path_error"]
    assert policy(".claude/stats.json", present_in_index=False) is None
    assert policy(".claude/stats.json", present_in_index=True) == (
        "forbidden release path: .claude/stats.json"
    )


def test_unlisted_top_level_paths_fail_closed():
    allowed = AUDIT["allowed"]
    assert not allowed(".env")
    assert not allowed("scratch.txt")
    assert not allowed("outputs/private-export.zip")


def test_release_size_limits_match_github_boundaries():
    assert AUDIT["MAX_GIT_BLOB"] == 100 * 1024 * 1024
    assert AUDIT["MAX_PAGES_TREE"] == 1024 * 1024 * 1024
