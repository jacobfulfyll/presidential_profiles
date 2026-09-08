from __future__ import annotations

import copy
import gzip
import subprocess

import pytest

from presidential_profiles.profile_connections_assets import (
    PROFILE_CONNECTIONS_CSS,
    PROFILE_CONNECTIONS_JS,
    PROFILE_CONNECTIONS_JS_CONFIG_KEYS,
    ProfileConnectionsRenderError,
    profile_connections_asset_sizes,
    render_profile_connections,
)


FUNCTIONS = (
    "legacy/inheritance",
    "institutional precedent",
    "policy inheritance",
    "historical comparison",
    "contemporary rivalry",
    "ceremonial/biographical",
    "other/unclear",
)
STANCES = ("positive", "negative", "mixed", "neutral", "unclear")
MILLER = "https://millercenter.org/the-presidency/presidential-speeches/test-address"
def _topic_edge(edge_id: str, president: str, topic: str, share: float) -> dict:
    return {
        "edge_id": edge_id,
        "edge_type": "president_topic",
        "scope_type": "corpus",
        "scope_id": "all-corpus",
        "topic_level": "level1",
        "president_profile_id": president,
        "topic_id": topic,
        "topic_paragraph_count": 12,
        "topic_appearance_count": 4,
        "eligible_president_paragraph_count": 100,
        "eligible_president_appearance_count": 8,
        "scope_topic_paragraph_count": 80,
        "scope_eligible_paragraph_count": 1000,
        "speaker_paragraph_share": share,
        "topic_contribution_share": 0.15,
        "topic_scope_share": 0.08,
        "president_support_status": "supported",
        "edge_support_status": "supported",
        "default_visible": True,
        "evidence_receipt_count": 1,
    }


def _invocation_edge(
    edge_id: str,
    source: str,
    source_name: str,
    target: str,
    target_name: str,
    *,
    reference_paragraphs: int = 1,
    raw_mentions: int = 1,
    distinct_speeches: int = 1,
) -> dict:
    return {
        "edge_id": edge_id,
        "edge_type": "actual_speaker_invocation",
        "source_president_profile_id": source,
        "source_president_name": source_name,
        "target_president_profile_id": target,
        "target_president_name": target_name,
        "raw_mentions": raw_mentions,
        "reference_paragraph_count": reference_paragraphs,
        "distinct_speeches": distinct_speeches,
        "function_counts": {
            name: raw_mentions if name == "historical comparison" else 0
            for name in FUNCTIONS
        },
        "stance_counts": {
            name: raw_mentions if name == "neutral" else 0
            for name in STANCES
        },
        "first_speech_date": "2001-01-01",
        "last_speech_date": "2001-01-01",
        "evidence_row_count": 1,
        "evidence_status": "accepted",
    }


def _receipt(candidate_id: str, edge_id: str, *, source: str = "alpha", target: str = "beta") -> dict:
    return {
        "candidate_id": candidate_id,
        "edge_id": edge_id,
        "source_president_profile_id": source,
        "source_president_name": "President <Alpha>",
        "target_president_profile_id": target,
        "target_president_name": "President Beta",
        "doc_name": "test-address",
        "para_idx": 2,
        "speech_date": "2001-01-01",
        "story_era_id": "modern",
        "story_era_label": "Modern",
        "speech_title": "A <script>safe</script> address",
        "source_url": MILLER,
        "raw_mention": "President Beta",
        "function": "historical comparison",
        "stance": "neutral",
        "evidence_span": "A <b>named comparison</b>.",
        "rationale": "Closed-rubric result.",
        "quotation_status": "narrative",
        "target_status": "former",
        "rubric_version": "invocation-v2",
        "evidence_status": "accepted",
        "annotation_speaker": "President Alpha",
        "source_document_owner_profile_id": "alpha",
        "source_document_owner": "President Alpha",
        "cross_owner": False,
    }


@pytest.fixture()
def contexts() -> tuple[dict, dict, dict]:
    view = {
        "schema_version": "president-profile-v3",
        "president": "President Alpha",
        "slug": "alpha",
        "_view": {"display_name": "President <Alpha>"},
    }
    presidents = [
        {"president_profile_id": "alpha", "president_name": "President Alpha",
         "president_display_name": "President <Alpha>", "slug": "alpha", "party": "A",
         "display_order": 1, "node_type": "president"},
        {"president_profile_id": "beta", "president_name": "President Beta",
         "president_display_name": "President Beta", "slug": "beta", "party": "B",
         "display_order": 2, "node_type": "president"},
        {"president_profile_id": "gamma", "president_name": "President Gamma",
         "president_display_name": "President Gamma", "slug": "gamma", "party": "C",
         "display_order": 3, "node_type": "president"},
    ]
    topics = [
        {"topic_id": "economy", "topic_level": "level1", "topic_label": "Economy & work",
         "topic_definition": "Economic topics", "topic_kind": "substantive",
         "parent_topic_id": None, "parent_topic_label": None, "display_order": 1,
         "node_type": "topic"},
        {"topic_id": "security", "topic_level": "level1", "topic_label": "Security",
         "topic_definition": "Security topics", "topic_kind": "substantive",
         "parent_topic_id": None, "parent_topic_label": None, "display_order": 2,
         "node_type": "topic"},
    ]
    focal = [
        _topic_edge("topic-alpha-economy", "alpha", "economy", 0.20),
        _topic_edge("topic-alpha-security", "alpha", "security", 0.10),
    ]
    peers = [
        _topic_edge("topic-beta-economy", "beta", "economy", 0.18),
        _topic_edge("topic-gamma-security", "gamma", "security", 0.12),
    ]
    outgoing = [
        _invocation_edge(
            f"inv-out-{number}", "alpha", "President Alpha", "beta",
            f"President Beta {number + 1}",
            reference_paragraphs=10 - number,
            raw_mentions=12 - number,
            distinct_speeches=max(1, 5 - number),
        )
        for number in range(7)
    ]
    incoming = [
        _invocation_edge(
            "inv-in-0", "beta", "President Beta", "alpha", "President Alpha",
            reference_paragraphs=5, raw_mentions=6, distinct_speeches=3,
        )
    ]
    invocation_receipts = {
        edge["edge_id"]: [_receipt(f'candidate-{edge["edge_id"]}', edge["edge_id"],
                                   source=edge["source_president_profile_id"],
                                   target=edge["target_president_profile_id"])]
        for edge in [*outgoing, *incoming]
    }
    selected = {
        edge_id: [rows[0]["candidate_id"]]
        for edge_id, rows in invocation_receipts.items()
    }
    index = {
        "schema_version": "president-profile-context-index-v1",
        "presidents": presidents,
        "level1_topics": topics,
        "topic_edges": peers,
        "invocation_edges": [*outgoing, *incoming],
    }
    shard = {
        "schema_version": "president-profile-context-president-v1",
        "president": presidents[0],
        "president_support": {
            "president_profile_id": "alpha",
            "eligible_president_paragraph_count": 100,
            "eligible_president_appearance_count": 8,
            "president_support_status": "supported",
        },
        "topic_edges": focal,
        "topic_receipts": [{
            "receipt_id": "receipt-alpha-economy",
            "edge_id": "topic-alpha-economy",
            "doc_name": "test-address",
            "para_idx": 2,
            "appearance_id": "appearance-1",
            "actual_speaker_profile_id": "alpha",
            "actual_speaker": "President <Alpha>",
            "source_document_owner_profile_id": "alpha",
            "source_document_owner": "President Alpha",
            "cross_owner": False,
            "story_era_id": "modern",
            "story_era_label": "Modern",
            "speech_date": "2001-01-01",
            "speech_title": "A <script>safe</script> address",
            "source_url": MILLER,
            "evidence_excerpt": "An <em>economic</em> excerpt.",
            "selection_role": "first receipt",
            "selection_position": 0,
        }],
        "topic_selections": {
            "default": {
                "focal_edge_ids": ["topic-alpha-economy"],
                "topic_node_ids": ["economy"],
                "peer_president_profile_ids": ["beta"],
                "peer_edge_ids": ["topic-beta-economy"],
                "node_ids": ["alpha", "economy", "beta"],
                "edge_ids": ["topic-alpha-economy", "topic-beta-economy"],
                "mobile_topic_peer_node_ids": ["beta"],
            },
            "expanded": {
                "focal_edge_ids": ["topic-alpha-economy", "topic-alpha-security"],
                "topic_node_ids": ["economy", "security"],
                "peer_president_profile_ids": ["beta", "gamma"],
                "peer_edge_ids": ["topic-beta-economy", "topic-gamma-security"],
                "node_ids": ["alpha", "economy", "security", "beta", "gamma"],
                "edge_ids": ["topic-alpha-economy", "topic-alpha-security",
                             "topic-beta-economy", "topic-gamma-security"],
                "mobile_topic_peer_node_ids": ["beta", "gamma"],
            },
            "observed_thin": {
                "focal_edge_ids": [], "topic_node_ids": [],
                "peer_president_profile_ids": [], "peer_edge_ids": [],
                "node_ids": [], "edge_ids": [], "mobile_topic_peer_node_ids": [],
            },
        },
        "invocations": {
            "outgoing_edge_ids": [edge["edge_id"] for edge in outgoing],
            "incoming_edge_ids": [edge["edge_id"] for edge in incoming],
            "receipts_by_edge": invocation_receipts,
            "selected_receipt_ids_by_edge": selected,
        },
    }
    return view, index, shard


def test_server_renderer_has_stable_sections_default_graph_and_no_js_fallback(contexts):
    page = render_profile_connections(*contexts)

    assert page.startswith('<section class="profile-connections" id="connections"')
    assert page.count('id="topic-network"') == 1
    assert page.count('id="invocations"') == 1
    assert "Fixed radial layout" in page
    assert "president → topic → president" in page
    assert "speaker_paragraph_share" in page
    assert 'data-pc-relation="t0"' in page
    assert 'class="pc-topic-node pc-topic-control' in page
    assert 'role="button" tabindex="0"' in page
    assert 'aria-pressed="false" aria-label="Select Economy &amp; work shared-topic paths' in page
    assert 'class="pc-topic-hit-area"' in page
    assert "Economy &amp; work" in page
    assert "20.00%" in page
    assert "Topic contribution share" not in page
    assert "Topic scope share" not in page
    assert "Complete focal Level-1 topic record" not in page
    assert "Actual-speaker topic edge" not in page
    assert "An &lt;em&gt;economic&lt;/em&gt; excerpt." not in page
    assert "similarity score, influence claim, or summed prominence" in page
    assert "Select a topic in the graph or the list to emphasize its shared paths" in page
    assert "Hover or focus" not in page
    assert "data-pc-initial-payload" not in page


def test_invocations_share_one_segmented_location_with_server_fallback(contexts):
    page = render_profile_connections(*contexts)
    outgoing = page.split('id="pc-invocation-outgoing-alpha"', 1)[1].split(
        'id="pc-invocation-incoming-alpha"', 1
    )[0]
    incoming = page.split('id="pc-invocation-incoming-alpha"', 1)[1]

    assert 'role="tab"' not in page
    assert "data-pc-tab" not in page
    assert "pc-invocation-svg" not in page
    assert page.count("<svg") == 1  # topic ego only
    assert 'role="radiogroup" aria-label="Invocation view"' in page
    assert 'data-pc-invocation-switch hidden' in page
    assert page.count('data-pc-invocation-choice') == 2
    assert '<span>Invoked</span>' in page
    assert '<span>Invoked by</span>' in page
    assert 'data-default-invocation-direction="outgoing"' in page
    assert 'value="outgoing"' in page and 'aria-controls="pc-invocation-outgoing-alpha" checked' in page
    assert "Most invoked former presidents" in page
    assert "Most frequent later invokers" in page
    assert outgoing.count('class="pc-invocation-bar-row"') == 5
    assert incoming.count('class="pc-invocation-bar-row"') == 1
    assert [outgoing.index(f"President Beta {number}") for number in range(1, 6)] == sorted(
        outgoing.index(f"President Beta {number}") for number in range(1, 6)
    )
    assert "10 reference paragraphs" in outgoing
    assert "12 raw mentions" in outgoing
    assert "5 distinct speeches" in outgoing
    # One shared per-profile maximum makes widths comparable across directions.
    assert 'style="width:100.0000%"' in outgoing
    assert 'style="width:50.0000%"' in incoming
    assert "Complete outgoing relationship list (7)" in page
    assert "Complete incoming relationship list (1)" in page
    assert 'data-pc-invocation-direction="outgoing"' in page
    assert 'data-pc-invocation-direction="incoming"' in page
    assert "Show more relationships" not in page
    assert "Function counts" in page and "Stance counts" in page
    assert "First distinct speech" in page
    assert "A &lt;script&gt;safe&lt;/script&gt; address" in page
    assert "A <script>safe</script> address" not in page
    assert "Bar length compares reference-paragraph counts across both directions" in page
    assert "Valid 2025–2026 later-corpus Trump → Biden evidence" in page
    assert "actual_speaker_invocation_edges_v1.csv" in page
    assert "actual_speaker_invocation_evidence_v1.csv" in page


def test_server_renderer_emits_explicit_thin_and_empty_states(contexts):
    view, index, shard = copy.deepcopy(contexts)
    shard["president_support"].update({
        "president_support_status": "thin",
        "eligible_president_paragraph_count": 57,
        "eligible_president_appearance_count": 1,
    })
    shard["topic_selections"]["observed_thin"] = {
        "focal_edge_ids": ["topic-alpha-economy"],
        "topic_node_ids": ["economy"],
        "peer_president_profile_ids": [],
        "peer_edge_ids": [],
        "node_ids": ["alpha", "economy"],
        "edge_ids": ["topic-alpha-economy"],
        "mobile_topic_peer_node_ids": [],
    }
    shard["invocations"] = {
        "outgoing_edge_ids": [], "incoming_edge_ids": [],
        "receipts_by_edge": {}, "selected_receipt_ids_by_edge": {},
    }
    page = render_profile_connections(view, index, shard)

    assert "Thin actual-speaker record." in page
    assert "57 eligible paragraphs across 1 appearances" in page
    assert "no peer comparison or comparative ranking" in page
    assert "Show observed thin-record topics" in page
    assert 'aria-label="No comparative topic network for President &lt;Alpha&gt;" focusable="false"' in page
    assert page.count("No qualifying named former-president invocation was found in this corpus") == 2
    assert "This does not mean none happened historically" in page
    assert 'id="pc-invocation-outgoing-alpha"' in page
    assert 'id="pc-invocation-incoming-alpha"' in page
    assert 'data-default-invocation-direction="incoming"' in page


def test_renderer_consumes_only_precomputed_node_and_edge_id_lists(contexts):
    view, index, shard = copy.deepcopy(contexts)
    for selection in shard["topic_selections"].values():
        for redundant in (
            "focal_edge_ids", "topic_node_ids", "peer_president_profile_ids", "peer_edge_ids",
        ):
            selection.pop(redundant)
    page = render_profile_connections(view, index, shard)

    assert 'data-pc-relation="t0"' in page
    assert "Economy &amp; work" in page
    assert "Show expanded topic network" in page

    view, index, shard = copy.deepcopy(contexts)
    shard["topic_selections"]["default"]["mobile_topic_peer_node_ids"] = ["not-selected"]
    with pytest.raises(ProfileConnectionsRenderError, match="at most six selected nodes"):
        render_profile_connections(view, index, shard)


def test_renderer_rejects_unsafe_receipt_and_asset_urls(contexts):
    view, index, shard = copy.deepcopy(contexts)
    leading = shard["invocations"]["outgoing_edge_ids"][0]
    shard["invocations"]["receipts_by_edge"][leading][0]["source_url"] = "https://evil.example/speech"
    with pytest.raises(ProfileConnectionsRenderError, match="unsafe Miller"):
        render_profile_connections(view, index, shard)

    view, index, shard = copy.deepcopy(contexts)
    with pytest.raises(ProfileConnectionsRenderError, match="profile-context"):
        render_profile_connections(view, index, shard, index_url="https://evil.example/index.json")


def test_renderer_rejects_nonfinite_share_and_incomplete_count_enums(contexts):
    view, index, shard = copy.deepcopy(contexts)
    shard["topic_edges"][0]["speaker_paragraph_share"] = float("nan")
    with pytest.raises(ProfileConnectionsRenderError, match="not finite"):
        render_profile_connections(view, index, shard)

    view, index, shard = copy.deepcopy(contexts)
    index["invocation_edges"][0]["function_counts"].pop("other/unclear")
    with pytest.raises(ProfileConnectionsRenderError, match="incomplete"):
        render_profile_connections(view, index, shard)


def test_renderer_emits_no_trailing_whitespace(contexts):
    view, index, shard = contexts
    page = render_profile_connections(view, index, shard)

    assert all(line == line.rstrip() for line in page.splitlines())


def test_css_locks_responsive_accessibility_and_system_preferences():
    assert "min-height:44px" in PROFILE_CONNECTIONS_CSS
    assert "outline:3px" in PROFILE_CONNECTIONS_CSS
    assert "@media(max-width:768px)" in PROFILE_CONNECTIONS_CSS
    assert "@media(max-width:390px)" in PROFILE_CONNECTIONS_CSS
    assert ".pc-mobile-hidden{display:none}" in PROFILE_CONNECTIONS_CSS
    assert "prefers-reduced-motion:reduce" in PROFILE_CONNECTIONS_CSS
    assert "forced-colors:active" in PROFILE_CONNECTIONS_CSS
    assert "stroke-dasharray" in PROFILE_CONNECTIONS_CSS
    assert "currentColor" in PROFILE_CONNECTIONS_CSS
    assert ".pc-invocation-directions{margin-top" in PROFILE_CONNECTIONS_CSS
    assert ".pc-invocation-bar-track" in PROFILE_CONNECTIONS_CSS
    assert ".pc-segmented-control" in PROFILE_CONNECTIONS_CSS
    assert ".pc-segmented-control input:checked+span" in PROFILE_CONNECTIONS_CSS
    assert ".pc-topic-control:focus-visible" in PROFILE_CONNECTIONS_CSS
    assert ".pc-invocation-edge" not in PROFILE_CONNECTIONS_CSS
    assert ".pc-tablist" not in PROFILE_CONNECTIONS_CSS
    assert ".is-preview" not in PROFILE_CONNECTIONS_CSS


def test_module_contract_is_lazy_safe_transient_and_within_budget(tmp_path):
    sizes = profile_connections_asset_sizes()
    assert sizes["javascript_raw_bytes"] == len(PROFILE_CONNECTIONS_JS.encode()) <= 60_000
    assert sizes["javascript_gzip_bytes"] == len(
        gzip.compress(PROFILE_CONNECTIONS_JS.encode(), compresslevel=9, mtime=0)
    ) <= 20_000
    assert PROFILE_CONNECTIONS_JS_CONFIG_KEYS == (
        "indexUrl", "shardUrl", "presidentProfileId",
    )
    assert "export async function enhanceProfileConnections(root, config)" in PROFILE_CONNECTIONS_JS
    assert "Promise.all([fetchImpl(indexUrl" in PROFILE_CONNECTIONS_JS
    assert "if (loaded) return loaded" in PROFILE_CONNECTIONS_JS
    assert "if (loading) return loading" in PROFILE_CONNECTIONS_JS
    assert "requestData(false)" in PROFILE_CONNECTIONS_JS
    assert "IntersectionObserver" not in PROFILE_CONNECTIONS_JS  # the page bootstrap owns proximity
    assert "innerHTML" not in PROFILE_CONNECTIONS_JS
    assert "eval(" not in PROFILE_CONNECTIONS_JS
    assert "localStorage" not in PROFILE_CONNECTIONS_JS
    assert "sessionStorage" not in PROFILE_CONNECTIONS_JS
    assert "history." not in PROFILE_CONNECTIONS_JS
    assert "location.hash" not in PROFILE_CONNECTIONS_JS
    assert "createElement(" in PROFILE_CONNECTIONS_JS
    assert "createElementNS(" in PROFILE_CONNECTIONS_JS
    assert "textContent" in PROFILE_CONNECTIONS_JS
    assert 'event.key === "Escape"' in PROFILE_CONNECTIONS_JS
    assert 'event.key === "Enter" || event.key === " "' in PROFILE_CONNECTIONS_JS
    assert "pointerover" not in PROFILE_CONNECTIONS_JS
    assert "pointerout" not in PROFILE_CONNECTIONS_JS
    assert 'addEventListener("focusin"' not in PROFILE_CONNECTIONS_JS
    assert 'addEventListener("focusout"' not in PROFILE_CONNECTIONS_JS
    assert "previewRelationFor" not in PROFILE_CONNECTIONS_JS
    assert "is-preview" not in PROFILE_CONNECTIONS_JS
    assert "renderInvocation" not in PROFILE_CONNECTIONS_JS
    assert "buildInvocationSvg" not in PROFILE_CONNECTIONS_JS
    assert "data-pc-tab" not in PROFILE_CONNECTIONS_JS
    assert "data-pc-readout" not in PROFILE_CONNECTIONS_JS
    assert "appendTopicReadout" not in PROFILE_CONNECTIONS_JS
    assert "showInvocationDirection" in PROFILE_CONNECTIONS_JS
    assert 'invocationSwitch.addEventListener("change"' in PROFILE_CONNECTIONS_JS
    assert 'invocationSwitch.addEventListener("keydown"' in PROFILE_CONNECTIONS_JS
    assert '["ArrowLeft", "ArrowRight", "Home", "End"]' in PROFILE_CONNECTIONS_JS
    assert 'node.dataset.pcRelation === key ? "true" : "false"' in PROFILE_CONNECTIONS_JS
    assert "DecompressionStream" not in PROFILE_CONNECTIONS_JS
    assert "retryUsed" in PROFILE_CONNECTIONS_JS
    assert "restoreCompactTopic()" in PROFILE_CONNECTIONS_JS
    assert 'button.setAttribute("aria-expanded", String(expanded))' in PROFILE_CONNECTIONS_JS
    assert "if (restoreFocus) button.focus()" in PROFILE_CONNECTIONS_JS
    assert '"Show compact topic network"' in PROFILE_CONNECTIONS_JS
    assert '"Hide observed thin-record topics"' in PROFILE_CONNECTIONS_JS
    assert PROFILE_CONNECTIONS_JS.index("renderTopic(root, payload.index, payload.shard, mode)") \
        < PROFILE_CONNECTIONS_JS.index("setToggleState(button, true)")

    module = tmp_path / "profile-connections-v1.mjs"
    module.write_text(PROFILE_CONNECTIONS_JS, encoding="utf-8")
    subprocess.run(["node", "--check", str(module)], check=True, capture_output=True, text=True)
