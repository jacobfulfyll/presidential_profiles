"""Server fallback and progressive assets for president-profile Connections.

``render_profile_connections`` accepts a normalized ``president-profile-v3``
view plus the in-memory ``president-profile-context-v1`` index and the matching
president shard.  It renders the complete no-JavaScript fallback: the default
topic ego diagram, semantic focal-topic controls, top-five invocation bar
lists, audit receipts, limitations, empty/thin states, and governed CSV links.

The page bootstrap dynamically imports :data:`PROFILE_CONNECTIONS_JS` near the
section and calls ``enhanceProfileConnections(root, config)``.  The required
configuration keys are ``indexUrl``, ``shardUrl``, and
``presidentProfileId``.  Importing or initializing the module never fetches.
Only an explicit topic expansion starts the concurrent, single-flight index
and shard requests.  The expanded topic view can always be returned to its
server-rendered default.  ``fetchImpl`` is an optional test seam.
"""

from __future__ import annotations

import html
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit


PROFILE_CONNECTIONS_JS_CONFIG_KEYS = (
    "indexUrl",
    "shardUrl",
    "presidentProfileId",
)

_MILLER_URL_RE = re.compile(
    r"https://millercenter\.org/the-presidency/presidential-speeches/"
    r"[a-z0-9]+(?:-[a-z0-9]+)*"
)
_PROFILE_ID_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_CONTEXT_ASSET_RE = re.compile(
    r"(?:\.\./|\./|/)?data/profile-context/(?:"
    r"index_v1\.json|presidents/[a-z0-9]+(?:-[a-z0-9]+)*_v1\.json|"
    r"actual_speaker_invocation_(?:edges|evidence)_v1\.csv)"
)

_FUNCTION_ORDER = (
    "legacy/inheritance",
    "institutional precedent",
    "policy inheritance",
    "historical comparison",
    "contemporary rivalry",
    "ceremonial/biographical",
    "other/unclear",
)
_STANCE_ORDER = ("positive", "negative", "mixed", "neutral", "unclear")


class ProfileConnectionsRenderError(ValueError):
    """The server-rendered Connections fallback could not be produced safely."""


def _escape(value: Any, *, quote: bool = False) -> str:
    return html.escape(str(value), quote=quote)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileConnectionsRenderError(f"{label} must be an object")
    return value


def _rows(value: Any, label: str) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProfileConnectionsRenderError(f"{label} must be an array")
    rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(value):
        rows.append(_mapping(row, f"{label}[{index}]"))
    return rows


def _strings(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProfileConnectionsRenderError(f"{label} must be an array")
    result = [str(item) for item in value]
    if len(set(result)) != len(result):
        raise ProfileConnectionsRenderError(f"{label} contains duplicate identifiers")
    return result


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ProfileConnectionsRenderError(f"{label} is not numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProfileConnectionsRenderError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise ProfileConnectionsRenderError(f"{label} is not finite")
    return number


def _integer(value: Any, label: str) -> int:
    number = _finite(value, label)
    if not number.is_integer() or number < 0:
        raise ProfileConnectionsRenderError(f"{label} is not a non-negative integer")
    return int(number)


def _pct(value: Any) -> str:
    return f"{_finite(value, 'share') * 100:.2f}%"


def _safe_miller_url(value: Any) -> str:
    url = str(value or "")
    if not _MILLER_URL_RE.fullmatch(url):
        raise ProfileConnectionsRenderError("receipt contains an unsafe Miller Center URL")
    return url


def _safe_context_asset(value: Any, label: str) -> str:
    url = str(value or "")
    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or "\\" in url:
        raise ProfileConnectionsRenderError(f"{label} must be a local profile-context asset")
    if not _CONTEXT_ASSET_RE.fullmatch(url):
        raise ProfileConnectionsRenderError(f"{label} is outside data/profile-context")
    return url


def _index_by(rows: Iterable[Mapping[str, Any]], key: str, label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        identity = str(row.get(key, ""))
        if not identity or identity in result:
            raise ProfileConnectionsRenderError(f"{label} has a missing or duplicate {key}")
        result[identity] = row
    return result


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _president_name(row: Mapping[str, Any]) -> str:
    return str(_first(row, "president_display_name", "president_name", default=""))


def _wrap_words(value: Any, *, width: int = 15, lines: int = 3) -> list[str]:
    words = str(value).split()
    if not words:
        return [""]
    wrapped: list[str] = []
    current = words.pop(0)
    while words and len(wrapped) + 1 < lines:
        if len(current) + 1 + len(words[0]) <= width:
            current += " " + words.pop(0)
        else:
            wrapped.append(current)
            current = words.pop(0)
    if words:
        current += " " + " ".join(words)
    wrapped.append(current)
    return wrapped


def _svg_number(value: float) -> str:
    """Return a stable compact coordinate without implying analytical precision."""
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _svg_label(label: Any, x: float, y: float, css_class: str) -> str:
    lines = _wrap_words(label)
    start = y - (len(lines) - 1) * 7
    tspans = "".join(
        f'<tspan x="{_svg_number(x)}" y="{_svg_number(start + index * 14)}">{_escape(line)}</tspan>'
        for index, line in enumerate(lines)
    )
    return f'<text class="{css_class}" aria-hidden="true">{tspans}</text>'


def _share_width(value: Any) -> float:
    # Direct, bounded mapping of this individual edge's speaker share.
    return 1.25 + min(max(_finite(value, "speaker_paragraph_share"), 0.0), 1.0) * 26.0


def _topic_radius(value: Any) -> float:
    # Circle area, not radius, increases linearly with the individual focal share.
    share = min(max(_finite(value, "speaker_paragraph_share"), 0.0), 1.0)
    return math.sqrt(196.0 + share * 1_900.0)


def _selection(shard: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    selections = _mapping(shard.get("topic_selections", {}), "topic_selections")
    value = selections.get(name, {})
    return _mapping(value, f"topic_selections.{name}")


def _selection_ids(selection: Mapping[str, Any], key: str) -> list[str]:
    return _strings(selection.get(key, []), f"topic selection {key}")


def _resolved_selection(
    selection: Mapping[str, Any],
    *,
    profile_id: str,
    topic_by_id: Mapping[str, Mapping[str, Any]],
    president_by_id: Mapping[str, Mapping[str, Any]],
    edge_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Resolve typed subsets from producer-ordered node/edge ID lists.

    The public contract only needs to persist ``node_ids`` and ``edge_ids``.
    Older focused fixtures may additionally carry redundant typed lists.  This
    adapter classifies IDs by their governed record type and focal identity; it
    never sorts, ranks, scores, or invents a relationship.
    """
    node_ids = _selection_ids(selection, "node_ids")
    edge_ids = _selection_ids(selection, "edge_ids")
    unknown_nodes = set(node_ids) - ({profile_id} | set(topic_by_id) | set(president_by_id))
    unknown_edges = set(edge_ids) - set(edge_by_id)
    if unknown_nodes or unknown_edges:
        raise ProfileConnectionsRenderError("topic selection refers to unknown nodes or edges")
    topic_ids = (
        _selection_ids(selection, "topic_node_ids")
        if "topic_node_ids" in selection
        else [node_id for node_id in node_ids if node_id in topic_by_id]
    )
    peer_ids = (
        _selection_ids(selection, "peer_president_profile_ids")
        if "peer_president_profile_ids" in selection
        else [
            node_id for node_id in node_ids
            if node_id in president_by_id and node_id != profile_id
        ]
    )
    focal_ids = (
        _selection_ids(selection, "focal_edge_ids")
        if "focal_edge_ids" in selection
        else [
            edge_id for edge_id in edge_ids
            if str(edge_by_id[edge_id].get("president_profile_id")) == profile_id
        ]
    )
    peer_edge_ids = (
        _selection_ids(selection, "peer_edge_ids")
        if "peer_edge_ids" in selection
        else [edge_id for edge_id in edge_ids if edge_id not in set(focal_ids)]
    )
    mobile_ids = _selection_ids(selection, "mobile_topic_peer_node_ids")
    typed = {
        "node_ids": node_ids,
        "edge_ids": edge_ids,
        "topic_node_ids": topic_ids,
        "peer_president_profile_ids": peer_ids,
        "focal_edge_ids": focal_ids,
        "peer_edge_ids": peer_edge_ids,
        "mobile_topic_peer_node_ids": mobile_ids,
    }
    if set(topic_ids) - set(topic_by_id) or set(peer_ids) - set(president_by_id):
        raise ProfileConnectionsRenderError("topic selection contains a mistyped node")
    if set(focal_ids + peer_edge_ids) != set(edge_ids):
        raise ProfileConnectionsRenderError("typed topic edges do not cover the selection")
    if len(mobile_ids) > 6 or set(mobile_ids) - set(topic_ids + peer_ids):
        raise ProfileConnectionsRenderError("mobile topic selection must contain at most six selected nodes")
    return typed


def _topic_svg(
    *,
    profile_id: str,
    display_name: str,
    selection: Mapping[str, Any],
    topic_by_id: Mapping[str, Mapping[str, Any]],
    president_by_id: Mapping[str, Mapping[str, Any]],
    edge_by_id: Mapping[str, Mapping[str, Any]],
    thin: bool = False,
) -> str:
    resolved = _resolved_selection(
        selection,
        profile_id=profile_id,
        topic_by_id=topic_by_id,
        president_by_id=president_by_id,
        edge_by_id=edge_by_id,
    )
    topic_ids = resolved["topic_node_ids"]
    peer_ids = resolved["peer_president_profile_ids"]
    focal_edge_ids = resolved["focal_edge_ids"]
    peer_edge_ids = resolved["peer_edge_ids"]
    mobile_ids = set(
        resolved["mobile_topic_peer_node_ids"]
        if resolved["mobile_topic_peer_node_ids"]
        else [*topic_ids, *peer_ids][:6]
    )
    if not topic_ids:
        return (
            '<svg class="pc-network-svg" viewBox="0 0 720 560" role="img" '
            f'aria-label="No comparative topic network for {_escape(display_name, quote=True)}" focusable="false">'
            f'<circle class="pc-focal-node" cx="360" cy="280" r="22"/>'
            f'{_svg_label(display_name, 360, 335, "pc-svg-label pc-focal-label")}'
            '<text class="pc-svg-empty" x="360" y="405">No supported peer comparison</text>'
            '</svg>'
        )

    missing_topics = set(topic_ids) - set(topic_by_id)
    missing_peers = set(peer_ids) - set(president_by_id)
    missing_edges = set(focal_edge_ids + peer_edge_ids) - set(edge_by_id)
    if missing_topics or missing_peers or missing_edges:
        raise ProfileConnectionsRenderError(
            "topic selection refers to unknown nodes or edges"
        )

    center_x, center_y = 360.0, 280.0
    relation_by_topic = {
        topic_id: f"t{index}" for index, topic_id in enumerate(topic_ids)
    }
    topic_positions: dict[str, tuple[float, float]] = {}
    for index, topic_id in enumerate(topic_ids):
        angle = -math.pi / 2 + 2 * math.pi * index / max(1, len(topic_ids))
        topic_positions[topic_id] = (
            center_x + math.cos(angle) * 132.0,
            center_y + math.sin(angle) * 132.0,
        )
    peer_positions: dict[str, tuple[float, float]] = {}
    for index, peer_id in enumerate(peer_ids):
        angle = -math.pi / 2 + 2 * math.pi * index / max(1, len(peer_ids))
        peer_positions[peer_id] = (
            center_x + math.cos(angle) * 246.0,
            center_y + math.sin(angle) * 224.0,
        )

    edge_parts: list[str] = []
    focal_by_topic: dict[str, Mapping[str, Any]] = {}
    for edge_id in focal_edge_ids:
        edge = edge_by_id[edge_id]
        topic_id = str(edge.get("topic_id", ""))
        if str(edge.get("president_profile_id", "")) != profile_id or topic_id not in topic_positions:
            raise ProfileConnectionsRenderError("focal topic edge disagrees with its selection")
        focal_by_topic[topic_id] = edge
        x, y = topic_positions[topic_id]
        relation = relation_by_topic[topic_id]
        dash = " pc-observed-edge" if thin else ""
        mobile_class = "" if topic_id in mobile_ids else " pc-mobile-hidden"
        edge_parts.append(
            f'<line class="pc-topic-edge{dash}{mobile_class}" x1="{_svg_number(center_x)}" y1="{_svg_number(center_y)}" '
            f'x2="{_svg_number(x)}" y2="{_svg_number(y)}" style="--pc-edge-width:{_svg_number(_share_width(edge.get("speaker_paragraph_share")))}px" '
            f'data-pc-relation="{_escape(relation, quote=True)}"/>'
        )
    for edge_id in peer_edge_ids:
        edge = edge_by_id[edge_id]
        topic_id = str(edge.get("topic_id", ""))
        peer_id = str(edge.get("president_profile_id", ""))
        if topic_id not in topic_positions or peer_id not in peer_positions:
            raise ProfileConnectionsRenderError("peer topic edge disagrees with its selection")
        x1, y1 = peer_positions[peer_id]
        x2, y2 = topic_positions[topic_id]
        mobile_class = "" if peer_id in mobile_ids and topic_id in mobile_ids else " pc-mobile-hidden"
        relation = relation_by_topic[topic_id]
        edge_parts.append(
            f'<line class="pc-topic-edge pc-peer-edge{mobile_class}" x1="{_svg_number(x1)}" y1="{_svg_number(y1)}" '
            f'x2="{_svg_number(x2)}" y2="{_svg_number(y2)}" style="--pc-edge-width:{_svg_number(_share_width(edge.get("speaker_paragraph_share")))}px" '
            f'data-pc-relation="{_escape(relation, quote=True)}"/>'
        )

    topic_parts: list[str] = []
    for topic_id in topic_ids:
        topic = topic_by_id[topic_id]
        edge = focal_by_topic.get(topic_id)
        if edge is None:
            raise ProfileConnectionsRenderError("topic node has no focal edge")
        x, y = topic_positions[topic_id]
        relation = relation_by_topic[topic_id]
        mobile_class = "" if topic_id in mobile_ids else " pc-mobile-hidden"
        topic_label = str(topic.get("topic_label", topic_id))
        topic_parts.append(
            f'<g class="pc-topic-node pc-topic-control{mobile_class}" role="button" tabindex="0" '
            f'aria-pressed="false" aria-label="Select {_escape(topic_label, quote=True)} shared-topic paths, '
            f'{_pct(edge.get("speaker_paragraph_share"))} of eligible actual-speaker paragraphs" '
            f'data-pc-relation="{_escape(relation, quote=True)}">'
            f'<circle class="pc-topic-hit-area" cx="{_svg_number(x)}" cy="{_svg_number(y)}" r="{_svg_number(max(22.0, _topic_radius(edge.get("speaker_paragraph_share"))))}"/>'
            f'<circle cx="{_svg_number(x)}" cy="{_svg_number(y)}" r="{_svg_number(_topic_radius(edge.get("speaker_paragraph_share")))}"/>'
            f'{_svg_label(topic_label, x, y + 4, "pc-svg-label pc-topic-label")}</g>'
        )

    peer_parts: list[str] = []
    for peer_id in peer_ids:
        president = president_by_id[peer_id]
        x, y = peer_positions[peer_id]
        mobile_class = "" if peer_id in mobile_ids else " pc-mobile-hidden"
        related_topics = {
            str(edge_by_id[edge_id].get("topic_id", ""))
            for edge_id in peer_edge_ids
            if str(edge_by_id[edge_id].get("president_profile_id", "")) == peer_id
        }
        relations = " ".join(
            relation_by_topic[topic_id]
            for topic_id in topic_ids if topic_id in related_topics
        )
        peer_parts.append(
            f'<g class="pc-president-node pc-peer-node{mobile_class}" data-pc-relations="{_escape(relations, quote=True)}">'
            f'<circle cx="{_svg_number(x)}" cy="{_svg_number(y)}" r="22"/>'
            f'{_svg_label(_president_name(president), x, y + 37, "pc-svg-label pc-peer-label")}</g>'
        )

    title = (
        f"Actual-speaker topic ego network for {display_name}. "
        "Topics occupy the inner ring and supported comparison presidents the outer ring."
    )
    return (
        '<svg class="pc-network-svg" viewBox="0 0 720 560" role="group" '
        f'aria-label="{_escape(title, quote=True)}" focusable="false">'
        + "".join(edge_parts)
        + "".join(topic_parts)
        + "".join(peer_parts)
        + f'<circle class="pc-focal-node" cx="{_svg_number(center_x)}" cy="{_svg_number(center_y)}" r="22"/>'
        + _svg_label(display_name, center_x, center_y + 4, "pc-svg-label pc-focal-label")
        + "</svg>"
    )


def _topic_controls(
    *,
    profile_id: str,
    selection: Mapping[str, Any],
    topic_by_id: Mapping[str, Mapping[str, Any]],
    president_by_id: Mapping[str, Mapping[str, Any]],
    edge_by_id: Mapping[str, Mapping[str, Any]],
) -> str:
    resolved = _resolved_selection(
        selection,
        profile_id=profile_id,
        topic_by_id=topic_by_id,
        president_by_id=president_by_id,
        edge_by_id=edge_by_id,
    )
    topic_ids = resolved["topic_node_ids"]
    focal_ids = resolved["focal_edge_ids"]
    peer_ids = resolved["peer_edge_ids"]
    focal_by_topic = {
        str(edge_by_id[edge_id].get("topic_id", "")): edge_by_id[edge_id]
        for edge_id in focal_ids
    }
    peers_by_topic: dict[str, list[Mapping[str, Any]]] = {topic_id: [] for topic_id in topic_ids}
    for edge_id in peer_ids:
        edge = edge_by_id[edge_id]
        peers_by_topic.setdefault(str(edge.get("topic_id", "")), []).append(edge)
    controls: list[str] = []
    for topic_index, topic_id in enumerate(topic_ids):
        topic = topic_by_id[topic_id]
        edge = focal_by_topic.get(topic_id)
        if edge is None or str(edge.get("president_profile_id", "")) != profile_id:
            raise ProfileConnectionsRenderError("topic control has no matching focal edge")
        relation = f"t{topic_index}"
        label = str(topic.get("topic_label", topic_id))
        peers = peers_by_topic.get(topic_id, [])
        controls.append(
            f'<li><button type="button" class="pc-relationship-button" aria-pressed="false" '
            f'data-pc-relation="{_escape(relation, quote=True)}"><strong>{_escape(label)}</strong>'
            f'<span>{_pct(edge.get("speaker_paragraph_share"))} of eligible actual-speaker paragraphs'
            f' · {len(peers)} comparison president{"s" if len(peers) != 1 else ""}</span></button></li>'
        )
    return "<ol class=\"pc-relationship-list\" data-pc-topic-list>" + "".join(controls) + "</ol>"


def _count_items(counts: Any, order: Sequence[str], label: str) -> str:
    values = _mapping(counts, label)
    if set(values) != set(order):
        raise ProfileConnectionsRenderError(f"{label} does not contain the complete fixed enum")
    return "".join(
        f'<li><span>{_escape(item)}</span><strong>{_integer(values[item], label):,}</strong></li>'
        for item in order
    )


def _selected_invocation_receipts(
    edge_id: str,
    invocation: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    receipt_map = _mapping(invocation.get("receipts_by_edge", {}), "invocations.receipts_by_edge")
    available = _rows(receipt_map.get(edge_id, []), f"receipts for {edge_id}")
    by_id = {str(row.get("candidate_id", "")): row for row in available}
    positions_map = _mapping(
        invocation.get("selected_receipt_positions_by_edge", {}),
        "invocation selected receipt positions",
    )
    positions = positions_map.get(edge_id)
    roles = ("First distinct speech", "Lower-middle distinct speech", "Last distinct speech")
    if positions is not None:
        if isinstance(positions, (str, bytes)) or not isinstance(positions, Sequence):
            raise ProfileConnectionsRenderError("selected invocation receipt positions have invalid shape")
        result: list[tuple[str, Mapping[str, Any]]] = []
        seen_positions: set[int] = set()
        for role, raw_position in zip(roles, positions, strict=False):
            position = _integer(raw_position, "selected invocation receipt position")
            if position >= len(available):
                raise ProfileConnectionsRenderError("selected invocation receipt position is missing")
            if position not in seen_positions:
                seen_positions.add(position)
                result.append((role, available[position]))
        return result
    selected_map = _mapping(
        _first(
            invocation,
            "selected_receipt_ids_by_edge",
            "selected_evidence_candidate_ids_by_edge",
            default={},
        ),
        "invocation selected receipts",
    )
    selected = selected_map.get(edge_id, [])
    pairs: list[tuple[str, str]] = []
    if isinstance(selected, Mapping):
        role_keys = ("first", "lower_middle", "last")
        for role, key in zip(roles, role_keys, strict=True):
            if key in selected:
                pairs.append((role, str(selected[key])))
    elif isinstance(selected, Sequence) and not isinstance(selected, (str, bytes)):
        pairs = [(roles[min(index, 2)], str(candidate_id)) for index, candidate_id in enumerate(selected)]
    elif selected:
        raise ProfileConnectionsRenderError("selected invocation receipts have invalid shape")
    result: list[tuple[str, Mapping[str, Any]]] = []
    seen: set[str] = set()
    for role, candidate_id in pairs:
        if candidate_id in seen:
            continue
        if candidate_id not in by_id:
            raise ProfileConnectionsRenderError("selected invocation receipt is missing")
        seen.add(candidate_id)
        result.append((role, by_id[candidate_id]))
    if not result and available:
        result.append(("Audit example", available[0]))
    return result


def _invocation_receipt_html(role: str, receipt: Mapping[str, Any]) -> str:
    url = _safe_miller_url(receipt.get("source_url"))
    key = f'({receipt.get("doc_name", "")}, {receipt.get("para_idx", "")})'
    title = str(receipt.get("speech_title") or receipt.get("doc_name") or "Corpus speech")
    actual = str(receipt.get("source_president_name") or receipt.get("annotation_speaker") or "Unresolved")
    owner = str(receipt.get("source_document_owner") or "Unresolved")
    cross = "cross-owner" if bool(receipt.get("cross_owner")) else "same owner and speaker"
    return f'''<article class="pc-receipt">
  <p class="pc-receipt-role">{_escape(role)} · keyed paragraph {_escape(key)}</p>
  <blockquote><p>{_escape(receipt.get("evidence_span") or receipt.get("raw_mention") or "No evidence span available.")}</p></blockquote>
  <p><a href="{_escape(url, quote=True)}">{_escape(title)}</a> · {_escape(receipt.get("speech_date") or "Date unavailable")}</p>
  <p>Function: <strong>{_escape(receipt.get("function") or "unclear")}</strong> · Stance:
  <strong>{_escape(receipt.get("stance") or "unclear")}</strong> · Target status:
  {_escape(receipt.get("target_status") or "unavailable")}.</p>
  <p>Actual speaker: <strong>{_escape(actual)}</strong> · Source-document owner:
  <strong>{_escape(owner)}</strong> · {_escape(cross)}.</p>
  <p class="pc-rationale">Annotation rationale: {_escape(receipt.get("rationale") or "Not supplied.")}</p>
</article>'''


def _invocation_audit_html(
    edge: Mapping[str, Any],
    invocation: Mapping[str, Any],
) -> str:
    edge_id = str(edge.get("edge_id", ""))
    receipts = _selected_invocation_receipts(edge_id, invocation)
    receipts_html = "".join(_invocation_receipt_html(role, row) for role, row in receipts)
    if not receipts_html:
        receipts_html = '<p class="pc-empty">No selected keyed audit receipt is available.</p>'
    functions = _count_items(edge.get("function_counts"), _FUNCTION_ORDER, "function_counts")
    stances = _count_items(edge.get("stance_counts"), _STANCE_ORDER, "stance_counts")
    return f'''<article class="pc-invocation-audit-record">
  <p class="pc-kicker">Named former-president invocation</p>
  <h5>{_escape(edge.get("source_president_name"))} → {_escape(edge.get("target_president_name"))}</h5>
  <dl class="pc-audit-values">
    <div><dt>Reference paragraphs</dt><dd>{_integer(edge.get("reference_paragraph_count"), "reference paragraphs"):,}</dd></div>
    <div><dt>Raw mentions</dt><dd>{_integer(edge.get("raw_mentions"), "raw mentions"):,}</dd></div>
    <div><dt>Distinct speeches</dt><dd>{_integer(edge.get("distinct_speeches"), "distinct speeches"):,}</dd></div>
    <div><dt>Speech range</dt><dd>{_escape(edge.get("first_speech_date"))}–{_escape(edge.get("last_speech_date"))}</dd></div>
  </dl>
  <div class="pc-count-columns"><section><h6>Function counts</h6><ul>{functions}</ul></section>
  <section><h6>Stance counts</h6><ul>{stances}</ul></section></div>
  <h6>Distinct-speech audit examples</h6>{receipts_html}
  <p class="pc-limitation">These are deterministic audit examples, not proof of intent,
  representativeness, influence, historical importance, or policy success. The download contains every retained row.</p>
</article>'''


def _invocation_edge_summary(edge: Mapping[str, Any], direction: str) -> str:
    counterpart = (
        edge.get("target_president_name")
        if direction == "outgoing"
        else edge.get("source_president_name")
    )
    verb = "invoked" if direction == "outgoing" else "invoked this president"
    return (
        f'<strong>{_escape(counterpart)}</strong><span>{verb} in '
        f'{_integer(edge.get("reference_paragraph_count"), "reference paragraphs"):,} reference paragraphs · '
        f'{_integer(edge.get("raw_mentions"), "raw mentions"):,} raw mentions · '
        f'{_integer(edge.get("distinct_speeches"), "distinct speeches"):,} speeches</span>'
    )


def _invocation_direction(
    *,
    direction: str,
    profile_id: str,
    edge_ids: Sequence[str],
    edge_by_id: Mapping[str, Mapping[str, Any]],
    invocation: Mapping[str, Any],
    maximum_reference_paragraphs: int,
) -> str:
    label = (
        "Most invoked former presidents"
        if direction == "outgoing"
        else "Most frequent later invokers"
    )
    description = (
        "Up to five named former presidents invoked by this actual speaker, in governed producer order."
        if direction == "outgoing"
        else (
            "Up to five actual speakers who invoked this president in later corpus speeches, "
            "in governed producer order."
        )
    )
    resolved_edges: list[Mapping[str, Any]] = []
    for edge_id in edge_ids:
        if edge_id not in edge_by_id:
            raise ProfileConnectionsRenderError("invocation list refers to an unknown edge")
        edge = edge_by_id[edge_id]
        if direction == "outgoing":
            if str(edge.get("source_president_profile_id")) != profile_id:
                raise ProfileConnectionsRenderError("outgoing invocation edge has the wrong source")
        elif str(edge.get("target_president_profile_id")) != profile_id:
            raise ProfileConnectionsRenderError("incoming invocation edge has the wrong target")
        if set(_mapping(edge.get("function_counts"), "function_counts")) != set(_FUNCTION_ORDER):
            raise ProfileConnectionsRenderError("function_counts has an incomplete fixed enum")
        if set(_mapping(edge.get("stance_counts"), "stance_counts")) != set(_STANCE_ORDER):
            raise ProfileConnectionsRenderError("stance_counts has an incomplete fixed enum")
        resolved_edges.append(edge)

    leading = resolved_edges[:5]
    bar_rows: list[str] = []
    for edge in leading:
        counterpart = (
            edge.get("target_president_name")
            if direction == "outgoing"
            else edge.get("source_president_name")
        )
        paragraphs = _integer(edge.get("reference_paragraph_count"), "reference paragraphs")
        raw_mentions = _integer(edge.get("raw_mentions"), "raw mentions")
        speeches = _integer(edge.get("distinct_speeches"), "distinct speeches")
        width = paragraphs / maximum_reference_paragraphs * 100 if maximum_reference_paragraphs else 0
        bar_rows.append(f'''<li class="pc-invocation-bar-row">
  <div class="pc-invocation-bar-heading"><strong>{_escape(counterpart)}</strong>
  <span>{paragraphs:,} reference paragraph{"s" if paragraphs != 1 else ""}</span></div>
  <span class="pc-invocation-bar-track" aria-hidden="true"><i style="width:{width:.4f}%"></i></span>
  <p>{raw_mentions:,} raw mention{"s" if raw_mentions != 1 else ""} ·
  {speeches:,} distinct speech{"es" if speeches != 1 else ""}</p>
</li>''')
    bars = (
        f'<ol class="pc-invocation-bars" aria-label="{_escape(label, quote=True)}">'
        + "".join(bar_rows)
        + "</ol>"
        if bar_rows else ""
    )
    full_html = "".join(
        f'<li>{_invocation_edge_summary(edge_by_id[edge_id], direction)}</li>'
        for edge_id in edge_ids
    )
    audit = (
        '<details class="pc-invocation-audit"><summary>Audit details for the leading relationship</summary>'
        + _invocation_audit_html(leading[0], invocation)
        + "</details>"
        if leading else ""
    )
    empty = (
        '<p class="pc-empty">No qualifying named former-president invocation was found in this corpus. '
        'This does not mean none happened historically.</p>'
        if not edge_ids else ""
    )
    panel_id = f"pc-invocation-{direction}-{profile_id}"
    heading_id = f"{panel_id}-heading"
    return f'''<section class="pc-direction" id="{panel_id}" data-pc-invocation-direction="{direction}" aria-labelledby="{heading_id}">
  <h4 id="{heading_id}">{_escape(label)}</h4><p>{_escape(description)}</p>
  {empty}{bars}{audit}
  <details class="pc-complete-list"><summary>Complete {direction} relationship list ({len(edge_ids)})</summary><ol>{full_html}</ol></details>
</section>'''


def render_profile_connections(
    president_view: Mapping[str, Any],
    context_index: Mapping[str, Any],
    context_shard: Mapping[str, Any],
    *,
    index_url: str = "../data/profile-context/index_v1.json",
    shard_url: str | None = None,
    invocation_edges_url: str = "../data/profile-context/actual_speaker_invocation_edges_v1.csv",
    invocation_evidence_url: str = "../data/profile-context/actual_speaker_invocation_evidence_v1.csv",
) -> str:
    """Render the accessible, no-request Connections baseline for one profile.

    ``context_index`` and ``context_shard`` are the validated public payloads,
    not browser-derived analytics.  The renderer preserves every producer
    ordering and never computes topic or invocation ranks.
    """
    view = _mapping(president_view, "president view")
    index = _mapping(context_index, "profile context index")
    shard = _mapping(context_shard, "profile context shard")
    president_record = _mapping(shard.get("president", {}), "shard president")
    profile_id = str(president_record.get("president_profile_id") or view.get("slug") or "")
    if not _PROFILE_ID_RE.fullmatch(profile_id):
        raise ProfileConnectionsRenderError("invalid president profile identifier")
    if str(view.get("slug", profile_id)) != profile_id:
        raise ProfileConnectionsRenderError("profile view and context shard disagree")
    display = _mapping(view.get("_view", {}), "profile display fields")
    display_name = str(
        president_record.get("president_display_name")
        or display.get("display_name")
        or president_record.get("president_name")
        or view.get("president")
        or profile_id
    )
    if shard_url is None:
        shard_url = f"../data/profile-context/presidents/{profile_id}_v1.json"
    safe_index_url = _safe_context_asset(index_url, "index URL")
    safe_shard_url = _safe_context_asset(shard_url, "shard URL")
    safe_edges_url = _safe_context_asset(invocation_edges_url, "edge download URL")
    safe_evidence_url = _safe_context_asset(invocation_evidence_url, "evidence download URL")

    presidents = _rows(index.get("presidents"), "context presidents")
    topics = _rows(index.get("level1_topics"), "context Level-1 topics")
    index_topic_edges = _rows(index.get("topic_edges"), "context topic edges")
    shard_topic_edges = _rows(context_shard.get("topic_edges"), "shard topic edges")
    _rows(context_shard.get("topic_receipts"), "shard topic receipts")
    invocation_edges = _rows(index.get("invocation_edges"), "context invocation edges")
    president_by_id = _index_by(presidents, "president_profile_id", "president catalog")
    topic_by_id = _index_by(topics, "topic_id", "topic catalog")
    topic_edge_by_id = _index_by(
        [*index_topic_edges, *[row for row in shard_topic_edges if str(row.get("edge_id")) not in {str(item.get("edge_id")) for item in index_topic_edges}]],
        "edge_id",
        "topic edges",
    )
    invocation_edge_by_id = _index_by(invocation_edges, "edge_id", "invocation edges")
    if profile_id not in president_by_id:
        raise ProfileConnectionsRenderError("context president is absent from the index")

    support = _mapping(shard.get("president_support", {}), "president support")
    support_status = str(support.get("president_support_status") or "thin")
    thin = support_status != "supported"
    default_selection = _selection(shard, "default")
    observed_selection = _selection(shard, "observed_thin")
    default_resolved = _resolved_selection(
        default_selection, profile_id=profile_id, topic_by_id=topic_by_id,
        president_by_id=president_by_id, edge_by_id=topic_edge_by_id,
    )
    observed_resolved = _resolved_selection(
        observed_selection, profile_id=profile_id, topic_by_id=topic_by_id,
        president_by_id=president_by_id, edge_by_id=topic_edge_by_id,
    )
    rendered_selection = {} if thin else default_selection
    if thin:
        rendered_selection = {
            "focal_edge_ids": [], "topic_node_ids": [], "peer_president_profile_ids": [],
            "peer_edge_ids": [], "node_ids": [], "edge_ids": [],
        }

    topic_svg = _topic_svg(
        profile_id=profile_id,
        display_name=display_name,
        selection=rendered_selection,
        topic_by_id=topic_by_id,
        president_by_id=president_by_id,
        edge_by_id=topic_edge_by_id,
        thin=thin,
    )
    topic_list = _topic_controls(
        profile_id=profile_id,
        selection=rendered_selection,
        topic_by_id=topic_by_id,
        president_by_id=president_by_id,
        edge_by_id=topic_edge_by_id,
    )
    expanded = _selection(shard, "expanded")
    expanded_resolved = _resolved_selection(
        expanded, profile_id=profile_id, topic_by_id=topic_by_id,
        president_by_id=president_by_id, edge_by_id=topic_edge_by_id,
    )
    topic_controls: list[str] = []
    if thin and observed_resolved["focal_edge_ids"]:
        topic_controls.append(
            '<button type="button" class="pc-expand-button" data-pc-expand="topic-observed" '
            'aria-expanded="false">Show observed thin-record topics</button>'
        )
    elif not thin and expanded_resolved["edge_ids"] != default_resolved["edge_ids"]:
        topic_controls.append(
            '<button type="button" class="pc-expand-button" data-pc-expand="topic-expanded" '
            'aria-expanded="false">Show expanded topic network</button>'
        )
    thin_notice = ""
    if thin:
        paragraphs = _integer(support.get("eligible_president_paragraph_count", 0), "eligible paragraphs")
        appearances = _integer(support.get("eligible_president_appearance_count", 0), "eligible appearances")
        thin_notice = (
            '<aside class="pc-warning" role="note"><strong>Thin actual-speaker record.</strong> '
            f'{paragraphs:,} eligible paragraphs across {appearances:,} appearances. '
            'Observed counts remain available, but this default view makes no peer comparison or comparative ranking.</aside>'
        )
    thin_notice_line = f"    {thin_notice}\n" if thin_notice else ""

    invocation = _mapping(shard.get("invocations", {}), "shard invocations")
    outgoing_ids = _strings(invocation.get("outgoing_edge_ids", []), "outgoing invocation edge IDs")
    incoming_ids = _strings(invocation.get("incoming_edge_ids", []), "incoming invocation edge IDs")
    for edge_id in outgoing_ids + incoming_ids:
        if edge_id not in invocation_edge_by_id:
            raise ProfileConnectionsRenderError("shard invocation edge is absent from index")
    invocation_scale_max = max(
        (
            _integer(
                invocation_edge_by_id[edge_id].get("reference_paragraph_count"),
                "reference paragraphs",
            )
            for edge_id in [*outgoing_ids[:5], *incoming_ids[:5]]
        ),
        default=1,
    )
    outgoing = _invocation_direction(
        direction="outgoing", profile_id=profile_id,
        edge_ids=outgoing_ids, edge_by_id=invocation_edge_by_id, invocation=invocation,
        maximum_reference_paragraphs=invocation_scale_max,
    )
    incoming = _invocation_direction(
        direction="incoming", profile_id=profile_id,
        edge_ids=incoming_ids, edge_by_id=invocation_edge_by_id, invocation=invocation,
        maximum_reference_paragraphs=invocation_scale_max,
    )
    default_invocation_direction = "outgoing" if outgoing_ids else "incoming"
    outgoing_checked = ' checked' if default_invocation_direction == "outgoing" else ""
    incoming_checked = ' checked' if default_invocation_direction == "incoming" else ""
    invocation_switch = f'''<div class="pc-invocation-switch" role="radiogroup" aria-label="Invocation view"
      data-pc-invocation-switch hidden>
      <span class="pc-invocation-switch-label" aria-hidden="true">View</span>
      <div class="pc-segmented-control">
        <label><input type="radio" name="pc-invocation-view-{_escape(profile_id, quote=True)}" value="outgoing"
          data-pc-invocation-choice aria-controls="pc-invocation-outgoing-{_escape(profile_id, quote=True)}"{outgoing_checked}>
          <span>Invoked</span></label>
        <label><input type="radio" name="pc-invocation-view-{_escape(profile_id, quote=True)}" value="incoming"
          data-pc-invocation-choice aria-controls="pc-invocation-incoming-{_escape(profile_id, quote=True)}"{incoming_checked}>
          <span>Invoked by</span></label>
      </div>
    </div>'''
    root_label = f"Connections for {display_name}"
    return f'''<section class="profile-connections" id="connections" aria-label="{_escape(root_label, quote=True)}"
  data-profile-connections data-index-url="{_escape(safe_index_url, quote=True)}"
  data-shard-url="{_escape(safe_shard_url, quote=True)}"
  data-president-profile-id="{_escape(profile_id, quote=True)}">
  <p class="section-kicker">Actual-speaker relationships · governed projections</p><h2>Connections</h2>
  <p class="pc-population"><strong>Connections population:</strong> eligible paragraphs attributed to the actual speaker.
  Core v3 profile measures elsewhere remain based on source documents assigned to this president.</p>
  <p class="pc-scope-note">These corpus relationships describe observed speech evidence. They do not establish historical importance,
  personal affinity, similarity, influence, causality, policy success, or everything that happened outside the corpus.</p>
  <p class="pc-status" role="status" aria-live="polite" data-pc-status></p>
  <button type="button" class="pc-retry" data-pc-retry hidden>Retry Connections data</button>
  <section class="pc-subsection" id="topic-network">
    <p class="pc-kicker">Topic ego · actual speaker</p><h3>Shared topic emphasis</h3>
    <p>Topic prominence and every edge width map the individual <code>speaker_paragraph_share</code>.
    Select a topic in the graph or the list to emphasize its shared paths; select it again or press Escape to clear.
    Ring position and distance have no analytical meaning.</p>
{thin_notice_line}    <div class="pc-network-layout">
      <figure data-pc-topic-stage><p class="pc-network-cue" aria-hidden="true">Swipe or scroll to inspect the network →</p>
        <div class="pc-network-scroll" role="region" aria-label="Shared topic network; scroll horizontally on narrow screens" tabindex="0">{topic_svg}</div>
        <figcaption>Fixed radial layout: focal president at center, topics on the inner ring, and comparison presidents on the outer ring.
        Shared emphasis remains a bipartite path: president → topic → president; president nodes are uniform.</figcaption></figure>
    </div>
    {topic_list}
    <div class="pc-enhancement-controls" data-pc-enhancement-controls hidden>{''.join(topic_controls)}</div>
    <p class="pc-limitation">Multi-label shares are never summed into prominence,
    and no president-president edge, overlap score, confidence, similarity score, influence claim, or summed prominence is created.</p>
  </section>
  <section class="pc-subsection" id="invocations">
    <p class="pc-kicker">Named former presidents · actual speaker</p><h3>Invocations in corpus speeches</h3>
    <p>These lists aggregate accepted actual-speaker evidence. Bar length compares reference-paragraph counts across both directions;
    exact reference-paragraph, raw-mention, and distinct-speech counts are printed for every displayed relationship. No normalized rate,
    confidence, or majority function/stance is derived.</p>
    <p class="pc-limitation"><code>target_status</code> uses corpus speech dates rather than an independent legal-term calendar.
    Valid 2025–2026 later-corpus Trump → Biden evidence is therefore retained.</p>
    {invocation_switch}
    <div class="pc-invocation-directions" data-pc-invocation-directions
      data-default-invocation-direction="{default_invocation_direction}">{outgoing}{incoming}</div>
    <p class="pc-downloads"><a href="{_escape(safe_edges_url, quote=True)}" download>Download every invocation edge (CSV)</a>
    <a href="{_escape(safe_evidence_url, quote=True)}" download>Download every retained invocation evidence row (CSV)</a></p>
  </section>
</section>'''


PROFILE_CONNECTIONS_CSS = r"""
.profile-connections{max-width:100%;overflow:clip}.profile-connections *{box-sizing:border-box;min-width:0}
.pc-population{padding:10px 12px;border-left:4px solid #315f78}.pc-scope-note,.pc-limitation{color:var(--muted);font-size:.8rem;line-height:1.55}
.pc-subsection{margin-top:28px;scroll-margin-top:calc(var(--global-nav-height,0px) + 72px)}
.pc-kicker{margin:0 0 5px;color:#6a421b;font-size:.68rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase}
.pc-status{position:absolute;width:1px;height:1px;margin:-1px;padding:0;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
.profile-connections :where(.pc-retry,.pc-expand-button,.pc-relationship-button,.pc-downloads a,summary,.pc-receipt a,.pc-segmented-control label){min-height:44px}.profile-connections button{font:inherit;cursor:pointer}
.profile-connections :where(button,a,summary):focus-visible{outline:3px solid #b45f06;outline-offset:3px}
.pc-warning{margin:14px 0;padding:12px;border:1px solid #a47b22;border-left-width:5px;background:#fff7df;color:#4e3b13}
.pc-network-layout{margin-top:16px}
.profile-connections :where(figure,.pc-direction){border:1px solid var(--border);border-radius:10px}.profile-connections figure{margin:0;padding:10px;overflow:hidden}
.pc-network-cue{display:none;color:var(--muted);font-size:12px;font-weight:740}.pc-network-scroll{max-width:100%}.pc-network-scroll:focus-visible{outline:3px solid #b45f06;outline-offset:3px}
.profile-connections figcaption{margin:7px 4px 0;color:var(--muted);font-size:.7rem}.pc-network-svg{display:block;width:100%;height:auto;max-height:600px}
.profile-connections .pc-topic-edge{stroke:#72563f;stroke-width:var(--pc-edge-width,2px);stroke-linecap:round;opacity:.52;transition:opacity .15s ease}
.pc-observed-edge{stroke-dasharray:8 6}.pc-topic-node circle:not(.pc-topic-hit-area){fill:#f3dfc5;stroke:#72431e;stroke-width:2}.pc-topic-hit-area{fill:transparent;stroke:none;pointer-events:all}
.pc-topic-control{cursor:pointer}.pc-topic-control:focus-visible{outline:3px solid #b45f06;outline-offset:4px}.pc-topic-control[aria-pressed=true] circle:not(.pc-topic-hit-area){fill:#fff2df;stroke:#9b3f16;stroke-width:5}
.pc-president-node circle{fill:#f7fbfd;stroke:#315f78;stroke-width:2}.pc-focal-node{fill:#fff8ee;stroke:#532e12;stroke-width:3}
.pc-svg-label{fill:#211a15;font:700 11px system-ui,sans-serif;text-anchor:middle;paint-order:stroke;stroke:#fff;stroke-width:3px;pointer-events:none}.pc-topic-label,.pc-focal-label{font-size:10px}.pc-peer-label{font-size:9px}
.pc-svg-empty{fill:#62574f;font:650 15px system-ui,sans-serif;text-anchor:middle}
.profile-connections :where([data-pc-relation],[data-pc-relations]).is-pinned{opacity:1}
.profile-connections line[data-pc-relation].is-pinned{stroke:#9b3f16;stroke-width:calc(var(--pc-edge-width,2px) + 2px)}
.pc-rationale{color:var(--muted)}
.pc-invocation-audit-record :where(h5,h6){margin:8px 0}.pc-invocation-audit-record h5{font:700 1.08rem Georgia,serif}.pc-audit-values,.pc-count-columns{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.pc-audit-values div,.pc-receipt{padding:8px;border:1px solid var(--grid)}.pc-audit-values dt{color:var(--muted);font-size:.68rem}.pc-audit-values dd{margin:3px 0 0;font-weight:700}
.pc-count-columns ul{margin:0;padding:0;list-style:none}.pc-count-columns li{display:flex;justify-content:space-between;gap:8px;border-top:1px solid var(--grid);font-size:.73rem}
.pc-receipt{margin-top:9px}.pc-receipt p{margin:5px 0;font-size:.75rem}.pc-receipt blockquote{margin:7px 0;padding-left:10px;border-left:3px solid currentColor}.pc-receipt a{display:inline-flex;align-items:center}
.pc-relationship-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;margin:12px 0 0;padding:0;list-style:none}
.pc-relationship-button{display:flex;width:100%;align-items:flex-start;justify-content:center;flex-direction:column;padding:8px 10px;border:1px solid #9d826a;background:#fff;color:var(--ink);text-align:left}.pc-relationship-button span{color:var(--muted);font-size:.7rem}
.pc-relationship-button[aria-pressed=true]{border-color:#71330d;background:#fff2df;box-shadow:inset 0 0 0 2px #71330d}
.pc-enhancement-controls{margin-top:10px}.profile-connections :where(.pc-expand-button,.pc-retry){padding:8px 12px;border:1px solid #704421;background:#fff8ee;color:#573116;font-weight:700}
.pc-complete-list{margin-top:12px;border:1px solid var(--border)}.pc-complete-list>summary{display:flex;align-items:center;padding:10px 12px;font-weight:700}.pc-complete-list ol{margin:0 12px 12px;padding-left:22px}.pc-complete-list li{margin:7px 0;font-size:.78rem}
.pc-invocation-switch{display:flex;align-items:center;gap:14px;margin-top:16px}.pc-invocation-switch-label{font-weight:800}.pc-segmented-control{display:inline-grid;grid-template-columns:repeat(2,minmax(112px,1fr));overflow:hidden;border:1px solid #9d826a;border-radius:12px;background:var(--surface)}
.pc-segmented-control label{position:relative;display:flex;align-items:stretch;cursor:pointer}.pc-segmented-control label+label{border-left:1px solid #9d826a}.pc-segmented-control input{position:absolute;width:1px;height:1px;opacity:0}.pc-segmented-control span{display:flex;width:100%;min-height:48px;align-items:center;justify-content:center;padding:8px 18px;color:var(--ink2);font-weight:780}.pc-segmented-control input:checked+span{background:#315f78;color:#fff}.pc-segmented-control input:focus-visible+span{outline:3px solid #b45f06;outline-offset:-4px}
.pc-invocation-directions{margin-top:12px}.pc-direction{padding:16px}.pc-direction h4{margin-top:0}
.pc-invocation-bars{display:grid;gap:13px;margin:16px 0 0;padding:0;list-style:none}.pc-invocation-bar-heading{display:flex;justify-content:space-between;gap:10px;align-items:baseline}.pc-invocation-bar-heading span,.pc-invocation-bar-row p{color:var(--muted);font-size:.72rem;font-variant-numeric:tabular-nums}.pc-invocation-bar-row p{margin:4px 0 0}.pc-invocation-bar-track{display:block;height:14px;margin-top:5px;overflow:hidden;border:1px solid #b9b7b0;border-radius:99px;background:#ecebe7}.pc-invocation-bar-track i{display:block;height:100%;background:#315f78}
.pc-invocation-audit{margin-top:16px;border:1px solid var(--border);border-radius:9px}.pc-invocation-audit>summary{display:flex;align-items:center;padding:10px 12px;font-weight:700}.pc-invocation-audit-record{padding:0 12px 12px}.pc-downloads{display:flex;flex-wrap:wrap;gap:9px}.pc-downloads a{display:inline-flex;align-items:center;padding:6px 9px}
.profile-connections [hidden]{display:none!important}
@media(max-width:768px){.pc-relationship-list{grid-template-columns:1fr}.pc-network-svg{max-height:none}}
@media(max-width:760px){.profile-connections .pc-mobile-hidden{display:none}.pc-network-cue{display:block}.pc-network-scroll{overflow-x:auto;overscroll-behavior-inline:contain;scrollbar-gutter:stable}.pc-network-svg{width:720px;min-width:720px}.profile-connections figcaption,.pc-kicker,.pc-audit-values dt,.pc-count-columns li,.pc-relationship-button span,.pc-invocation-bar-heading span,.pc-invocation-bar-row p,.pc-invocation-audit-record h6{font-size:12px}.pc-svg-label{font-size:11px}.pc-peer-label{font-size:11px}}
@media(max-width:390px){.profile-connections :where(.pc-count-columns,.pc-audit-values){grid-template-columns:1fr}.profile-connections :where(figure,.pc-direction){padding:8px}.pc-invocation-switch{align-items:stretch;gap:8px}.pc-invocation-switch-label{display:flex;align-items:center}.pc-segmented-control{width:100%;grid-template-columns:repeat(2,minmax(0,1fr))}.pc-segmented-control span{padding-inline:10px}}
@media(prefers-reduced-motion:reduce){.profile-connections *,.profile-connections *::before,.profile-connections *::after{scroll-behavior:auto!important;animation:none!important;transition:none!important}}
@media(forced-colors:active){.profile-connections :where(figure,.pc-direction,.pc-warning,.pc-complete-list,.pc-invocation-audit,.pc-invocation-bar-track,button,.pc-segmented-control,.pc-segmented-control label+label){border-color:currentColor}.profile-connections .pc-topic-edge{stroke:currentColor;opacity:1}.profile-connections .pc-invocation-bar-track i{background:Highlight}.profile-connections :where(.pc-topic-node circle:not(.pc-topic-hit-area),.pc-president-node circle,.pc-focal-node){fill:Canvas;stroke:currentColor}.profile-connections :where(.pc-svg-label,.pc-svg-empty){fill:CanvasText;stroke:Canvas}.profile-connections :where(.pc-relationship-button[aria-pressed=true],.pc-segmented-control input:checked+span){forced-color-adjust:none;background:Highlight;color:HighlightText}.profile-connections :where(button,a,summary):focus-visible,.profile-connections .pc-topic-control:focus-visible,.profile-connections .pc-segmented-control input:focus-visible+span{outline-color:Highlight}}
"""


PROFILE_CONNECTIONS_JS = r'''"use strict";

export const PROFILE_CONNECTIONS_REQUIRED_CONFIG = Object.freeze([
  "indexUrl", "shardUrl", "presidentProfileId"
]);

const NS = "http://www.w3.org/2000/svg";
const INDEX_SCHEMA = "president-profile-context-index-v1";
const SHARD_SCHEMA = "president-profile-context-president-v1";

function array(value) { return Array.isArray(value) ? value : []; }
function object(value) { return value && typeof value === "object" && !Array.isArray(value) ? value : {}; }
function finite(value) { const number = Number(value); return Number.isFinite(number) ? number : 0; }
function percent(value) { return `${(finite(value) * 100).toFixed(2)}%`; }
function byId(rows, key) { return new Map(array(rows).map(row => [String(row[key]), row])); }

function safeDataUrl(value, kind, doc) {
  const url = new URL(String(value || ""), doc.baseURI);
  if (url.origin !== doc.location.origin || url.search || url.hash) throw new Error("unsafe context URL");
  const path = url.pathname;
  const valid = kind === "index"
    ? /\/data\/profile-context\/index_v1\.json$/.test(path)
    : /\/data\/profile-context\/presidents\/[a-z0-9]+(?:-[a-z0-9]+)*_v1\.json$/.test(path);
  if (!valid) throw new Error("context URL is outside the governed projection");
  return url.href;
}

function make(doc, tag, className, value) {
  const node = doc.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined) node.textContent = String(value);
  return node;
}

function svg(doc, tag, attrs = {}) {
  const allowed = new Set(["viewBox", "role", "aria-label", "aria-pressed", "tabindex", "focusable", "class", "x", "y", "x1", "y1",
    "x2", "y2", "cx", "cy", "r", "d", "markerWidth", "markerHeight", "refX", "refY", "orient",
    "markerUnits", "id", "marker-end"]);
  const node = doc.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([key, value]) => { if (allowed.has(key)) node.setAttribute(key, String(value)); });
  return node;
}

function textSvg(doc, label, x, y, className) {
  const node = svg(doc, "text", {x, y, class: className});
  node.textContent = String(label || "");
  return node;
}

function setRelation(node, relation) { node.dataset.pcRelation = relation; }
function edgeWidth(value) { return 1.25 + Math.min(Math.max(finite(value), 0), 1) * 26; }
function topicRadius(value) { return Math.sqrt(196 + Math.min(Math.max(finite(value), 0), 1) * 1900); }
function presidentName(row) { return String(row?.president_display_name || row?.president_name || "Unknown president"); }

function resolvedTopicSelection(selection, profileId, topics, presidents, edges) {
  const nodeIds = array(selection.node_ids).map(String), edgeIds = array(selection.edge_ids).map(String);
  if (nodeIds.some(id => id !== profileId && !topics.has(id) && !presidents.has(id))
      || edgeIds.some(id => !edges.has(id))) throw new Error("unknown precomputed topic selection ID");
  const topicIds = array(selection.topic_node_ids).length ? array(selection.topic_node_ids).map(String)
    : nodeIds.filter(id => topics.has(id));
  const peerIds = array(selection.peer_president_profile_ids).length
    ? array(selection.peer_president_profile_ids).map(String)
    : nodeIds.filter(id => id !== profileId && presidents.has(id));
  const focalIds = array(selection.focal_edge_ids).length ? array(selection.focal_edge_ids).map(String)
    : edgeIds.filter(id => String(edges.get(id)?.president_profile_id) === profileId);
  const focalSet = new Set(focalIds);
  const peerEdgeIds = array(selection.peer_edge_ids).length ? array(selection.peer_edge_ids).map(String)
    : edgeIds.filter(id => !focalSet.has(id));
  const mobileIds = array(selection.mobile_topic_peer_node_ids).map(String);
  if (new Set([...focalIds, ...peerEdgeIds]).size !== new Set(edgeIds).size)
    throw new Error("typed topic selection does not cover its precomputed edges");
  if (mobileIds.length > 6 || mobileIds.some(id => !topicIds.includes(id) && !peerIds.includes(id)))
    throw new Error("invalid precomputed mobile topic selection");
  return {node_ids: nodeIds, edge_ids: edgeIds, topic_node_ids: topicIds,
    peer_president_profile_ids: peerIds, focal_edge_ids: focalIds, peer_edge_ids: peerEdgeIds,
    mobile_topic_peer_node_ids: mobileIds};
}

function relationButton(doc, relation, title, summary) {
  const item = make(doc, "li", "");
  const button = make(doc, "button", "pc-relationship-button");
  button.type = "button"; button.setAttribute("aria-pressed", "false"); setRelation(button, relation);
  button.append(make(doc, "strong", "", title), make(doc, "span", "", summary));
  item.append(button); return item;
}

function buildTopicSvg(doc, profileId, displayName, selection, topics, presidents, edges, observed) {
  const graphic = svg(doc, "svg", {class: "pc-network-svg", viewBox: "0 0 720 560", role: "group",
    "aria-label": `Actual-speaker topic ego network for ${displayName}`, focusable: "false"});
  const resolved = resolvedTopicSelection(selection, profileId, topics, presidents, edges);
  const topicIds = resolved.topic_node_ids, peerIds = resolved.peer_president_profile_ids;
  const focalIds = resolved.focal_edge_ids, peerEdgeIds = resolved.peer_edge_ids;
  const relationByTopic = new Map(topicIds.map((id, index) => [id, `t${index}`]));
  const mobileIds = new Set(resolved.mobile_topic_peer_node_ids.length
    ? resolved.mobile_topic_peer_node_ids : [...topicIds, ...peerIds].slice(0, 6));
  const center = [360, 280], topicPositions = new Map(), peerPositions = new Map();
  topicIds.forEach((id, index) => { const angle = -Math.PI / 2 + Math.PI * 2 * index / Math.max(1, topicIds.length);
    topicPositions.set(id, [center[0] + Math.cos(angle) * 132, center[1] + Math.sin(angle) * 132]); });
  peerIds.forEach((id, index) => { const angle = -Math.PI / 2 + Math.PI * 2 * index / Math.max(1, peerIds.length);
    peerPositions.set(id, [center[0] + Math.cos(angle) * 246, center[1] + Math.sin(angle) * 224]); });
  const focalByTopic = new Map();
  focalIds.forEach(id => {
    const edge = edges.get(id); if (!edge) throw new Error("unknown focal topic edge");
    const topicId = String(edge.topic_id), position = topicPositions.get(topicId);
    if (!position || String(edge.president_profile_id) !== profileId) throw new Error("invalid focal topic selection");
    focalByTopic.set(topicId, edge);
    const line = svg(doc, "line", {class: `pc-topic-edge${observed ? " pc-observed-edge" : ""}${mobileIds.has(topicId) ? "" : " pc-mobile-hidden"}`,
      x1: center[0], y1: center[1], x2: position[0], y2: position[1]});
    line.style.setProperty("--pc-edge-width", `${edgeWidth(edge.speaker_paragraph_share)}px`);
    setRelation(line, relationByTopic.get(topicId)); graphic.append(line);
  });
  peerEdgeIds.forEach(id => {
    const edge = edges.get(id); if (!edge) throw new Error("unknown peer topic edge");
    const topicId = String(edge.topic_id), peerId = String(edge.president_profile_id);
    const start = peerPositions.get(peerId), end = topicPositions.get(topicId);
    if (!start || !end) throw new Error("invalid peer topic selection");
    const line = svg(doc, "line", {class: `pc-topic-edge pc-peer-edge${mobileIds.has(peerId) && mobileIds.has(topicId) ? "" : " pc-mobile-hidden"}`,
      x1: start[0], y1: start[1], x2: end[0], y2: end[1]});
    line.style.setProperty("--pc-edge-width", `${edgeWidth(edge.speaker_paragraph_share)}px`);
    setRelation(line, relationByTopic.get(topicId)); graphic.append(line);
  });
  topicIds.forEach(id => {
    const topic = topics.get(id), edge = focalByTopic.get(id), position = topicPositions.get(id);
    if (!topic || !edge) throw new Error("unknown topic node");
    const label = String(topic.topic_label || id);
    const group = svg(doc, "g", {class: `pc-topic-node pc-topic-control${mobileIds.has(id) ? "" : " pc-mobile-hidden"}`,
      role: "button", tabindex: "0", "aria-pressed": "false",
      "aria-label": `Select ${label} shared-topic paths, ${percent(edge.speaker_paragraph_share)} of eligible actual-speaker paragraphs`});
    setRelation(group, relationByTopic.get(id));
    const radius = topicRadius(edge.speaker_paragraph_share);
    group.append(svg(doc, "circle", {class: "pc-topic-hit-area", cx: position[0], cy: position[1], r: Math.max(22, radius)}),
      svg(doc, "circle", {cx: position[0], cy: position[1], r: radius}),
      textSvg(doc, label, position[0], position[1] + 4, "pc-svg-label pc-topic-label"));
    graphic.append(group);
  });
  peerIds.forEach(id => {
    const row = presidents.get(id), position = peerPositions.get(id); if (!row) throw new Error("unknown peer president");
    const group = svg(doc, "g", {class: `pc-president-node pc-peer-node${mobileIds.has(id) ? "" : " pc-mobile-hidden"}`});
    group.dataset.pcRelations = peerEdgeIds.map(edgeId => edges.get(edgeId)).filter(edge =>
      edge && String(edge.president_profile_id) === id).map(edge => relationByTopic.get(String(edge.topic_id))).join(" ");
    group.append(svg(doc, "circle", {cx: position[0], cy: position[1], r: 22}),
      textSvg(doc, presidentName(row), position[0], position[1] + 38, "pc-svg-label pc-peer-label")); graphic.append(group);
  });
  graphic.append(svg(doc, "circle", {class: "pc-focal-node", cx: center[0], cy: center[1], r: 22}),
    textSvg(doc, displayName, center[0], center[1] + 4, "pc-svg-label pc-focal-label"));
  return graphic;
}

function renderTopic(root, index, shard, mode) {
  const doc = root.ownerDocument, selection = object(object(shard.topic_selections)[mode]);
  const topics = byId(index.level1_topics, "topic_id"), presidents = byId(index.presidents, "president_profile_id");
  const edges = byId([...array(index.topic_edges), ...array(shard.topic_edges)], "edge_id");
  const profileId = String(root.dataset.presidentProfileId), displayName = presidentName(presidents.get(profileId));
  const resolved = resolvedTopicSelection(selection, profileId, topics, presidents, edges);
  const stage = root.querySelector("[data-pc-topic-stage]"), oldGraphic = stage?.querySelector("svg");
  if (!stage || !oldGraphic) throw new Error("topic stage is unavailable");
  oldGraphic.replaceWith(buildTopicSvg(doc, profileId, displayName, selection, topics, presidents, edges, mode === "observed_thin"));
  const list = root.querySelector("[data-pc-topic-list]");
  if (!list) throw new Error("topic semantic controls are unavailable");
  list.replaceChildren();
  const focalByTopic = new Map(resolved.focal_edge_ids.map(id => {
    const edge = edges.get(String(id)); return [String(edge?.topic_id), edge]; }));
  const peerEdges = resolved.peer_edge_ids.map(id => edges.get(String(id))).filter(Boolean);
  resolved.topic_node_ids.forEach((topicId, topicIndex) => {
    const topic = topics.get(topicId), focal = focalByTopic.get(topicId); if (!topic || !focal) throw new Error("invalid topic selection");
    const peers = peerEdges.filter(edge => String(edge.topic_id) === topicId);
    const relation = `t${topicIndex}`;
    list.append(relationButton(doc, relation, String(topic.topic_label || topicId),
      `${percent(focal.speaker_paragraph_share)} of eligible actual-speaker paragraphs · ${peers.length} comparison presidents`));
  });
}

export async function enhanceProfileConnections(root, config) {
  if (!root || typeof root.querySelector !== "function") throw new TypeError("Connections root is required");
  if (root.dataset.pcEnhanced === "true") return root.__profileConnectionsController;
  const options = object(config);
  PROFILE_CONNECTIONS_REQUIRED_CONFIG.forEach(key => { if (!options[key]) throw new TypeError(`Missing ${key}`); });
  const doc = root.ownerDocument;
  const indexUrl = safeDataUrl(options.indexUrl, "index", doc), shardUrl = safeDataUrl(options.shardUrl, "shard", doc);
  const profileId = String(options.presidentProfileId);
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(profileId) || profileId !== root.dataset.presidentProfileId)
    throw new Error("Connections president identity mismatch");
  const fetchImpl = options.fetchImpl || globalThis.fetch;
  if (typeof fetchImpl !== "function") throw new TypeError("fetch is unavailable");
  const status = root.querySelector("[data-pc-status]"), retry = root.querySelector("[data-pc-retry]");
  let loaded = null, loading = null, retryUsed = false, pendingAction = null;
  let pinnedKey = null, pinnedControl = null;

  const compactStage = root.querySelector("[data-pc-topic-stage]"),
    compactList = root.querySelector("[data-pc-topic-list]");
  const compactGraphic = compactStage?.querySelector("svg")?.cloneNode(true);
  const compactListChildren = compactList ? [...compactList.childNodes].map(node => node.cloneNode(true)) : [];
  if (!compactStage || !compactList || !compactGraphic)
    throw new Error("compact topic view is unavailable");

  function announce(message) { if (status) status.textContent = message; }
  function payloadValid(index, shard) {
    if (index?.schema_version !== INDEX_SCHEMA || shard?.schema_version !== SHARD_SCHEMA
        || String(shard?.president?.president_profile_id) !== profileId) throw new Error("unsupported Connections payload");
  }
  async function requestData(isRetry) {
    if (loaded) return loaded;
    if (loading) return loading;
    if (isRetry) retryUsed = true;
    announce(isRetry ? "Retrying Connections data…" : "Loading expanded Connections data…");
    loading = Promise.all([fetchImpl(indexUrl, {credentials: "same-origin"}),
      fetchImpl(shardUrl, {credentials: "same-origin"})])
      .then(async responses => {
        if (responses.some(response => !response.ok)) throw new Error("Connections request failed");
        const payloads = await Promise.all(responses.map(response => response.json()));
        payloadValid(payloads[0], payloads[1]); loaded = {index: payloads[0], shard: payloads[1]};
        if (retry) retry.hidden = true; announce("Expanded Connections data loaded."); return loaded;
      }).catch(error => {
        announce(retryUsed ? "Connections data still could not load; the complete server-rendered fallback remains available."
          : "Connections data could not load; the complete server-rendered fallback remains available. Retry is available once.");
        if (retry) retry.hidden = retryUsed; throw error;
      }).finally(() => { loading = null; });
    return loading;
  }

  function showRelation(key) {
    root.querySelectorAll("[data-pc-relation]").forEach(node => {
      const active = node.dataset.pcRelation === key;
      node.classList.toggle("is-pinned", active);
    });
    root.querySelectorAll("[data-pc-relations]").forEach(node => {
      const active = String(node.dataset.pcRelations || "").split(/\s+/).includes(key);
      node.classList.toggle("is-pinned", active);
    });
  }
  function clearPin(restoreFocus) {
    const control = pinnedControl; pinnedKey = null; pinnedControl = null;
    root.querySelectorAll("[data-pc-relation][aria-pressed]").forEach(node => node.setAttribute("aria-pressed", "false"));
    showRelation(null); if (restoreFocus && control) control.focus(); announce("Topic emphasis cleared.");
  }
  function activateRelation(control) {
    const key = control.dataset.pcRelation; if (!key) return;
    if (pinnedKey === key) { clearPin(false); return; }
    pinnedKey = key; pinnedControl = control;
    root.querySelectorAll("[data-pc-relation][aria-pressed]").forEach(node =>
      node.setAttribute("aria-pressed", node.dataset.pcRelation === key ? "true" : "false"));
    showRelation(key); announce("Topic paths emphasized. Select again or press Escape to clear.");
  }

  function restoreCompactTopic() {
    const stage = root.querySelector("[data-pc-topic-stage]"), oldGraphic = stage?.querySelector("svg");
    const list = root.querySelector("[data-pc-topic-list]");
    if (!oldGraphic || !list) throw new Error("topic semantic controls are unavailable");
    oldGraphic.replaceWith(compactGraphic.cloneNode(true));
    list.replaceChildren(...compactListChildren.map(node => node.cloneNode(true)));
  }
  function setToggleState(button, expanded) {
    button.setAttribute("aria-expanded", String(expanded));
    if (!expanded) button.textContent = button.dataset.pcExpand === "topic-observed"
      ? "Show observed thin-record topics" : "Show expanded topic network";
    else button.textContent = button.dataset.pcExpand === "topic-observed"
      ? "Hide observed thin-record topics" : "Show compact topic network";
  }
  function applyExpansion(action, payload, button) {
    const mode = action === "topic-observed" ? "observed_thin" : "expanded";
    clearPin(false); renderTopic(root, payload.index, payload.shard, mode);
    setToggleState(button, true); pendingAction = null;
    announce(action === "topic-observed"
      ? "Observed thin-record topics are shown."
      : "Expanded topic relationships are shown. Use Show compact topic network to return.");
  }
  async function toggleTopic(button) {
    if (button.getAttribute("aria-expanded") === "true") {
      clearPin(false); restoreCompactTopic(); setToggleState(button, false); pendingAction = null;
      announce(button.dataset.pcExpand === "topic-observed"
        ? "Observed thin-record topics are hidden." : "Compact topic relationships are shown.");
      return;
    }
    const action = String(button.dataset.pcExpand || "");
    if (action !== "topic-expanded" && action !== "topic-observed") return;
    const restoreFocus = doc.activeElement === button;
    pendingAction = action; button.disabled = true;
    try { applyExpansion(action, await requestData(false), button); }
    catch (_) { /* The status and one retry preserve the server fallback. */ }
    finally { button.disabled = false; if (restoreFocus) button.focus(); }
  }

  root.addEventListener("click", event => {
    const relation = event.target.closest?.("[data-pc-relation][aria-pressed]");
    if (relation && root.contains(relation)) { activateRelation(relation); return; }
    const button = event.target.closest?.("[data-pc-expand]");
    if (button && root.contains(button)) toggleTopic(button);
  });
  root.addEventListener("keydown", event => {
    const relation = event.target.closest?.("[data-pc-relation][aria-pressed]");
    if (relation && root.contains(relation) && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault(); activateRelation(relation); return;
    }
    if (event.key === "Escape" && pinnedKey) { event.preventDefault(); clearPin(true); }
  });
  if (retry) retry.addEventListener("click", async () => {
    if (retryUsed || !pendingAction) return;
    const action = pendingAction, button = root.querySelector(`[data-pc-expand="${action}"]`);
    if (!button) return;
    button.disabled = true;
    try { applyExpansion(action, await requestData(true), button);
    } catch (_) { /* A second failure is final and the fallback remains. */ }
    finally { button.disabled = false; button.focus(); }
  });

  const invocationSwitch = root.querySelector("[data-pc-invocation-switch]");
  const invocationDirections = root.querySelector("[data-pc-invocation-directions]");
  function showInvocationDirection(direction, shouldAnnounce) {
    if (!invocationDirections || (direction !== "outgoing" && direction !== "incoming")) return;
    invocationDirections.querySelectorAll("[data-pc-invocation-direction]").forEach(panel => {
      panel.hidden = panel.dataset.pcInvocationDirection !== direction;
    });
    if (shouldAnnounce) announce(direction === "outgoing"
      ? "Most invoked former presidents shown." : "Most frequent later invokers shown.");
  }
  if (invocationSwitch && invocationDirections) {
    const initialDirection = String(invocationDirections.dataset.defaultInvocationDirection || "outgoing");
    showInvocationDirection(initialDirection, false); invocationSwitch.hidden = false;
    invocationSwitch.addEventListener("change", event => {
      const choice = event.target.closest?.("[data-pc-invocation-choice]");
      if (choice?.checked) showInvocationDirection(String(choice.value), true);
    });
    invocationSwitch.addEventListener("keydown", event => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      const choices = [...invocationSwitch.querySelectorAll("[data-pc-invocation-choice]")];
      const current = event.target.closest?.("[data-pc-invocation-choice]");
      if (!current || !choices.includes(current)) return;
      event.preventDefault();
      const currentIndex = choices.indexOf(current);
      const nextIndex = event.key === "Home" ? 0 : event.key === "End" ? choices.length - 1
        : event.key === "ArrowLeft" ? (currentIndex - 1 + choices.length) % choices.length
        : (currentIndex + 1) % choices.length;
      const next = choices[nextIndex]; next.checked = true; next.focus();
      showInvocationDirection(String(next.value), true);
    });
  }
  root.querySelectorAll("[data-pc-enhancement-controls]").forEach(node => { node.hidden = false; });
  root.classList.add("is-enhanced"); root.dataset.pcEnhanced = "true";
  const controller = Object.freeze({clearPin: () => clearPin(false)});
  root.__profileConnectionsController = controller; return controller;
}
'''


def profile_connections_asset_sizes() -> dict[str, int]:
    """Return deterministic raw/gzip sizes for budget tests and build receipts."""
    import gzip

    raw = PROFILE_CONNECTIONS_JS.encode("utf-8")
    return {
        "javascript_raw_bytes": len(raw),
        "javascript_gzip_bytes": len(gzip.compress(raw, compresslevel=9, mtime=0)),
    }
