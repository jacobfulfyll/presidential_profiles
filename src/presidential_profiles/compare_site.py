"""Render the accessible, mobile-first president comparison workspace.

Compare consumes the normalized Profile V3 model but deliberately projects a
much smaller page payload.  Canonical profile JSON remains the downloadable
source of truth and is fetched only when a reader requests a comparison bundle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import html
import json
import math

from .figures import GRID, INK, INK2, MUTED, SERIES, SURFACE, REPO_ROOT
from .html_safety import json_for_script
from .profiles import (
    FEATURE_SIMILARITY,
    format_speech_count,
    president_chronology,
    public_display_name,
    slug,
)
from . import profiles_site
from .site_style import FONT, PAGE_CSS


COMPARE_PAYLOAD_VERSION = "president-comparison-page-v2"
DOWNLOAD_SCHEMA_VERSION = "president-comparison-v2"
DEFAULT_PRESIDENTS = ("Abraham Lincoln", "Franklin D. Roosevelt")
SLOT_LABELS = ("A", "B", "C")


def _finite_number(value, field: str) -> float | None:
    """Return a JSON-safe finite float while preserving a real zero."""
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Compare field {field} must be finite")
    return number


def _measure_rows(shared: dict, layer: str) -> list[dict]:
    """Project the shared Profile V3 formatter rows needed by one matrix."""
    ai_missing = layer == "ai_radar" and not shared.get("ai")
    rows = []
    for row in profiles_site.profile_measure_rows(shared, layer):
        percentile = _finite_number(
            row.get("percentile"), f"{layer}.{row['key']}.percentile"
        )
        if percentile is None:
            if ai_missing:
                rank_display = "N/A · AI labels unavailable"
            elif shared["sample"]["thin_record"]:
                rank_display = "N/A · Insufficient record"
            else:
                rank_display = "N/A · Not ranked"
        else:
            rank_display = row["percentile_display"]
        rows.append({
            "key": row["key"],
            "label": row["label"],
            "unit": row["unit"],
            "absolute": _finite_number(
                row.get("absolute"), f"{layer}.{row['key']}.absolute"
            ),
            "absolute_display": row["absolute_display"],
            "percentile": percentile,
            "rank_display": rank_display,
        })
    return rows


def _agenda_records(shared: dict) -> dict:
    """Read the full canonical arrays required for deterministic shared rows."""
    ai = shared.get("ai") or {}
    domains = [{
        "name": item["name"],
        "share": _finite_number(item.get("share"), f"domain.{item['name']}.share"),
        "rel": _finite_number(item.get("rel"), f"domain.{item['name']}.rel"),
    } for item in ai.get("domain_attention", [])]
    topics = [{
        "name": item["name"],
        "domain": item.get("domain", ""),
        "share": _finite_number(item.get("share"), f"topic.{item['name']}.share"),
        "rel": _finite_number(item.get("rel"), f"topic.{item['name']}.rel"),
    } for item in ai.get("topic_attention", [])]
    legacy = [{
        "name": item["label"],
        "key": item["key"],
        # Profile V3 stores legacy shares as fractions; Compare uses percent.
        "share": round(
            100 * _finite_number(
                item.get("share"), f"legacy.{item['key']}.share"
            ),
            4,
        ),
        "rel": _finite_number(
            item.get("era_relative_difference"),
            f"legacy.{item['key']}.era_relative_difference",
        ),
    } for item in shared.get("legacy_issue_attention", [])]
    proposal_values = None
    if ai.get("proposal_values"):
        proposal_values = {
            key: _finite_number(
                ai["proposal_values"].get(key), f"proposal_values.{key}"
            )
            for key in ("proposal", "mixed", "values", "neither")
        }
    return {
        "domains": domains,
        "topics": topics,
        "legacy": legacy,
        "proposal_values": proposal_values,
        "speech_types": [{
            "name": item["name"],
            "share": _finite_number(
                item.get("value"), f"speech_type.{item['name']}.share"
            ),
        } for item in ai.get("speech_types", [])],
    }


def _neighbor_projection(shared: dict) -> dict:
    """Preserve the five governed instruments without inventing pairwise scores."""
    source = shared.get("feature_neighbors") or {}
    return {
        key: [[
            slug(item["president"]),
            _finite_number(
                item.get("similarity"), f"feature_neighbors.{key}.similarity"
            ),
        ] for item in source.get(key, [])[:3]]
        for key in FEATURE_SIMILARITY
    }


def _evidence_projection(shared: dict) -> dict:
    """Project the selected-president evidence shown by Compare."""
    ai = shared.get("ai") or {}
    cards = (shared.get("issue_evidence") or {}).get("cards", [])
    classified = shared.get("classified_invocations") or []
    legacy = (shared.get("legacy_invocations") or {}).get("invokes", [])
    invocation_source = "ai" if classified else ("legacy" if legacy else "none")
    invocations = [{
        "target": item.get("target", ""),
        "function": item.get("function", ""),
        "stance": item.get("stance", ""),
        "mentions": int(item.get("mentions", item.get("n", 0))),
    } for item in (classified or legacy)[:6]]
    return {
        "n_words": int(shared["sample"].get("n_words", 0)),
        "n_paragraphs": int(
            ai.get(
                "n_paragraphs",
                (shared.get("issue_evidence") or {}).get("n_paragraphs", 0),
            )
        ),
        "adversaries": [{
            "name": item.get("name", ""),
            "mentions": int(item.get("n", 0)),
        } for item in ai.get("adversaries", [])[:3]],
        "invocations": invocations,
        "invocation_source": invocation_source,
        "vocabulary": [
            item.get("term", "")
            for item in shared.get("distinctive_vocabulary", [])[:6]
        ],
        "signatures": [{
            "title": item.get("title", ""),
            "year": int(item["year"]) if item.get("year") is not None else None,
            "url": item.get("url", ""),
        } for item in shared.get("signature_speeches", [])[:2]],
        # Profile evidence is HTML-escaped for its server renderer.  Compare's
        # client renderer uses text escaping, so restore the text first.
        "excerpts": [{
            "issue": item.get("issue", ""),
            "quote": html.unescape(str(item.get("quote", ""))),
            "cite": html.unescape(str(item.get("cite", ""))),
        } for item in cards if item.get("quote")][:2],
    }


def _comparison_catalogs(
    public_profiles: Mapping[str, dict], order: Sequence[str]
) -> dict:
    """Build shared measure/taxonomy catalogs and validate canonical alignment."""
    if not order:
        return {
            "measures": {"corpus": [], "ai": []},
            "agenda": {"domains": [], "topics": [], "legacy": []},
            "similarity": {},
        }
    first = public_profiles[order[0]]
    measure_catalogs = {}
    for output_layer, profile_layer in (
        ("corpus", "rhetorical_radar"),
        ("ai", "ai_radar"),
    ):
        measure_catalogs[output_layer] = [
            {"key": row["key"], "label": row["label"], "unit": row["unit"]}
            for row in _measure_rows(first, profile_layer)
        ]

    agenda_source = next(
        (
            public_profiles[president]
            for president in order
            if public_profiles[president].get("ai")
        ),
        first,
    )
    source_records = _agenda_records(agenda_source)
    agenda_catalogs = {
        "domains": [
            {"name": row["name"]} for row in source_records["domains"]
        ],
        "topics": [
            {"name": row["name"], "domain": row.get("domain", "")}
            for row in source_records["topics"]
        ],
        "legacy": [
            {"name": row["name"], "key": row["key"]}
            for row in _agenda_records(first)["legacy"]
        ],
    }
    for president in order:
        records = _agenda_records(public_profiles[president])
        for layer in ("domains", "topics"):
            if not records[layer]:
                continue
            observed = [row["name"] for row in records[layer]]
            expected = [row["name"] for row in agenda_catalogs[layer]]
            if observed != expected:
                raise ValueError(
                    f"{president} {layer} rows disagree with canonical taxonomy order"
                )
        observed_legacy = [(row["key"], row["name"]) for row in records["legacy"]]
        expected_legacy = [
            (row["key"], row["name"]) for row in agenda_catalogs["legacy"]
        ]
        if observed_legacy != expected_legacy:
            raise ValueError(
                f"{president} legacy rows disagree with canonical taxonomy order"
            )
    return {
        "measures": measure_catalogs,
        "agenda": agenda_catalogs,
        "similarity": {
            key: {
                "label": spec["label"],
                "description": spec["description"],
            }
            for key, spec in FEATURE_SIMILARITY.items()
        },
    }


def compact_profile_projection(shared: dict, catalogs: dict | None = None) -> dict:
    """Project one canonical Profile V3 record onto Compare's compact contract."""
    profiles_site.validate_profile_view_model(shared)
    if catalogs is None:
        catalogs = _comparison_catalogs(
            {shared["president"]: shared}, [shared["president"]]
        )
    president = shared["president"]
    n_speeches = int(shared["sample"]["n_speeches"])
    thin = bool(shared["sample"]["thin_record"])
    measures = {}
    for output_layer, profile_layer in (
        ("corpus", "rhetorical_radar"),
        ("ai", "ai_radar"),
    ):
        rows = _measure_rows(shared, profile_layer)
        if [row["key"] for row in rows] != [
            row["key"] for row in catalogs["measures"][output_layer]
        ]:
            raise ValueError(
                f"{president} {output_layer} measures disagree with the shared catalog"
            )
        measures[output_layer] = [[
            row["absolute"], row["absolute_display"],
            row["percentile"], row["rank_display"],
        ] for row in rows]

    agenda_records = _agenda_records(shared)
    agenda = {}
    for layer in ("domains", "topics", "legacy"):
        rows = agenda_records[layer]
        catalog = catalogs["agenda"][layer]
        if not rows and layer in ("domains", "topics"):
            agenda[layer] = None
            continue
        if [row["name"] for row in rows] != [row["name"] for row in catalog]:
            raise ValueError(
                f"{president} {layer} values disagree with the shared catalog"
            )
        agenda[layer] = [[row["share"], row["rel"]] for row in rows]
    agenda["proposal_values"] = agenda_records["proposal_values"]
    agenda["speech_types"] = agenda_records["speech_types"]
    return {
        "president": president,
        "display_name": public_display_name(president),
        "slug": shared["slug"],
        "party": shared.get("party"),
        "years": {
            "first": int(shared["years"]["first"]),
            "last": int(shared["years"]["last"]),
        },
        "sample": {
            "n_speeches": n_speeches,
            "speech_count_label": format_speech_count(n_speeches),
            "thin_record": thin,
            "support_label": (
                "Insufficient record for percentile ranking"
                if thin
                else "Supported for percentile ranking"
            ),
            "ai_available": shared.get("ai") is not None,
        },
        "measures": measures,
        "agenda": agenda,
        "neighbors": _neighbor_projection(shared),
        "evidence": _evidence_projection(shared),
    }


def build_payload(
    data: dict,
    display_issues: list[str],
    *,
    profile_views: dict[str, dict] | None = None,
) -> dict:
    """Build a chronological, finite comparison-only projection.

    The compatibility wrapper still accepts raw profile inputs, but unlike the
    V1 page it never spreads a complete public profile into the page payload.
    """
    if profile_views is None:
        public_profiles = {
            president: profiles_site.public_profile_payload(
                president, data, display_issues
            )
            for president in data["scores"].index
        }
    else:
        public_profiles = {
            president: profiles_site.profile_public_payload(view)
            for president, view in profile_views.items()
        }
    return comparison_payload(
        public_profiles,
        require_complete=len(public_profiles) == 45,
    )


def comparison_payload(
    public_profiles: Mapping[str, dict], *, require_complete: bool = False
) -> dict:
    """Build the compact page contract from canonical public Profile V3 records."""
    order = president_chronology(
        public_profiles,
        require_complete=require_complete,
    )
    catalogs = _comparison_catalogs(public_profiles, order)
    payload = {
        "schema_version": COMPARE_PAYLOAD_VERSION,
        "profile_schema_version": profiles_site.PROFILE_SCHEMA_VERSION,
        "order": order,
        "catalogs": catalogs,
        "presidents": {
            president: compact_profile_projection(
                public_profiles[president], catalogs
            )
            for president in order
        },
    }
    json.dumps(payload, allow_nan=False)
    return payload


def _canonical_names(payload: Mapping[str, dict], layer: str) -> list[str]:
    """Return the full producer order for an agenda layer."""
    return [row["name"] for row in payload["catalogs"]["agenda"][layer]]


def _agenda_values(payload: Mapping[str, dict], president: str, layer: str) -> list[dict]:
    """Expand aligned numeric values against the shared agenda catalog."""
    profile = payload["presidents"][president]
    values = profile["agenda"].get(layer)
    if values is None:
        return []
    catalog = payload["catalogs"]["agenda"][layer]
    return [
        {**definition, "share": pair[0], "rel": pair[1]}
        for definition, pair in zip(catalog, values, strict=True)
    ]


def shared_agenda_rows(
    payload: Mapping[str, dict],
    selected: Sequence[str],
    layer: str,
    *,
    top_n: int,
    cap: int | None,
    positive_relative: bool = False,
    ranking_key: str = "share",
) -> list[dict]:
    """Build a deterministic union ordered by max value then taxonomy order."""
    canonical = _canonical_names(payload, layer)
    canonical_index = {name: index for index, name in enumerate(canonical)}
    by_president = {
        president: {
            row["name"]: row
            for row in _agenda_values(payload, president, layer)
        }
        for president in selected
    }
    union: set[str] = set()
    for president in selected:
        rows = list(by_president[president].values())
        if positive_relative:
            rows = [row for row in rows if (row.get("rel") or 0) > 0]
        rows.sort(key=lambda row: (
            -(row.get(ranking_key) if row.get(ranking_key) is not None else -math.inf),
            canonical_index.get(row["name"], len(canonical)),
        ))
        union.update(row["name"] for row in rows[:top_n])

    def row_max(name: str) -> float:
        values = [
            by_president[president].get(name, {}).get(ranking_key)
            for president in selected
        ]
        finite = [float(value) for value in values if value is not None]
        return max(finite) if finite else -math.inf

    names = sorted(
        union,
        key=lambda name: (
            -row_max(name), canonical_index.get(name, len(canonical)),
        ),
    )
    if cap is not None:
        names = names[:cap]
    rows = []
    for name in names:
        first = next(
            (by_president[p].get(name) for p in selected if name in by_president[p]),
            {"name": name},
        )
        rows.append({
            "name": name,
            "domain": first.get("domain", ""),
            "values": {
                president: by_president[president].get(name)
                for president in selected
            },
        })
    return rows


def normalize_selection(
    order: Sequence[str],
    slug_to_name: Mapping[str, str],
    query: Mapping[str, str | None],
) -> tuple[list[str], bool]:
    """Python mirror of the initial a/b/c URL normalization contract."""
    order = list(order)
    if len(order) < 2:
        return order, False
    defaults = [name for name in DEFAULT_PRESIDENTS if name in order]
    defaults.extend(name for name in order if name not in defaults)
    default_a, default_b = defaults[:2]
    corrected = False

    raw_a = query.get("a")
    raw_b = query.get("b")
    raw_c = query.get("c")
    requested_a = slug_to_name.get(raw_a or "")
    a = requested_a
    if a is None:
        a = default_a
        corrected = corrected or raw_a is not None

    def adjacent(anchor: str, used: set[str]) -> str:
        index = order.index(anchor)
        candidates = order[index + 1:] + list(reversed(order[:index])) + order
        return next(name for name in candidates if name not in used)

    b = slug_to_name.get(raw_b or "")
    if b is None or b == a:
        if raw_b is None and requested_a is not None:
            b = adjacent(a, {a})
        elif default_b != a:
            b = default_b
        else:
            b = adjacent(a, {a})
        corrected = corrected or raw_b is not None

    selected = [a, b]
    if raw_c not in (None, ""):
        c = slug_to_name.get(raw_c)
        if c is None or c in selected:
            c = adjacent(b, set(selected))
            corrected = True
        selected.append(c)
    return selected, corrected


def _default_selection(payload: Mapping[str, dict]) -> list[str]:
    profiles = payload["presidents"]
    names = list(payload["order"])
    preferred = [name for name in DEFAULT_PRESIDENTS if name in profiles]
    preferred.extend(name for name in names if name not in preferred)
    return preferred[: min(2, len(preferred))]


def _number(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    text = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return f"{text}.0" if "." not in text else text


def _signed_number(value: float | None, digits: int = 3) -> str:
    """Format a percentage-point difference with an explicit sign."""
    text = _number(value, digits)
    if text == "N/A" or text.startswith("-"):
        return text
    return f"+{text}"


def _slot_marker(index: int) -> str:
    slot = SLOT_LABELS[index]
    return f'<span class="slot-marker slot-{slot.lower()}" aria-hidden="true">{slot}</span>'


def _president_heading(profile: dict, index: int) -> str:
    return (
        _slot_marker(index)
        + f'<span>{html.escape(profile["display_name"])}</span>'
    )


def selector_support_cards(payload: Mapping[str, dict], selected: Sequence[str]) -> str:
    """Server-render the default support cards for a meaningful no-JS page."""
    profiles = payload["presidents"]
    cards = []
    for index, president in enumerate(selected):
        profile = profiles[president]
        years = profile["years"]
        support_class = " thin" if profile["sample"]["thin_record"] else ""
        ai_status = (
            "AI labels available"
            if profile["sample"]["ai_available"]
            else "AI labels unavailable"
        )
        cards.append(f'''<article class="support-card{support_class}">
  <img src="portraits/{html.escape(profile['slug'], quote=True)}.png" alt="Portrait of {html.escape(profile['display_name'], quote=True)}">
  <div><h3>{_president_heading(profile, index)}</h3>
  <p>{html.escape(str(profile.get('party') or 'Party unavailable'))}</p>
  <dl><div><dt>Corpus record span</dt><dd>{years['first']}–{years['last']}</dd></div>
  <div><dt>Corpus support</dt><dd>{html.escape(profile['sample']['speech_count_label'])}</dd></div></dl>
  <p class="support-status">{html.escape(profile['sample']['support_label'])} · {html.escape(ai_status)}</p>
  <p><a href="presidents/{html.escape(profile['slug'], quote=True)}.html">Open full profile</a> ·
  <a href="data/presidents/{html.escape(profile['slug'], quote=True)}.json">Public JSON</a></p></div>
</article>''')
    return "".join(cards)


def _matrix_cell(value: list, display_name: str) -> str:
    return f'''<td data-label="{html.escape(display_name, quote=True)}">
  <strong>{html.escape(value[1])}</strong>
  <span>{html.escape(value[3])}</span></td>'''


def measure_matrix(
    payload: Mapping[str, dict],
    selected: Sequence[str],
    layer: str,
    caption: str,
) -> str:
    """Render a semantic native matrix with exact values and percentile states."""
    if not selected:
        return ""
    profiles = payload["presidents"]
    rows = payload["catalogs"]["measures"][layer]
    heads = "".join(
        f'<th scope="col">{_president_heading(profiles[president], index)}</th>'
        for index, president in enumerate(selected)
    )
    body = []
    for row_index, row in enumerate(rows):
        cells = "".join(
            _matrix_cell(
                profiles[president]["measures"][layer][row_index],
                profiles[president]["display_name"],
            )
            for president in selected
        )
        body.append(f'''<tr><th scope="row"><strong>{html.escape(row['label'])}</strong>
  <span>{html.escape(row['unit'])}</span></th>{cells}</tr>''')
    return f'''<table class="comparison-matrix measure-matrix">
  <caption>{html.escape(caption)}</caption>
  <thead><tr><th scope="col">Measure and unit</th>{heads}</tr></thead>
  <tbody>{''.join(body)}</tbody>
</table>'''


def _agenda_table(
    payload: Mapping[str, dict],
    selected: Sequence[str],
    rows: Sequence[dict],
    caption: str,
    *,
    relative_label: str,
) -> str:
    profiles = payload["presidents"]
    heads = "".join(
        f'<th scope="col">{_president_heading(profiles[president], index)}</th>'
        for index, president in enumerate(selected)
    )
    body = []
    for row in rows:
        label = html.escape(row["name"])
        if row.get("domain"):
            label += f'<span>{html.escape(row["domain"])}</span>'
        cells = []
        for president in selected:
            value = row["values"].get(president)
            display = profiles[president]["display_name"]
            if value is None:
                cells.append(
                    f'<td data-label="{html.escape(display, quote=True)}"><strong>N/A</strong><span>Not recorded</span></td>'
                )
            else:
                share = value.get("share")
                relative = value.get("rel")
                share_display = (
                    f'{_number(share)}%' if share is not None else "N/A"
                )
                detail = ""
                if relative is not None:
                    suffix = f" {html.escape(relative_label)}" if relative_label else ""
                    detail = f'{_signed_number(relative)} pp{suffix}'
                elif share is None:
                    detail = "Not recorded"
                cells.append(
                    f'<td data-label="{html.escape(display, quote=True)}"><strong>{share_display}</strong>'
                    f'{f"<span>{detail}</span>" if detail else ""}</td>'
                )
        body.append(f'<tr><th scope="row">{label}</th>{"".join(cells)}</tr>')
    if not rows:
        body.append(
            f'<tr><th scope="row">Unavailable</th><td colspan="{max(1, len(selected))}">No rows are recorded for this selection.</td></tr>'
        )
    return f'''<table class="comparison-matrix agenda-matrix">
  <caption>{html.escape(caption)}</caption>
  <thead><tr><th scope="col">Agenda row</th>{heads}</tr></thead>
  <tbody>{''.join(body)}</tbody>
</table>'''


def _composition_rows(
    payload: Mapping[str, dict], selected: Sequence[str], field: str
) -> list[dict]:
    profiles = payload["presidents"]
    canonical = []
    by_president = {}
    for profile in profiles.values():
        records = profile["agenda"].get(field) or []
        names = (
            list(records)
            if isinstance(records, dict)
            else [record["name"] for record in records]
        )
        for name in names:
            if name not in canonical:
                canonical.append(name)
    for president in selected:
        records = profiles[president]["agenda"].get(field)
        if isinstance(records, dict):
            by_president[president] = records
        else:
            by_president[president] = {
                record["name"]: record["share"] for record in (records or [])
            }
    names = sorted(
        canonical,
        key=lambda name: (
            -max(
                [
                    float(by_president[p].get(name))
                    for p in selected
                    if by_president[p].get(name) is not None
                ]
                or [-math.inf]
            ),
            canonical.index(name),
        ),
    )
    return [{
        "name": name.replace("_", " ").title(),
        "values": {
            president: (
                None
                if by_president[president].get(name) is None
                else {"share": by_president[president][name], "rel": None}
            )
            for president in selected
        },
    } for name in names]


def agenda_sections(payload: Mapping[str, dict], selected: Sequence[str]) -> str:
    """Render shared AI, legacy, and composition rows for the default pair."""
    domains = shared_agenda_rows(
        payload, selected, "domains", top_n=4, cap=12
    )
    topics = shared_agenda_rows(
        payload, selected, "topics", top_n=3, cap=9
    )
    legacy = shared_agenda_rows(
        payload,
        selected,
        "legacy",
        top_n=4,
        cap=12,
        positive_relative=True,
        ranking_key="rel",
    )
    extended_domains = shared_agenda_rows(
        payload, selected, "domains", top_n=6, cap=None
    )
    extended_topics = shared_agenda_rows(
        payload, selected, "topics", top_n=6, cap=None
    )
    legacy_all = shared_agenda_rows(
        payload,
        selected,
        "legacy",
        top_n=max(1, len(_canonical_names(payload, "legacy"))),
        cap=None,
    )
    proposal = _composition_rows(payload, selected, "proposal_values")
    speech_types = _composition_rows(payload, selected, "speech_types")
    return "".join((
        '<h3>Broad AI-labeled domains</h3>',
        _agenda_table(
            payload, selected, domains,
            "Shared broad-domain rows: each selected president’s top four",
            relative_label="vs contemporaries",
        ),
        '<h3>Fine AI-labeled topics</h3>',
        _agenda_table(
            payload, selected, topics,
            "Shared fine-topic rows: each selected president’s top three",
            relative_label="vs contemporaries",
        ),
        '''<details class="sub-disclosure"><summary>Show extended AI agenda rows</summary>
<div class="disclosure-body">''',
        _agenda_table(
            payload, selected, extended_domains,
            "Extended broad-domain union: each selected president’s top six",
            relative_label="vs contemporaries",
        ),
        _agenda_table(
            payload, selected, extended_topics,
            "Extended fine-topic union: each selected president’s top six",
            relative_label="vs contemporaries",
        ),
        '</div></details>',
        '<h3>Legacy issue lens</h3>',
        '<p class="section-note">Deterministic CorEx issue rows remain separate from AI-labeled topics.</p>',
        _agenda_table(
            payload, selected, legacy,
            "Positive era-relative legacy issues: each selected president’s top four",
            relative_label="vs era",
        ),
        '<details class="sub-disclosure"><summary>Show all legacy issues</summary><div class="disclosure-body">',
        _agenda_table(
            payload, selected, legacy_all,
            "All canonical legacy issues",
            relative_label="vs era",
        ),
        '</div></details>',
        '<h3>Proposal and values composition</h3>',
        _agenda_table(
            payload, selected, proposal,
            "Mutually exclusive AI-labeled paragraph composition",
            relative_label="",
        ),
        '<h3>Speech-type composition</h3>',
        _agenda_table(
            payload, selected, speech_types,
            "Leading assigned speech types in each corpus record",
            relative_label="",
        ),
    ))


def _bar_width(value: float | int | None, maximum: float | int) -> float:
    """Return a bounded visual width while preserving a real zero."""
    if value is None or maximum <= 0:
        return 0.0
    width = 100 * float(value) / float(maximum)
    return min(100.0, max(0.0, width))


def _neighbor_tile(profile: dict, item: list, rank: int) -> str:
    score = item[1]
    width = min(100.0, max(0.0, 100 * float(score or 0)))
    return f'''<li><a class="neighbor-tile" href="presidents/{html.escape(profile['slug'], quote=True)}.html">
  <span class="neighbor-rank" aria-label="Rank {rank}">{rank}</span>
  <img src="portraits/{html.escape(profile['slug'], quote=True)}.png" alt="">
  <span class="neighbor-name">{html.escape(profile['display_name'])}</span>
  <strong>{_number(score, 3)}</strong>
  <span class="neighbor-meter" aria-hidden="true"><i style="width:{width:.2f}%"></i></span>
</a></li>'''


def similarity_sections(payload: Mapping[str, dict], selected: Sequence[str]) -> str:
    """Render five independent nearest-neighbor lenses as visual lanes."""
    profiles = payload["presidents"]
    by_slug = {profile["slug"]: profile for profile in profiles.values()}
    cards = []
    for lens_index, (key, spec) in enumerate(
        payload["catalogs"]["similarity"].items(), start=1
    ):
        rows = []
        for index, president in enumerate(selected):
            profile = profiles[president]
            tiles = "".join(
                _neighbor_tile(by_slug[item[0]], item, rank)
                for rank, item in enumerate(
                    profile["neighbors"].get(key, []), start=1
                )
                if item[0] in by_slug
            )
            if not tiles:
                tiles = '<li class="empty-state">Not available</li>'
            rows.append(f'''<div class="neighbor-row">
  <h4>{_president_heading(profile, index)}</h4>
  <ol class="neighbor-strip">{tiles}</ol>
</div>''')
        cards.append(f'''<article class="similarity-panel">
  <header><p class="visual-kicker">Lens {lens_index} of 5</p><h3>{html.escape(spec['label'])}</h3>
  <details class="measure-note"><summary>What this lens uses</summary><p>{html.escape(spec['description'])}</p></details></header>
  <div class="neighbor-rows">{''.join(rows)}</div>
</article>''')
    return "".join(cards)


def _signal_row(
    label: str,
    value: int,
    maximum: int,
    *,
    detail: str = "",
) -> str:
    width = _bar_width(value, maximum)
    detail_html = f'<span>{html.escape(detail)}</span>' if detail else ""
    return f'''<div class="signal-row">
  <div class="signal-label"><strong>{html.escape(label)}</strong>{detail_html}</div>
  <div class="signal-meter" aria-hidden="true"><i style="width:{width:.2f}%"></i></div>
  <b>{value:,}</b>
</div>'''


def _footprint_visual(
    payload: Mapping[str, dict], selected: Sequence[str]
) -> str:
    profiles = payload["presidents"]
    metrics = (
        ("Speeches", lambda profile: int(profile["sample"]["n_speeches"])),
        ("Words", lambda profile: int(profile["evidence"]["n_words"])),
        (
            "Paragraphs",
            lambda profile: int(profile["evidence"]["n_paragraphs"]),
        ),
    )
    groups = []
    for label, accessor in metrics:
        maximum = max((accessor(profiles[name]) for name in selected), default=0)
        rows = []
        for index, president in enumerate(selected):
            profile = profiles[president]
            value = accessor(profile)
            rows.append(f'''<div class="footprint-row">
  <span class="footprint-owner">{_president_heading(profile, index)}</span>
  <span class="footprint-meter" aria-hidden="true"><i style="width:{_bar_width(value, maximum):.2f}%"></i></span>
  <strong>{value:,}</strong>
</div>''')
        groups.append(
            f'<section class="footprint-group"><h4>{label}</h4>{"".join(rows)}</section>'
        )
    return f'''<section class="footprint-visual" aria-labelledby="footprint-title">
  <header><p class="visual-kicker">Selected-record scale</p><h3 id="footprint-title">Corpus footprint</h3>
  <p>Each row uses its own shared maximum; exact counts remain printed.</p></header>
  <div class="footprint-groups">{''.join(groups)}</div>
</section>'''


def evidence_sections(payload: Mapping[str, dict], selected: Sequence[str]) -> str:
    """Render a compact visual evidence dashboard for selected presidents."""
    profiles = payload["presidents"]
    adversary_max = max(
        (
            int(item["mentions"])
            for president in selected
            for item in profiles[president]["evidence"]["adversaries"]
        ),
        default=0,
    )
    invocation_max = max(
        (
            int(item["mentions"])
            for president in selected
            for item in profiles[president]["evidence"]["invocations"]
        ),
        default=0,
    )
    cards = []
    for index, president in enumerate(selected):
        profile = profiles[president]
        evidence = profile["evidence"]
        adversaries = "".join(
            _signal_row(
                item["name"], int(item["mentions"]), adversary_max
            )
            for item in evidence["adversaries"]
        ) or '<p class="empty-state">No repeated adversarial entity.</p>'
        invocation_rows = []
        for item in evidence["invocations"]:
            descriptors = []
            if item["function"]:
                descriptors.append(f'Function: {item["function"]}')
            if item["stance"]:
                descriptors.append(f'Stance: {item["stance"]}')
            invocation_rows.append(
                _signal_row(
                    item["target"],
                    int(item["mentions"]),
                    invocation_max,
                    detail=" · ".join(descriptors),
                )
            )
        invocations = "".join(invocation_rows) or '<p class="empty-state">No presidential invocation.</p>'
        if evidence["invocation_source"] == "ai":
            invocation_source = '<span class="source-tag exploratory">AI-labeled · exploratory</span>'
        elif evidence["invocation_source"] == "legacy":
            invocation_source = '<span class="source-tag legacy">Legacy invocation counts</span>'
        else:
            invocation_source = '<span class="source-tag legacy">Invocation source unavailable</span>'
        vocabulary = "".join(
            f'<li>{html.escape(term)}</li>' for term in evidence["vocabulary"]
        ) or '<li>None recorded</li>'
        signatures = "".join(
            f'''<a class="speech-tile" href="{html.escape(item['url'], quote=True)}" target="_blank" rel="noopener">
  <span>{html.escape(item['title'])}</span>{f'<b>{item["year"]}</b>' if item['year'] is not None else ''}</a>'''
            for item in evidence["signatures"]
        ) or '<span class="empty-state">No signature speech recorded.</span>'
        excerpts = "".join(
            f'<figure><blockquote>{html.escape(item["quote"])}</blockquote><figcaption>{html.escape(item["issue"])} · {html.escape(item["cite"])}</figcaption></figure>'
            for item in evidence["excerpts"]
        ) or '<p>No excerpt selected for this record.</p>'
        cards.append(f'''<article class="evidence-card visual-evidence-card">
  <header><img src="portraits/{html.escape(profile['slug'], quote=True)}.png" alt=""><h3>{_president_heading(profile, index)}</h3></header>
  <div class="signal-grid">
    <section><div class="signal-heading"><h4>Adversarial entities</h4><span class="source-tag exploratory">AI · exploratory</span></div>{adversaries}</section>
    <section><div class="signal-heading"><h4>Presidential invocations</h4>{invocation_source}</div>{invocations}</section>
  </div>
  <section class="compact-evidence"><div><h4>Distinctive vocabulary</h4><ul class="term-list">{vocabulary}</ul></div>
  <div><h4>Signature speeches</h4><div class="speech-tiles">{signatures}</div></div></section>
  <details class="excerpt-disclosure"><summary>Read source excerpts ({len(evidence['excerpts'])})</summary><div>{excerpts}</div></details>
  <nav class="evidence-actions" aria-label="{html.escape(profile['display_name'], quote=True)} evidence links">
    <a href="presidents/{html.escape(profile['slug'], quote=True)}.html">Full evidence</a>
    <a href="data/presidents/{html.escape(profile['slug'], quote=True)}.json">Canonical JSON</a>
  </nav>
</article>''')
    return _footprint_visual(payload, selected) + (
        f'<div class="evidence-grid visual-evidence-grid">{"".join(cards)}</div>'
    )


def _select_options(payload: Mapping[str, dict], chosen: str | None) -> str:
    return "".join(
        f'<option value="{html.escape(profile["slug"], quote=True)}"'
        f'{" selected" if president == chosen else ""}>{html.escape(profile["display_name"])}</option>'
        for president, profile in payload["presidents"].items()
    )


def selector_fieldset(payload: Mapping[str, dict], selected: Sequence[str]) -> str:
    """Render labeled A/B/optional-C selectors and controls."""
    slots = []
    for index, label in enumerate(("President A", "President B", "Optional President C")):
        chosen = selected[index] if index < len(selected) else None
        slot = SLOT_LABELS[index].lower()
        disabled = " disabled" if index == 2 and chosen is None else ""
        empty_option = (
            '<option value="" selected>Not selected</option>'
            if index == 2 and chosen is None else ""
        )
        slots.append(f'''<div class="selector-field selector-{slot}">
  <label for="president-{slot}">{label}</label>
  <select id="president-{slot}" data-slot="{index}"{disabled}>
    {empty_option}{_select_options(payload, chosen)}</select>
  <span class="selector-hint">{'Use “Add third president” to activate.' if index == 2 and chosen is None else 'Selected president ' + SLOT_LABELS[index]}</span>
</div>''')
    return f'''<fieldset class="selector-fieldset"><legend>Choose two or three distinct presidents</legend>
  <div class="selector-grid">{''.join(slots)}</div>
  <div class="selection-controls" aria-label="Comparison controls">
    <button type="button" id="swap-presidents">Swap A/B</button>
    <button type="button" id="toggle-third">Add third president</button>
    <button type="button" id="copy-link">Copy link</button>
    <button type="button" id="reset-comparison">Reset</button>
  </div>
  <p id="selection-status" class="live-status" role="status" aria-live="polite"></p>
</fieldset>'''


def _page_template() -> str:
    """Return the static shell; sentinels are replaced without brace escaping."""
    return r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Compare presidents - Presidential Profiles</title>
<script src="assets/plotly-3.0.1.min.js"></script>
<style>
__PAGE_CSS__
  header { padding-bottom: 18px; }
  .crumbs { margin-bottom:18px; font-size:.88rem; }
  .crumbs a { color:var(--ink2); }
  .eyebrow { color:#7a3d00; font-size:.76rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
  .selector-fieldset { border:1px solid var(--border); border-radius:16px; background:var(--surface); padding:18px; margin-top:24px; }
  .selector-fieldset legend { font-weight:750; padding:0 7px; }
  .selector-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }
  .selector-field label { display:block; font-weight:750; margin-bottom:5px; }
  .selector-field select { width:100%; min-height:44px; border:1px solid #78736b; border-radius:8px; background:#fff; color:var(--ink); padding:8px 10px; font:inherit; }
  .selector-field select:disabled { background:#efefeb; color:#706d67; }
  .selector-hint { display:block; color:var(--muted); font-size:.76rem; margin-top:4px; }
  .selection-controls { display:flex; flex-wrap:wrap; gap:8px; margin-top:16px; }
  button { min-height:42px; border:1px solid #706d67; border-radius:8px; background:#fff; color:var(--ink); padding:8px 13px; font:inherit; font-weight:700; cursor:pointer; }
  button:hover { background:#f2f0eb; }
  button:disabled { cursor:wait; opacity:.65; }
  .live-status { min-height:1.5em; color:#5c4515; font-size:.86rem; margin-top:10px; }
  .on-page-nav { position:sticky; top:var(--global-nav-height,0px); z-index:9; display:flex; gap:5px; overflow-x:auto; background:rgba(249,249,247,.96); border-block:1px solid var(--grid); padding:9px 0; margin-bottom:22px; }
  .on-page-nav a { flex:0 0 auto; color:var(--ink2); font-size:.82rem; font-weight:750; padding:5px 9px; text-decoration:none; border-radius:999px; }
  .on-page-nav a:hover { background:#e9e6df; }
  .workspace-section { scroll-margin-top:calc(var(--global-nav-height,0px) + 70px); margin-top:36px; }
  .workspace-section > h2 { font-size:1.42rem; }
  .section-intro,.section-note { color:var(--ink2); max-width:48rem; margin:7px 0 16px; }
  .support-grid,.evidence-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(270px,1fr)); gap:14px; }
  .support-card,.evidence-card { background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:16px; min-width:0; }
  .support-card { display:grid; grid-template-columns:72px 1fr; gap:14px; }
  .support-card.thin { border-style:dashed; }
  .support-card img { width:72px; height:72px; border-radius:50%; object-fit:cover; border:2px solid #56514b; }
  .support-card h3,.evidence-card h3 { display:flex; align-items:center; gap:8px; font-size:1.02rem; }
  .support-card p { color:var(--ink2); font-size:.84rem; margin-top:5px; }
  .support-card dl { display:grid; gap:4px; margin-top:9px; }
  .support-card dl div { display:flex; flex-wrap:wrap; justify-content:space-between; gap:8px; border-top:1px solid var(--grid); padding-top:4px; }
  .support-card dt { color:var(--muted); font-size:.75rem; }
  .support-card dd { font-weight:700; font-size:.82rem; }
  .support-status { font-weight:700; }
  .slot-marker { display:inline-grid; place-items:center; flex:0 0 25px; width:25px; height:25px; border:2px solid currentColor; color:#31302d; font-size:.75rem; font-weight:900; line-height:1; }
  .slot-a { border-radius:50%; }
  .slot-b { border-style:dashed; border-radius:4px; }
  .slot-c { border-style:double; border-width:4px; border-radius:0; }
  .radar-switcher { margin-top:16px; }
  .radar-tabs { display:flex; width:100%; border:1px solid var(--border); border-bottom:0; border-radius:14px 14px 0 0; background:#efede8; overflow:hidden; }
  .radar-tab { flex:1 1 50%; min-height:48px; border:0; border-radius:0; background:transparent; color:var(--ink2); padding:9px 16px; }
  .radar-tab + .radar-tab { border-left:1px solid var(--border); }
  .radar-tab:hover { background:#e6e2da; }
  .radar-tab[aria-selected="true"] { background:var(--surface); color:var(--ink); box-shadow:inset 0 3px 0 #8d4a22; }
  .radar-tab:focus-visible { position:relative; z-index:1; outline:3px solid #255f91; outline-offset:-4px; }
  .radar-stage { min-width:0; }
  .radar-panel[hidden] { display:none!important; }
  .radar-card { min-width:0; background:var(--surface); border:1px solid var(--border); border-radius:14px; overflow:hidden; }
  .radar-stage .radar-card { border-radius:0 0 14px 14px; }
  .radar-card > header { padding:16px 18px 0; }
  .radar-card h3 { margin:2px 0 0; font-size:1.05rem; }
  .visual-kicker { color:#7a3d00; font-size:.68rem; font-weight:850; letter-spacing:.08em; text-transform:uppercase; margin:0; }
  .radar-chart { width:100%; height:430px; }
  .radar-card .chart-note { color:var(--muted); font-size:.75rem; margin:0; padding:0 18px 15px; }
  .exact-data { border:1px solid var(--border); border-radius:10px; background:var(--surface); margin-top:14px; }
  .exact-data > summary { cursor:pointer; font-weight:750; padding:13px 15px; }
  .exact-data > div { border-top:1px solid var(--grid); padding:4px 14px 14px; }
  .matrix-stack { display:grid; gap:20px; }
  .comparison-matrix { width:100%; border-collapse:separate; border-spacing:0; background:var(--surface); border:1px solid var(--border); border-radius:12px; overflow:hidden; margin:12px 0 22px; table-layout:fixed; }
  .comparison-matrix caption { text-align:left; font-weight:750; color:var(--ink); padding:0 0 8px; caption-side:top; }
  .comparison-matrix th,.comparison-matrix td { border-top:1px solid var(--grid); padding:10px 12px; text-align:left; vertical-align:top; overflow-wrap:anywhere; }
  .comparison-matrix thead th { border-top:0; color:var(--ink2); font-size:.78rem; }
  .comparison-matrix thead th:not(:first-child) { display:table-cell; }
  .comparison-matrix thead th > span:not(.slot-marker) { vertical-align:middle; }
  .comparison-matrix tbody th { width:27%; font-size:.87rem; }
  .comparison-matrix tbody th span,.comparison-matrix td span { display:block; color:var(--muted); font-size:.75rem; font-weight:400; margin-top:2px; }
  .comparison-matrix td strong { font-variant-numeric:tabular-nums; font-size:.9rem; }
  .agenda-matrix tbody th span { margin-top:3px; }
  .workspace-section h3 { font-size:1.02rem; margin:25px 0 4px; }
  .sub-disclosure,.method-disclosure { border:1px solid var(--border); border-radius:10px; background:var(--surface); margin:12px 0 22px; }
  .sub-disclosure > summary,.method-disclosure > summary { cursor:pointer; font-weight:750; padding:13px 15px; }
  .disclosure-body { border-top:1px solid var(--grid); padding:4px 14px 14px; }
  .method-disclosure .disclosure-body p,.method-disclosure .disclosure-body li { color:var(--ink2); margin:8px 0; }
  .similarity-board { display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:14px; }
  .similarity-panel { min-width:0; background:var(--surface); border:1px solid var(--border); border-radius:14px; overflow:hidden; }
  .similarity-panel > header { padding:15px 16px 11px; border-bottom:1px solid var(--grid); }
  .similarity-panel h3 { margin:2px 0 0; }
  .measure-note { margin-top:7px; color:var(--muted); font-size:.75rem; }
  .measure-note summary { cursor:pointer; font-weight:750; }
  .measure-note p { margin:5px 0 0; color:var(--ink2); }
  .neighbor-row { padding:13px 15px; }
  .neighbor-row + .neighbor-row { border-top:1px solid var(--grid); }
  .neighbor-row h4 { display:flex; align-items:center; gap:7px; margin:0 0 9px; font-size:.82rem; }
  .neighbor-row .slot-marker { width:21px; height:21px; flex-basis:21px; font-size:.65rem; }
  .neighbor-strip { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:7px; padding:0; margin:0; list-style:none; }
  .neighbor-tile { position:relative; display:grid; grid-template-columns:24px 1fr auto; grid-template-rows:auto auto; gap:2px 6px; min-height:52px; color:var(--ink); text-decoration:none; background:var(--page); border:1px solid var(--grid); border-radius:9px; padding:7px; overflow:hidden; }
  .neighbor-tile:hover { border-color:#79756e; }
  .neighbor-tile img { width:24px; height:24px; border-radius:50%; object-fit:cover; grid-row:1; }
  .neighbor-rank { position:absolute; top:1px; left:1px; display:grid; place-items:center; width:14px; height:14px; border-radius:50%; background:var(--ink); color:#fff; font-size:.58rem; font-weight:900; z-index:1; }
  .neighbor-name { align-self:center; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:.7rem; font-weight:750; }
  .neighbor-tile strong { align-self:center; font-size:.65rem; font-variant-numeric:tabular-nums; }
  .neighbor-meter { grid-column:1/-1; height:4px; border-radius:999px; background:#dddad3; overflow:hidden; }
  .neighbor-meter i { display:block; height:100%; background:#315c88; border-radius:inherit; }
  .empty-state { color:var(--muted); font-size:.78rem; margin:7px 0; }
  .footprint-visual { display:grid; grid-template-columns:minmax(180px,.55fr) minmax(0,1.45fr); gap:18px; background:#162431; color:#fff; border-radius:14px; padding:18px; margin-bottom:14px; }
  .footprint-visual header p:last-child { color:#cdd7df; font-size:.76rem; margin-top:7px; }
  .footprint-visual .visual-kicker { color:#f1be5b; }
  .footprint-visual h3 { margin:2px 0 0; }
  .footprint-groups { display:grid; gap:11px; }
  .footprint-group h4 { color:#cdd7df; font-size:.7rem; letter-spacing:.07em; text-transform:uppercase; margin:0 0 4px; }
  .footprint-row { display:grid; grid-template-columns:minmax(110px,.75fr) minmax(90px,1fr) auto; align-items:center; gap:8px; min-height:26px; }
  .footprint-owner { display:flex; align-items:center; gap:6px; min-width:0; font-size:.72rem; font-weight:750; }
  .footprint-owner > span:last-child { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .footprint-owner .slot-marker { width:19px; height:19px; flex-basis:19px; color:#fff; font-size:.58rem; }
  .footprint-meter { height:7px; border-radius:999px; background:#43515d; overflow:hidden; }
  .footprint-meter i { display:block; height:100%; background:#f1be5b; border-radius:inherit; }
  .footprint-row > strong { font-size:.72rem; font-variant-numeric:tabular-nums; }
  .visual-evidence-grid { grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); }
  .visual-evidence-card > header { display:flex; align-items:center; gap:9px; border-bottom:1px solid var(--grid); padding-bottom:11px; }
  .visual-evidence-card > header img { width:40px; height:40px; border-radius:50%; object-fit:cover; }
  .visual-evidence-card > header h3 { margin:0; }
  .signal-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin-top:13px; }
  .signal-heading { min-height:48px; }
  .signal-heading h4,.compact-evidence h4 { font-size:.8rem; margin:0 0 5px; }
  .signal-heading .source-tag { margin-top:0; }
  .signal-row { display:grid; grid-template-columns:minmax(0,1fr) auto; gap:3px 7px; margin-top:8px; }
  .signal-label { min-width:0; }
  .signal-label strong { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:.72rem; }
  .signal-label span { display:block; color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:.62rem; }
  .signal-meter { grid-column:1; height:5px; background:#dfdcd5; border-radius:999px; overflow:hidden; }
  .signal-meter i { display:block; height:100%; background:#a24a3a; border-radius:inherit; }
  .signal-row > b { grid-column:2; grid-row:1/3; align-self:center; color:var(--ink2); font-size:.68rem; font-variant-numeric:tabular-nums; }
  .compact-evidence { display:grid; grid-template-columns:1fr 1fr; gap:14px; border-top:1px solid var(--grid); margin-top:14px; padding-top:13px; }
  .compact-evidence > div,.speech-tiles,.speech-tile { min-width:0; }
  .evidence-card ul { color:var(--ink2); font-size:.85rem; }
  .term-list { display:flex; flex-wrap:wrap; gap:6px; padding:0!important; list-style:none; }
  .term-list li { border:1px solid var(--border); border-radius:999px; padding:3px 9px; background:var(--page); }
  .source-tag { display:inline-block; border:1px solid currentColor; border-radius:999px; padding:2px 8px; margin-top:14px; font-size:.7rem; font-weight:800; letter-spacing:.03em; }
  .source-tag.exploratory { color:#704d0c; background:#fff8de; }
  .source-tag.corpus { color:#315c45; background:#edf8f1; }
  .source-tag.legacy { color:#604289; background:#f5effc; }
  .speech-tiles { display:grid; gap:5px; }
  .speech-tile { display:flex; justify-content:space-between; gap:8px; border:1px solid var(--grid); border-radius:8px; color:var(--ink2); padding:7px 8px; text-decoration:none; font-size:.7rem; }
  .speech-tile span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .speech-tile b { color:var(--muted); }
  .excerpt-disclosure { border-top:1px solid var(--grid); margin-top:13px; padding-top:11px; }
  .excerpt-disclosure summary { cursor:pointer; font-size:.78rem; font-weight:750; }
  .evidence-actions { display:flex; gap:7px; margin-top:12px; }
  .evidence-actions a { flex:1; border:1px solid var(--border); border-radius:8px; color:var(--ink); padding:7px 8px; text-align:center; text-decoration:none; font-size:.72rem; font-weight:750; }
  figure { border-left:3px solid #8c877e; padding-left:11px; margin:10px 0; }
  blockquote { color:var(--ink2); font-size:.86rem; }
  figcaption { color:var(--muted); font-size:.73rem; margin-top:4px; }
  .limitation { border:1px solid #b68a32; border-radius:12px; background:#fff8de; color:#554112; padding:13px 15px; margin:12px 0 18px; }
  .download-panel { border-top:1px solid var(--grid); margin-top:22px; padding-top:18px; }
  .download-panel p { color:var(--ink2); margin:7px 0; }
  #download-status { min-height:1.4em; font-size:.84rem; }
  @media (max-width:760px) {
    .selector-grid { grid-template-columns:1fr; }
    .on-page-nav { margin-inline:-20px; padding-inline:20px; }
    .similarity-board { grid-template-columns:1fr; }
    .radar-chart { height:390px; }
    .footprint-visual { grid-template-columns:1fr; }
    .comparison-matrix { display:block; border:0; background:transparent; table-layout:auto; }
    .comparison-matrix caption { display:block; }
    .comparison-matrix thead { position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0; }
    .comparison-matrix tbody { display:grid; gap:10px; }
    .comparison-matrix tr { display:grid; grid-template-columns:1fr; border:1px solid var(--border); border-radius:10px; background:var(--surface); overflow:hidden; }
    .comparison-matrix tbody th { width:auto; border-top:0; background:#efede8; padding:10px 12px; }
    .comparison-matrix td { display:grid; grid-template-columns:minmax(95px,38%) 1fr; gap:10px; border-top:1px solid var(--grid); padding:10px 12px; }
    .comparison-matrix td::before { content:attr(data-label); font-size:.75rem; font-weight:800; color:var(--ink2); }
    .comparison-matrix td strong,.comparison-matrix td span { grid-column:2; }
    .comparison-matrix td strong { grid-row:1; }
    .comparison-matrix td span { grid-row:2; }
  }
  @media (max-width:420px) {
    header { padding-top:32px; }
    header h1 { font-size:1.7rem; }
    .support-card { grid-template-columns:56px 1fr; padding:13px; }
    .support-card img { width:56px; height:56px; }
    .selection-controls button { flex:1 1 calc(50% - 8px); }
    .radar-tab { min-height:52px; padding-inline:9px; font-size:.78rem; }
    .radar-chart { height:360px; }
    .neighbor-strip { grid-template-columns:1fr; }
    .neighbor-tile { grid-template-columns:24px 1fr auto; }
    .signal-grid,.compact-evidence { grid-template-columns:1fr; }
    .footprint-row { grid-template-columns:minmax(95px,.8fr) minmax(60px,1fr) auto; }
  }
  @media (prefers-reduced-motion:reduce) { *,*::before,*::after { scroll-behavior:auto!important; transition-duration:.01ms!important; } }
</style>
</head>
<body>
<header>
  <p class="crumbs"><a href="index.html">← Story</a> · <a href="presidents/index.html">All presidents</a> · <a href="methodology.html">Methodology</a></p>
  <p class="eyebrow">Profile V3 comparison workspace</p>
  <h1>Compare presidents</h1>
  <p class="sub">Overlay two or three presidential speech records, then move from visual fingerprints to the exact evidence underneath. Values describe this curated corpus—not governing performance, public opinion, or a contest with a winner.</p>
  <noscript><p class="limitation">The complete Lincoln–Roosevelt default comparison is shown below. Enable JavaScript only to change presidents, use browser history, copy a link, or prepare a download.</p></noscript>
  __SELECTOR_FIELDSET__
</header>
<main>
  <nav class="on-page-nav" aria-label="On this page">
    <a href="#overview">Overview</a><a href="#rhetoric">Rhetoric</a><a href="#agenda">Agenda</a><a href="#similarity">Nearest-neighbor context</a><a href="#evidence">Evidence &amp; data</a>
  </nav>
  <section class="workspace-section" id="overview">
    <h2>Overview</h2><p class="section-intro">Corpus years are the span of speeches in this archive, not presidential tenure. Lettered and shaped markers keep each column identifiable without color.</p>
    <div class="support-grid" id="support-cards">__SUPPORT_CARDS__</div>
    <details class="method-disclosure" id="method"><summary>How this comparison is measured</summary><div class="disclosure-body">
      <p>Corpus-derived rhetoric uses eight named lexical measures. AI-labeled rhetoric uses six separately declared paragraph measures. Each cell prints the absolute value and unit; supported records also show rank among presidents represented by at least five corpus speeches.</p>
      <p>Agenda tables union each selected president’s leading rows from the full canonical arrays. Rows are ordered by the largest selected value, then by the producer’s taxonomy order. A recorded zero remains zero; an absent record is shown as unavailable.</p>
      <p>AI topics, broad domains, legacy CorEx issues, and the five nearest-neighbor instruments remain separate because they measure different things. <a href="methodology.html">Read the full methodology and reliability audit.</a></p>
    </div></details>
  </section>
  <section class="workspace-section" id="rhetoric">
    <h2>Rhetoric</h2><p class="section-intro">Choose one percentile fingerprint. Both occupy the same graph stage; farther from the center means a higher position among eligible presidents—not a better score.</p>
    <div class="radar-switcher">
      <div class="radar-tabs" role="tablist" aria-label="Rhetoric radar view">
        <button class="radar-tab" id="radar-tab-corpus" type="button" role="tab" aria-selected="true" aria-controls="radar-panel-corpus">Corpus-derived</button>
        <button class="radar-tab" id="radar-tab-ai" type="button" role="tab" aria-selected="false" aria-controls="radar-panel-ai" tabindex="-1">AI-labeled</button>
      </div>
      <div class="radar-stage">
        <article class="radar-card radar-panel" id="radar-panel-corpus" role="tabpanel" aria-labelledby="radar-tab-corpus"><header><p class="visual-kicker">Corpus-derived</p><h3>Lexical fingerprint</h3></header>
          <div class="radar-chart" id="corpus-radar" role="img" aria-label="Corpus-derived rhetorical percentile radar chart"><noscript><p>Use the exact-value tables below.</p></noscript></div>
          <p class="chart-note">Eight named lexical measures · eligible-president percentiles</p></article>
        <article class="radar-card radar-panel" id="radar-panel-ai" role="tabpanel" aria-labelledby="radar-tab-ai" hidden><header><p class="visual-kicker">AI-labeled · exploratory</p><h3>Paragraph-signal fingerprint</h3></header>
          <div class="radar-chart" id="ai-radar" role="img" aria-label="AI-labeled paragraph-signal percentile radar chart"><noscript><p>Use the exact-value tables below.</p></noscript></div>
          <p class="chart-note">Six separately declared signals · eligible-president percentiles</p></article>
      </div>
    </div>
    <details class="exact-data"><summary>Exact measures, units, and percentile states</summary><div class="matrix-stack" id="rhetoric-matrices">__RHETORIC_MATRICES__</div></details>
  </section>
  <section class="workspace-section" id="agenda">
    <h2>Agenda</h2><p class="section-intro">Shared rows make differences legible without hiding a selected president’s leading subjects. AI domains, AI fine topics, and legacy issues use visibly separate tables.</p>
    <div id="agenda-content">__AGENDA_CONTENT__</div>
  </section>
  <section class="workspace-section" id="similarity">
    <h2>Nearest-neighbor context</h2><p class="section-intro">Five independent lenses; three nearest records per president. Bar length is cosine similarity from 0 to 1. No composite score is synthesized.</p>
    <div class="similarity-board" id="similarity-content">__SIMILARITY_CONTENT__</div>
  </section>
  <section class="workspace-section" id="evidence">
    <h2>Evidence &amp; data</h2>
    <p class="limitation"><strong>Population note:</strong> profile evidence follows document ownership. The separate Conflict treatment uses the completed speaker-attribution audit.</p>
    <div id="evidence-content">__EVIDENCE_CONTENT__</div>
    <div class="download-panel"><h3>Download selected comparison</h3>
      <p>One bundle, in visible A/B/C order, with canonical Profile V3 source links.</p>
      <button type="button" id="download-comparison" aria-describedby="download-status">Download selected comparison</button>
      <p id="download-status" role="status" aria-live="polite"></p>
    </div>
  </section>
</main>
<footer><p>Source corpus: <a href="https://data.millercenter.org">Miller Center of Public Affairs, University of Virginia</a>. AI-derived labels are exploratory and retain their source-status labels.</p></footer>
<script>
const PAGE = __PAYLOAD__;
const P = PAGE.presidents;
const ORDER = PAGE.order;
const FEATURE_META = PAGE.catalogs.similarity;
const DEFAULT_NAMES = __DEFAULT_NAMES__;
const DOWNLOAD_SCHEMA_VERSION = "president-comparison-v2";
const SLOT_LABELS = ["A", "B", "C"];
const CHART_COLORS = __CHART_COLORS__;
const CHART_DASHES = ["solid", "dash", "dot"];
const CHART_SYMBOLS = ["circle", "square", "diamond"];
const CHART_FONT = __CHART_FONT__;
const CHART_CHROME = __CHART_CHROME__;
const bySlug = Object.fromEntries(ORDER.map(name => [P[name].slug, name]));
const escapeHTML = value => String(value ?? "").replace(/[&<>"']/g, char => ({
  "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"
}[char]));
const formatNumber = (value, digits=3) => {
  if (!Number.isFinite(value)) return "N/A";
  const text = Number(value).toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
  return text.includes(".") ? text : text + ".0";
};
const formatSignedNumber = (value, digits=3) => {
  const text = formatNumber(value, digits);
  return text === "N/A" || text.startsWith("-") ? text : `+${text}`;
};
const markerHTML = index => `<span class="slot-marker slot-${SLOT_LABELS[index].toLowerCase()}" aria-hidden="true">${SLOT_LABELS[index]}</span>`;
const headingHTML = (name, index) => markerHTML(index) + `<span>${escapeHTML(P[name].display_name)}</span>`;
let selected = [];

function adjacentDistinct(anchor, used) {
  const index = ORDER.indexOf(anchor);
  const candidates = ORDER.slice(index + 1).concat(ORDER.slice(0, index).reverse(), ORDER);
  return candidates.find(name => !used.includes(name));
}

function normalizeFromUrl(url) {
  const params = url.searchParams;
  const rawA = params.get("a");
  const rawB = params.get("b");
  const rawC = params.get("c");
  const hadAny = params.has("a") || params.has("b") || params.has("c");
  let corrected = false;
  let inferred = false;
  let a = bySlug[rawA];
  if (!a) {
    a = DEFAULT_NAMES[0];
    if (params.has("a")) corrected = true;
  }
  let b = bySlug[rawB];
  if (!b || b === a) {
    if (!params.has("b") && params.has("a") && bySlug[rawA]) {
      b = adjacentDistinct(a, [a]);
      inferred = true;
    } else {
      b = DEFAULT_NAMES.find(name => name !== a) || adjacentDistinct(a, [a]);
      if (params.has("b")) corrected = true;
    }
  }
  const next = [a, b];
  if (params.has("c") && rawC) {
    let c = bySlug[rawC];
    if (!c || next.includes(c)) {
      c = adjacentDistinct(b, next);
      corrected = true;
    }
    if (c) next.push(c);
  }
  let message = "";
  if (corrected) message = "Invalid or duplicate selections were corrected to distinct presidents.";
  else if (inferred) message = "President B was added as the next chronological distinct president.";
  else if (!hadAny) message = "";
  return {selected:next, message};
}

function urlFor(next) {
  const url = new URL(location.href);
  const hash = url.hash;
  url.search = "";
  url.searchParams.set("a", P[next[0]].slug);
  url.searchParams.set("b", P[next[1]].slug);
  if (next[2]) url.searchParams.set("c", P[next[2]].slug);
  url.hash = hash;
  return url;
}

function fillSelectors() {
  ["a", "b", "c"].forEach((slot, index) => {
    const select = document.getElementById(`president-${slot}`);
    const current = selected[index] || "";
    select.replaceChildren();
    if (index === 2 && !current) {
      const empty = document.createElement("option");
      empty.value = ""; empty.textContent = "Not selected"; empty.selected = true;
      select.appendChild(empty);
    }
    ORDER.forEach(name => {
      const option = document.createElement("option");
      option.value = P[name].slug;
      option.textContent = P[name].display_name;
      option.selected = name === current;
      option.disabled = selected.some((chosen, chosenIndex) => chosen === name && chosenIndex !== index);
      select.appendChild(option);
    });
    select.disabled = index === 2 && !selected[2];
    const hint=select.parentElement.querySelector(".selector-hint");
    hint.textContent=index===2 && !selected[2]
      ? "Use “Add third president” to activate."
      : `Selected president ${SLOT_LABELS[index]}`;
  });
  document.getElementById("toggle-third").textContent = selected[2] ? "Remove third president" : "Add third president";
}

function supportCardsHTML() {
  return selected.map((name, index) => {
    const p = P[name];
    const aiStatus = p.sample.ai_available ? "AI labels available" : "AI labels unavailable";
    return `<article class="support-card${p.sample.thin_record ? " thin" : ""}">
      <img src="portraits/${escapeHTML(p.slug)}.png" alt="Portrait of ${escapeHTML(p.display_name)}">
      <div><h3>${headingHTML(name,index)}</h3><p>${escapeHTML(p.party || "Party unavailable")}</p>
      <dl><div><dt>Corpus record span</dt><dd>${p.years.first}–${p.years.last}</dd></div>
      <div><dt>Corpus support</dt><dd>${escapeHTML(p.sample.speech_count_label)}</dd></div></dl>
      <p class="support-status">${escapeHTML(p.sample.support_label)} · ${escapeHTML(aiStatus)}</p>
      <p><a href="presidents/${escapeHTML(p.slug)}.html">Open full profile</a> · <a href="data/presidents/${escapeHTML(p.slug)}.json">Public JSON</a></p></div></article>`;
  }).join("");
}

function measureMatrixHTML(layer, caption) {
  const rows = PAGE.catalogs.measures[layer];
  const heads = selected.map((name,index) => `<th scope="col">${headingHTML(name,index)}</th>`).join("");
  const body = rows.map((row,rowIndex) => {
    const cells = selected.map(name => {
      const value = P[name].measures[layer][rowIndex];
      return `<td data-label="${escapeHTML(P[name].display_name)}"><strong>${escapeHTML(value[1])}</strong><span>${escapeHTML(value[3])}</span></td>`;
    }).join("");
    return `<tr><th scope="row"><strong>${escapeHTML(row.label)}</strong><span>${escapeHTML(row.unit)}</span></th>${cells}</tr>`;
  }).join("");
  return `<table class="comparison-matrix measure-matrix"><caption>${escapeHTML(caption)}</caption><thead><tr><th scope="col">Measure and unit</th>${heads}</tr></thead><tbody>${body}</tbody></table>`;
}

function radarTrace(name, index, layer) {
  const rows = PAGE.catalogs.measures[layer];
  const measures = P[name].measures[layer];
  const values = measures.map(value => value[2]);
  const theta = rows.map(row => row.label);
  const closedValues = values.concat([values[0]]);
  const closedTheta = theta.concat([theta[0]]);
  const absolute = measures.map(value => value[1]).concat([measures[0][1]]);
  const complete = values.every(Number.isFinite);
  return {
    type:"scatterpolar", mode:"lines+markers", name:P[name].display_name,
    r:closedValues, theta:closedTheta, customdata:absolute,
    fill:complete ? "toself" : "none",
    fillcolor:CHART_COLORS[index]+"24",
    line:{color:CHART_COLORS[index],width:2.5,dash:CHART_DASHES[index]},
    marker:{color:CHART_COLORS[index],size:6,symbol:CHART_SYMBOLS[index],line:{color:CHART_CHROME.surface,width:1}},
    connectgaps:false,
    hovertemplate:"%{theta}<br>Percentile %{r:.0f}<br>%{customdata}<extra>"+escapeHTML(P[name].display_name)+"</extra>"
  };
}

function renderRadar(layer, chartId) {
  const chart=document.getElementById(chartId);
  const available=selected.filter(name=>P[name].measures[layer].some(value=>Number.isFinite(value[2])));
  chart.setAttribute("aria-label",`${layer === "corpus" ? "Corpus-derived rhetorical" : "AI-labeled paragraph-signal"} percentile radar comparing ${selected.map(name=>P[name].display_name).join(", ")}. Exact values follow in the table disclosure.`);
  if (!window.Plotly) {
    chart.innerHTML='<p class="empty-state">Chart unavailable. Exact values remain available below.</p>';
    return;
  }
  const compact=window.innerWidth < 520;
  const annotations=available.length ? [] : [{text:"No eligible percentile shapes",showarrow:false,font:{color:CHART_CHROME.muted,size:13}}];
  Plotly.react(chart,available.map(name=>radarTrace(name,selected.indexOf(name),layer)),{
    autosize:true,showlegend:true,hovermode:"closest",paper_bgcolor:CHART_CHROME.surface,
    font:{family:CHART_FONT,color:CHART_CHROME.ink2,size:11},
    polar:{bgcolor:CHART_CHROME.surface,
      radialaxis:{range:[0,100],tickvals:[0,25,50,75,100],tickfont:{size:9,color:CHART_CHROME.muted},gridcolor:CHART_CHROME.grid,linecolor:CHART_CHROME.grid,angle:90},
      angularaxis:{rotation:90,direction:"clockwise",tickfont:{size:compact?9:10,color:CHART_CHROME.ink},gridcolor:CHART_CHROME.grid,linecolor:CHART_CHROME.grid}},
    legend:{orientation:"h",x:0,y:1.13,font:{size:10}},annotations,
    margin:{l:compact?55:72,r:compact?55:72,t:48,b:34}
  },{displayModeBar:false,responsive:true});
}

const RADAR_LAYERS=["corpus","ai"];
let activeRadarLayer="corpus";

function renderActiveRadar() {
  renderRadar(activeRadarLayer,`${activeRadarLayer}-radar`);
}

function setRadarTab(layer, focusTab=false) {
  if (!RADAR_LAYERS.includes(layer)) return;
  activeRadarLayer=layer;
  RADAR_LAYERS.forEach(name=>{
    const selected=name===layer;
    const tab=document.getElementById(`radar-tab-${name}`);
    const panel=document.getElementById(`radar-panel-${name}`);
    tab.setAttribute("aria-selected",String(selected));
    tab.tabIndex=selected ? 0 : -1;
    panel.hidden=!selected;
  });
  if (focusTab) document.getElementById(`radar-tab-${layer}`).focus();
  requestAnimationFrame(renderActiveRadar);
}

document.querySelectorAll(".radar-tab").forEach(tab=>{
  const layer=tab.id.replace("radar-tab-","");
  tab.addEventListener("click",()=>setRadarTab(layer));
  tab.addEventListener("keydown",event=>{
    const current=RADAR_LAYERS.indexOf(layer);
    let next=null;
    if (event.key==="ArrowRight") next=RADAR_LAYERS[(current+1)%RADAR_LAYERS.length];
    if (event.key==="ArrowLeft") next=RADAR_LAYERS[(current-1+RADAR_LAYERS.length)%RADAR_LAYERS.length];
    if (event.key==="Home") next=RADAR_LAYERS[0];
    if (event.key==="End") next=RADAR_LAYERS[RADAR_LAYERS.length-1];
    if (!next) return;
    event.preventDefault();
    setRadarTab(next,true);
  });
});

function canonicalNames(layer) {
  return PAGE.catalogs.agenda[layer].map(row => row.name);
}

function agendaRecords(name, layer) {
  const values=P[name].agenda[layer];
  if (!values) return [];
  return PAGE.catalogs.agenda[layer].map((definition,index)=>({...definition,share:values[index][0],rel:values[index][1]}));
}

function sharedRows(layer, topN, cap, positiveRelative=false, rankingKey="share") {
  const canonical = canonicalNames(layer);
  const order = Object.fromEntries(canonical.map((name,index) => [name,index]));
  const maps = Object.fromEntries(selected.map(name => [name, Object.fromEntries(agendaRecords(name,layer).map(row => [row.name,row]))]));
  const union = new Set();
  const rankValue = value => Number.isFinite(value) ? value : -Infinity;
  selected.forEach(name => {
    let rows = Object.values(maps[name]);
    if (positiveRelative) rows = rows.filter(row => Number.isFinite(row.rel) && row.rel > 0);
    rows.sort((left,right) => (rankValue(right[rankingKey]) - rankValue(left[rankingKey])) || ((order[left.name] ?? canonical.length) - (order[right.name] ?? canonical.length)));
    rows.slice(0,topN).forEach(row => union.add(row.name));
  });
  const maximum = name => Math.max(...selected.map(president => maps[president][name]?.[rankingKey]).filter(Number.isFinite), -Infinity);
  let names = [...union].sort((left,right) => (maximum(right) - maximum(left)) || ((order[left] ?? canonical.length) - (order[right] ?? canonical.length)));
  if (Number.isFinite(cap)) names = names.slice(0,cap);
  return names.map(name => {
    const first = selected.map(president => maps[president][name]).find(Boolean) || {name};
    return {name, domain:first.domain || "", values:Object.fromEntries(selected.map(president => [president,maps[president][name] || null]))};
  });
}

function compositionRows(field) {
  const canonical = [];
  ORDER.forEach(name => {
    const records = P[name].agenda[field];
    const names = Array.isArray(records) ? records.map(row => row.name) : Object.keys(records || {});
    names.forEach(item => { if (!canonical.includes(item)) canonical.push(item); });
  });
  const maps = Object.fromEntries(selected.map(name => {
    const records = P[name].agenda[field];
    return [name, Array.isArray(records) ? Object.fromEntries(records.map(row => [row.name,row.share])) : (records || {})];
  }));
  const maximum = item => Math.max(...selected.map(name => maps[name][item]).filter(Number.isFinite), -Infinity);
  return canonical.sort((left,right) => (maximum(right)-maximum(left)) || (canonical.indexOf(left)-canonical.indexOf(right))).map(item => ({
    name:item.replaceAll("_"," ").replace(/\b\w/g, char => char.toUpperCase()), domain:"",
    values:Object.fromEntries(selected.map(name => [name, Number.isFinite(maps[name][item]) ? {share:maps[name][item],rel:null} : null]))
  }));
}

function agendaTableHTML(rows, caption, relativeLabel) {
  const heads = selected.map((name,index) => `<th scope="col">${headingHTML(name,index)}</th>`).join("");
  let body = rows.map(row => {
    const rowLabel = escapeHTML(row.name) + (row.domain ? `<span>${escapeHTML(row.domain)}</span>` : "");
    const cells = selected.map(name => {
      const value = row.values[name];
      if (!value) return `<td data-label="${escapeHTML(P[name].display_name)}"><strong>N/A</strong><span>Not recorded</span></td>`;
      const relative = Number.isFinite(value.rel) ? `${formatSignedNumber(value.rel)} pp${relativeLabel ? " "+escapeHTML(relativeLabel) : ""}` : "";
      const share = Number.isFinite(value.share) ? `${formatNumber(value.share)}%` : "N/A";
      const detail = relative || (Number.isFinite(value.share) ? "" : "Not recorded");
      return `<td data-label="${escapeHTML(P[name].display_name)}"><strong>${share}</strong>${detail ? `<span>${detail}</span>` : ""}</td>`;
    }).join("");
    return `<tr><th scope="row">${rowLabel}</th>${cells}</tr>`;
  }).join("");
  if (!rows.length) body = `<tr><th scope="row">Unavailable</th><td colspan="${Math.max(1,selected.length)}">No rows are recorded for this selection.</td></tr>`;
  return `<table class="comparison-matrix agenda-matrix"><caption>${escapeHTML(caption)}</caption><thead><tr><th scope="col">Agenda row</th>${heads}</tr></thead><tbody>${body}</tbody></table>`;
}

function agendaHTML() {
  const allLegacyCount = canonicalNames("legacy").length || 1;
  return `<h3>Broad AI-labeled domains</h3>${agendaTableHTML(sharedRows("domains",4,12),"Shared broad-domain rows: each selected president’s top four","vs contemporaries")}
    <h3>Fine AI-labeled topics</h3>${agendaTableHTML(sharedRows("topics",3,9),"Shared fine-topic rows: each selected president’s top three","vs contemporaries")}
    <details class="sub-disclosure"><summary>Show extended AI agenda rows</summary><div class="disclosure-body">
    ${agendaTableHTML(sharedRows("domains",6,null),"Extended broad-domain union: each selected president’s top six","vs contemporaries")}
    ${agendaTableHTML(sharedRows("topics",6,null),"Extended fine-topic union: each selected president’s top six","vs contemporaries")}</div></details>
    <h3>Legacy issue lens</h3><p class="section-note">Deterministic CorEx issue rows remain separate from AI-labeled topics.</p>
    ${agendaTableHTML(sharedRows("legacy",4,12,true,"rel"),"Positive era-relative legacy issues: each selected president’s top four","vs era")}
    <details class="sub-disclosure"><summary>Show all legacy issues</summary><div class="disclosure-body">${agendaTableHTML(sharedRows("legacy",allLegacyCount,null),"All canonical legacy issues","vs era")}</div></details>
    <h3>Proposal and values composition</h3>${agendaTableHTML(compositionRows("proposal_values"),"Mutually exclusive AI-labeled paragraph composition","")}
    <h3>Speech-type composition</h3>${agendaTableHTML(compositionRows("speech_types"),"Leading assigned speech types in each corpus record","")}`;
}

function similarityHTML() {
  return Object.entries(FEATURE_META).map(([key,spec],lensIndex) => {
    const rows=selected.map((name,index)=>{
      const tiles=(P[name].neighbors[key] || []).map((item,itemIndex)=>{
        const neighborName=bySlug[item[0]];
        if (!neighborName) return "";
        const neighbor=P[neighborName], width=Math.max(0,Math.min(100,100*(item[1] || 0)));
        return `<li><a class="neighbor-tile" href="presidents/${escapeHTML(neighbor.slug)}.html">
          <span class="neighbor-rank" aria-label="Rank ${itemIndex+1}">${itemIndex+1}</span>
          <img src="portraits/${escapeHTML(neighbor.slug)}.png" alt=""><span class="neighbor-name">${escapeHTML(neighbor.display_name)}</span>
          <strong>${formatNumber(item[1],3)}</strong><span class="neighbor-meter" aria-hidden="true"><i style="width:${width.toFixed(2)}%"></i></span></a></li>`;
      }).filter(Boolean).join("") || '<li class="empty-state">Not available</li>';
      return `<div class="neighbor-row"><h4>${headingHTML(name,index)}</h4><ol class="neighbor-strip">${tiles}</ol></div>`;
    }).join("");
    return `<article class="similarity-panel"><header><p class="visual-kicker">Lens ${lensIndex+1} of 5</p><h3>${escapeHTML(spec.label)}</h3>
      <details class="measure-note"><summary>What this lens uses</summary><p>${escapeHTML(spec.description)}</p></details></header><div class="neighbor-rows">${rows}</div></article>`;
  }).join("");
}

const visualWidth = (value,maximum) => maximum>0 && Number.isFinite(value) ? Math.max(0,Math.min(100,100*value/maximum)) : 0;

function signalRowHTML(label,value,maximum,detail="") {
  return `<div class="signal-row"><div class="signal-label"><strong>${escapeHTML(label)}</strong>${detail ? `<span>${escapeHTML(detail)}</span>` : ""}</div>
    <div class="signal-meter" aria-hidden="true"><i style="width:${visualWidth(value,maximum).toFixed(2)}%"></i></div><b>${Number(value).toLocaleString()}</b></div>`;
}

function footprintHTML() {
  const definitions=[
    ["Speeches",name=>P[name].sample.n_speeches],
    ["Words",name=>P[name].evidence.n_words],
    ["Paragraphs",name=>P[name].evidence.n_paragraphs]
  ];
  const groups=definitions.map(([label,getValue])=>{
    const maximum=Math.max(...selected.map(getValue),0);
    const rows=selected.map((name,index)=>{
      const value=getValue(name);
      return `<div class="footprint-row"><span class="footprint-owner">${headingHTML(name,index)}</span>
        <span class="footprint-meter" aria-hidden="true"><i style="width:${visualWidth(value,maximum).toFixed(2)}%"></i></span><strong>${value.toLocaleString()}</strong></div>`;
    }).join("");
    return `<section class="footprint-group"><h4>${label}</h4>${rows}</section>`;
  }).join("");
  return `<section class="footprint-visual" aria-labelledby="footprint-title"><header><p class="visual-kicker">Selected-record scale</p><h3 id="footprint-title">Corpus footprint</h3>
    <p>Each row uses its own shared maximum; exact counts remain printed.</p></header><div class="footprint-groups">${groups}</div></section>`;
}

function evidenceHTML() {
  const adversaryMax=Math.max(...selected.flatMap(name=>P[name].evidence.adversaries.map(item=>item.mentions)),0);
  const invocationMax=Math.max(...selected.flatMap(name=>P[name].evidence.invocations.map(item=>item.mentions)),0);
  const cards=selected.map((name,index) => {
    const p=P[name], e=p.evidence;
    const adversaries=e.adversaries.map(item=>signalRowHTML(item.name,item.mentions,adversaryMax)).join("") || '<p class="empty-state">No repeated adversarial entity.</p>';
    const invocations=e.invocations.map(item=>{
      const details=[];
      if (item.function) details.push(`Function: ${item.function}`);
      if (item.stance) details.push(`Stance: ${item.stance}`);
      return signalRowHTML(item.target,item.mentions,invocationMax,details.join(" · "));
    }).join("") || '<p class="empty-state">No presidential invocation.</p>';
    const invocationSource=e.invocation_source==="ai" ? '<span class="source-tag exploratory">AI-labeled · exploratory</span>'
      : e.invocation_source==="legacy" ? '<span class="source-tag legacy">Legacy invocation counts</span>'
      : '<span class="source-tag legacy">Invocation source unavailable</span>';
    const vocabulary=e.vocabulary.map(term=>`<li>${escapeHTML(term)}</li>`).join("") || "<li>None recorded</li>";
    const signatures=e.signatures.map(item=>`<a class="speech-tile" href="${escapeHTML(item.url)}" target="_blank" rel="noopener"><span>${escapeHTML(item.title)}</span>${Number.isFinite(item.year) ? `<b>${item.year}</b>` : ""}</a>`).join("") || '<span class="empty-state">No signature speech recorded.</span>';
    const excerpts=e.excerpts.map(item=>`<figure><blockquote>${escapeHTML(item.quote)}</blockquote><figcaption>${escapeHTML(item.issue)} · ${escapeHTML(item.cite)}</figcaption></figure>`).join("") || "<p>No excerpt selected for this record.</p>";
    return `<article class="evidence-card visual-evidence-card"><header><img src="portraits/${escapeHTML(p.slug)}.png" alt=""><h3>${headingHTML(name,index)}</h3></header>
      <div class="signal-grid"><section><div class="signal-heading"><h4>Adversarial entities</h4><span class="source-tag exploratory">AI · exploratory</span></div>${adversaries}</section>
      <section><div class="signal-heading"><h4>Presidential invocations</h4>${invocationSource}</div>${invocations}</section></div>
      <section class="compact-evidence"><div><h4>Distinctive vocabulary</h4><ul class="term-list">${vocabulary}</ul></div><div><h4>Signature speeches</h4><div class="speech-tiles">${signatures}</div></div></section>
      <details class="excerpt-disclosure"><summary>Read source excerpts (${e.excerpts.length})</summary><div>${excerpts}</div></details>
      <nav class="evidence-actions" aria-label="${escapeHTML(p.display_name)} evidence links"><a href="presidents/${escapeHTML(p.slug)}.html">Full evidence</a><a href="data/presidents/${escapeHTML(p.slug)}.json">Canonical JSON</a></nav></article>`;
  }).join("");
  return footprintHTML()+`<div class="evidence-grid visual-evidence-grid">${cards}</div>`;
}

function render(message="") {
  fillSelectors();
  document.getElementById("support-cards").innerHTML=supportCardsHTML();
  document.getElementById("rhetoric-matrices").innerHTML=measureMatrixHTML("corpus","All eight corpus-derived Profile V3 measures")+measureMatrixHTML("ai","All six AI-labeled Profile V3 measures");
  renderActiveRadar();
  document.getElementById("agenda-content").innerHTML=agendaHTML();
  document.getElementById("similarity-content").innerHTML=similarityHTML();
  document.getElementById("evidence-content").innerHTML=evidenceHTML();
  document.getElementById("selection-status").textContent=message;
}

function commitSelection(next, message) {
  selected=next;
  history.pushState({selection:selected.map(name=>P[name].slug)},"",urlFor(selected));
  render(message);
}

["a","b","c"].forEach((slot,index) => document.getElementById(`president-${slot}`).addEventListener("change", event => {
  const name=bySlug[event.target.value];
  if (!name || selected.some((chosen,chosenIndex)=>chosen===name && chosenIndex!==index)) return;
  const next=selected.slice(); next[index]=name;
  commitSelection(next,`${P[name].display_name} selected as President ${SLOT_LABELS[index]}.`);
}));
document.getElementById("swap-presidents").addEventListener("click",()=>commitSelection([selected[1],selected[0],...selected.slice(2)],"President A and President B were swapped."));
document.getElementById("toggle-third").addEventListener("click",()=>{
  if (selected[2]) commitSelection(selected.slice(0,2),"Optional President C was removed.");
  else {
    const third=adjacentDistinct(selected[1],selected);
    commitSelection([...selected,third],`${P[third].display_name} was added as optional President C.`);
  }
});
document.getElementById("reset-comparison").addEventListener("click",()=>commitSelection(DEFAULT_NAMES.slice(0,2),"Comparison reset to Abraham Lincoln and Franklin D. Roosevelt."));
document.getElementById("copy-link").addEventListener("click",async()=>{
  const status=document.getElementById("selection-status"), value=urlFor(selected).href;
  try {
    if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(value);
    else {
      const input=document.createElement("textarea"); input.value=value; input.setAttribute("readonly",""); input.style.position="fixed"; input.style.opacity="0";
      document.body.appendChild(input); input.select(); if (!document.execCommand("copy")) throw new Error("copy failed"); input.remove();
    }
    status.textContent="Comparison link copied.";
  } catch (error) { status.textContent="Could not copy the link. Copy it from the address bar."; }
});
window.addEventListener("popstate",()=>{
  const normalized=normalizeFromUrl(new URL(location.href));
  selected=normalized.selected;
  const canonical=urlFor(selected);
  if (canonical.href!==location.href) history.replaceState({selection:selected.map(name=>P[name].slug)},"",canonical);
  render(normalized.message || "Comparison restored from browser history.");
});

document.getElementById("download-comparison").addEventListener("click",async()=>{
  const button=document.getElementById("download-comparison"), status=document.getElementById("download-status");
  button.disabled=true; status.textContent="Preparing comparison download…";
  try {
    const presidents=await Promise.all(selected.map(async (name,index)=>{
      const sourcePath=`data/presidents/${P[name].slug}.json`;
      const response=await fetch(sourcePath);
      if (!response.ok) throw new Error(`Could not fetch ${sourcePath}`);
      const profile=await response.json();
      if (profile.schema_version!=="president-profile-v3" || profile.slug!==P[name].slug) throw new Error(`Invalid canonical profile ${sourcePath}`);
      return {position:SLOT_LABELS[index].toLowerCase(),slug:P[name].slug,source_url:new URL(sourcePath,location.href).href,source_path:sourcePath,profile};
    }));
    const bundle={schema_version:DOWNLOAD_SCHEMA_VERSION,profile_schema_version:"president-profile-v3",selection_order:presidents.map(item=>item.slug),source_urls:presidents.map(item=>item.source_url),presidents};
    const blob=new Blob([JSON.stringify(bundle,null,2)],{type:"application/json"});
    const link=document.createElement("a"); link.href=URL.createObjectURL(blob);
    link.download=`president-comparison-v2-${presidents.map(item=>item.slug).join("-vs-")}.json`;
    document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(link.href);
    status.textContent="Comparison download ready.";
  } catch (error) { status.textContent="Comparison download failed. Check the connection and try again."; }
  finally { button.disabled=false; }
});

const initial=normalizeFromUrl(new URL(location.href));
selected=initial.selected;
history.replaceState({selection:selected.map(name=>P[name].slug)},"",urlFor(selected));
render(initial.message);
</script>
</body>
</html>'''


def render_compare_page(payload: Mapping[str, dict]) -> str:
    """Render a complete Compare V2 page, including the no-JS default pair."""
    selected = _default_selection(payload)
    if len(payload["presidents"]) >= 2 and len(selected) != 2:
        raise ValueError("Compare requires two distinct default presidents")
    rhetoric = "".join((
        measure_matrix(
            payload, selected, "corpus",
            "All eight corpus-derived Profile V3 measures",
        ),
        measure_matrix(
            payload, selected, "ai",
            "All six AI-labeled Profile V3 measures",
        ),
    ))
    page = _page_template()
    replacements = {
        "__PAGE_CSS__": PAGE_CSS,
        "__SELECTOR_FIELDSET__": selector_fieldset(payload, selected),
        "__SUPPORT_CARDS__": selector_support_cards(payload, selected),
        "__RHETORIC_MATRICES__": rhetoric,
        "__AGENDA_CONTENT__": agenda_sections(payload, selected),
        "__SIMILARITY_CONTENT__": similarity_sections(payload, selected),
        "__EVIDENCE_CONTENT__": evidence_sections(payload, selected),
        "__PAYLOAD__": json_for_script(payload),
        "__DEFAULT_NAMES__": json_for_script(selected),
        "__CHART_COLORS__": json_for_script(
            [SERIES[0], SERIES[2], SERIES[4]]
        ),
        "__CHART_FONT__": json_for_script(FONT),
        "__CHART_CHROME__": json_for_script({
            "surface": SURFACE,
            "ink": INK,
            "ink2": INK2,
            "muted": MUTED,
            "grid": GRID,
        }),
    }
    for marker, value in replacements.items():
        page = page.replace(marker, value)
    unresolved = [marker for marker in replacements if marker in page]
    if unresolved:
        raise ValueError(f"Unresolved Compare template markers: {unresolved}")
    return page


def write_compare(
    data: dict,
    display_issues: list[str],
    *,
    profile_views: dict[str, dict] | None = None,
) -> None:
    """Write the deterministic Compare V2 page from normalized Profile V3 views."""
    payload = build_payload(
        data, display_issues, profile_views=profile_views
    )
    page = render_compare_page(payload)
    out = REPO_ROOT / "docs" / "compare.html"
    out.write_text(page, encoding="utf-8")
    print(f"  wrote docs/compare.html ({out.stat().st_size:,} bytes)")
