"""Build-time validation for generated navigation, links, anchors and JSON shards."""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

HREF_RE = re.compile(r"""(?:href|src)=["']([^"']+)["']""", re.I)
ID_RE = re.compile(r"""(?:id|name)=["']([^"']+)["']""", re.I)
SEMANTIC_CHART_RE = re.compile(
    r"<(?:figure|div)\b[^>]*\bdata-substantive-chart(?:=(?:[\"'][^\"']*[\"']))?[^>]*>",
    re.I,
)
SEMANTIC_FIGURE_BLOCK_RE = re.compile(
    r"(<figure\b[^>]*\bdata-substantive-chart(?:=(?:[\"'][^\"']*[\"']))?[^>]*>)"
    r"(.*?)</figure>",
    re.I | re.S,
)
FIGURE_RE = re.compile(r"<figure\b[^>]*>", re.I)
CHARTISH_CONTAINER_RE = re.compile(
    r"<(?:figure|div)\b[^>]*\bclass=[\"'][^\"']*(?:chart|plot)[^\"']*[\"'][^>]*>",
    re.I,
)
ATTRIBUTE_RE = re.compile(r"([:\w-]+)=[\"']([^\"']*)[\"']", re.I)
RETIRED_PUBLIC_ROUTES = {"networks.html"}
GOVERNED_ANALYTICAL_ROUTES = {
    "data-quality.html",
    "methodology.html",
    "era-boundaries.html",
    "label-models.html",
    "metrics.html",
}


def _reject_nonfinite_json(value: str):
    raise ValueError(f"non-finite JSON value {value}")


def _semantic_chart_errors(
    page: Path, text: str, *, governed_route: bool = False
) -> tuple[int, list[str]]:
    """Validate explicit substantive-figure semantics and metric receipts."""
    tags = SEMANTIC_CHART_RE.findall(text)
    errors: list[str] = []
    for index, tag in enumerate(tags, start=1):
        attributes = {
            name.casefold(): value.strip()
            for name, value in ATTRIBUTE_RE.findall(tag)
        }
        prefix = f"{page}: substantive chart {index}"
        if not tag.lstrip().casefold().startswith("<figure"):
            errors.append(f"{prefix} must use semantic figure markup")
        metric_value = attributes.get("data-metric", "")
        if not metric_value:
            errors.append(f"{prefix} is missing data-metric")
        else:
            from .metrics import METRICS

            metric_names = {
                name
                for name in re.split(r"[\s,]+", metric_value)
                if name
            }
            unknown = sorted(metric_names - set(METRICS))
            if unknown:
                errors.append(
                    f"{prefix} names unregistered data-metric values {unknown}"
                )
        if not attributes.get("data-evidence"):
            errors.append(f"{prefix} is missing data-evidence")
        if not (
            attributes.get("aria-label")
            or attributes.get("aria-labelledby")
        ):
            errors.append(f"{prefix} is missing an accessible name")
    if governed_route:
        unmarked_figures = [
            tag
            for tag in FIGURE_RE.findall(text)
            if "data-substantive-chart" not in tag.casefold()
        ]
        if unmarked_figures:
            errors.append(
                f"{page}: {len(unmarked_figures)} figure(s) lack "
                "data-substantive-chart"
            )

        outside_marked_figures = SEMANTIC_FIGURE_BLOCK_RE.sub("", text)
        unmarked_chartish = CHARTISH_CONTAINER_RE.findall(outside_marked_figures)
        if unmarked_chartish:
            errors.append(
                f"{page}: {len(unmarked_chartish)} chart/plot container(s) are "
                "outside a marked substantive figure"
            )

        for index, (tag, body) in enumerate(
            SEMANTIC_FIGURE_BLOCK_RE.findall(text), start=1
        ):
            attributes = {
                name.casefold(): value.strip()
                for name, value in ATTRIBUTE_RE.findall(tag)
            }
            metric_names = filter(
                None, re.split(r"[\s,]+", attributes.get("data-metric", ""))
            )
            for metric_name in metric_names:
                metric_link = re.compile(
                    rf"href=[\"'][^\"']*metrics\.html#{re.escape(metric_name)}[\"']",
                    re.I,
                )
                if not metric_link.search(body):
                    errors.append(
                        f"{page}: substantive chart {index} does not visibly link "
                        f"metric definition {metric_name}"
                    )
    return len(tags), errors


def _target(page: Path, site_dir: Path, href: str) -> tuple[Path | None, str]:
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc or href.startswith(("#", "mailto:", "javascript:", "data:")):
        return None, parsed.fragment
    path = unquote(parsed.path)
    target = (page.parent / path).resolve() if path else page.resolve()
    if target.is_dir():
        target = target / "index.html"
    return target, parsed.fragment


def validate_site(site_dir: Path, write_report: bool = True) -> dict:
    site_dir = site_dir.resolve()
    errors = []
    pages = sorted(page for page in site_dir.rglob("*.html")
                   if not page.name.endswith("_selfcontained.html"))
    for page in pages:
        relative_page = page.relative_to(site_dir).as_posix()
        if relative_page in RETIRED_PUBLIC_ROUTES:
            errors.append(f"{relative_page}: retired public route is still generated")
        text = page.read_text()
        if 'aria-label="Primary"' not in text:
            errors.append(f"{page.relative_to(site_dir)}: missing primary navigation")
        _, semantic_errors = _semantic_chart_errors(
            page.relative_to(site_dir),
            text,
            governed_route=relative_page in GOVERNED_ANALYTICAL_ROUTES,
        )
        errors.extend(semantic_errors)
        for href in HREF_RE.findall(text):
            if "${" in href or "function(" in href or len(href) > 500:
                continue
            parsed_href = urlsplit(href)
            if (
                not parsed_href.scheme
                and not parsed_href.netloc
                and parsed_href.path.rstrip("/").endswith("networks.html")
            ):
                errors.append(
                    f"{page.relative_to(site_dir)}: link targets retired route: {href}"
                )
                continue
            target, fragment = _target(page, site_dir, href)
            if target is None:
                continue
            try:
                target.relative_to(site_dir)
            except ValueError:
                errors.append(f"{page.relative_to(site_dir)}: link escapes site: {href}")
                continue
            if not target.exists():
                errors.append(f"{page.relative_to(site_dir)}: missing target: {href}")
                continue
            if fragment and target.suffix == ".html":
                ids = set(ID_RE.findall(target.read_text()))
                if unquote(fragment) not in ids:
                    errors.append(f"{page.relative_to(site_dir)}: missing anchor: {href}")
    invalid_json = []
    for path in site_dir.rglob("*.json"):
        try:
            json.loads(
                path.read_text(),
                parse_constant=_reject_nonfinite_json,
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            invalid_json.append(f"{path.relative_to(site_dir)}: {exc}")
    errors.extend(invalid_json)
    report = {"schema_version": "site-validation-v2", "html_pages": len(pages),
              "json_shards": len(list(site_dir.rglob("*.json"))),
              "errors": errors, "ok": not errors}
    if write_report:
        (site_dir / "validation-report.json").write_text(json.dumps(report, indent=2))
    if errors:
        raise ValueError("generated-site validation failed:\n" + "\n".join(errors[:30]))
    return report
