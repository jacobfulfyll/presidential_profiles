"""Build-time validation for generated navigation, links, anchors and JSON shards."""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

HREF_RE = re.compile(r"""(?:href|src)=["']([^"']+)["']""", re.I)
ID_RE = re.compile(r"""(?:id|name)=["']([^"']+)["']""", re.I)
CLASS_RE = re.compile(r"""class=["']([^"']+)["']""", re.I)
EVIDENCE_SUMMARY_RE = re.compile(
    r"<summary>\s*(?:Inspect (?:the |edge )?evidence|"
    r"<span[^>]*>.*?</span>\s*Evidence\s*·[^<]*)\s*</summary>", re.I
)
MEASURE_SUMMARY_RE = re.compile(
    r"<summary>\s*(?:Explain this measure|"
    r"<span[^>]*>.*?</span>\s*Measure\s*·[^<]*)\s*</summary>", re.I
)
RETIRED_PUBLIC_ROUTES = {"networks.html"}


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
        chart_count = sum(
            "chart" in classes.split() for classes in CLASS_RE.findall(text)
        )
        lesson_count = len(MEASURE_SUMMARY_RE.findall(text))
        evidence_count = len(EVIDENCE_SUMMARY_RE.findall(text))
        if lesson_count < chart_count:
            errors.append(
                f"{page.relative_to(site_dir)}: {chart_count} substantive charts "
                f"but only {lesson_count} metric lessons"
            )
        if evidence_count < chart_count:
            errors.append(
                f"{page.relative_to(site_dir)}: {chart_count} substantive charts "
                f"but only {evidence_count} evidence controls"
            )
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
            json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            invalid_json.append(f"{path.relative_to(site_dir)}: {exc}")
    errors.extend(invalid_json)
    report = {"schema_version": "site-validation-v1", "html_pages": len(pages),
              "json_shards": len(list(site_dir.rglob("*.json"))),
              "errors": errors, "ok": not errors}
    if write_report:
        (site_dir / "validation-report.json").write_text(json.dumps(report, indent=2))
    if errors:
        raise ValueError("generated-site validation failed:\n" + "\n".join(errors[:30]))
    return report
