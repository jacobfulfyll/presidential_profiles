"""Rendered whole-site phone acceptance backed by the dependency-free CDP audit."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPO = Path(__file__).parents[1]
SCRIPT = REPO / "scripts" / "audit_mobile_site.mjs"
CHROME_CANDIDATES = (
    os.environ.get("CHROME_BINARY"),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)
HAS_CHROME = any(candidate and Path(candidate).is_file() for candidate in CHROME_CANDIDATES)


@pytest.mark.skipif(
    not HAS_CHROME or shutil.which("node") is None,
    reason="local Chrome/Chromium and Node are required for rendered acceptance",
)
def test_every_generated_page_passes_the_mobile_browser_audit():
    result = subprocess.run(
        ["node", str(SCRIPT), "--docs", str(REPO / "docs")],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "Mobile site audit passed: 74 pages × 5 viewports" in result.stdout
