"""Rendered-browser acceptance for the five governed Data/Methods pages."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPO = Path(__file__).parents[1]
SCRIPT = REPO / "scripts" / "audit_data_trust_pages.mjs"
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
def test_data_trust_pages_pass_rendered_accessibility_and_responsive_audit():
    result = subprocess.run(
        ["node", str(SCRIPT), "--docs", str(REPO / "docs")],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "Data Trust browser audit passed" in result.stdout
