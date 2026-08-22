"""Fetch and cache president portrait thumbnails (public domain, Wikipedia).

Downloads each president's Wikipedia summary thumbnail once, circle-crops it
to a 64px PNG in docs/portraits/, and exposes them as data URIs for embedding
in charts (data URIs keep the artifact/preview builds self-contained).
"""

import base64
import io
import time

import requests
from PIL import Image, ImageDraw, ImageOps

from .figures import REPO_ROOT
from .profiles import PRESIDENT_DISPLAY_NAMES, slug

PORTRAIT_DIR = REPO_ROOT / "docs" / "portraits"

# Corpus name -> Wikipedia page title, where they differ.
WIKI_TITLES = {
    **PRESIDENT_DISPLAY_NAMES,
    "Richard M. Nixon": "Richard Nixon",
}

SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
UA = {"User-Agent": "presidential-profiles/2.0 (research; github.com/jacobfulfyll)"}


def _circle_crop(img: Image.Image, size: int = 64) -> Image.Image:
    img = ImageOps.fit(img.convert("RGB"), (size, size), Image.LANCZOS,
                       centering=(0.5, 0.35))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
    out = Image.new("RGBA", (size, size))
    out.paste(img, mask=mask)
    return out


def _get(url: str) -> requests.Response:
    """Polite fetch: paced, with backoff on Wikipedia's 429s."""
    for attempt in range(5):
        resp = requests.get(url, headers=UA, timeout=30)
        if resp.status_code != 429:
            return resp
        time.sleep(5 * (attempt + 1))
    return resp


def fetch_portraits(presidents: list[str]) -> None:
    PORTRAIT_DIR.mkdir(parents=True, exist_ok=True)
    for p in presidents:
        path = PORTRAIT_DIR / f"{slug(p)}.png"
        if path.exists():
            continue
        time.sleep(1.5)
        title = WIKI_TITLES.get(p, p).replace(" ", "_")
        meta_resp = _get(SUMMARY_URL.format(title))
        try:
            meta = meta_resp.json()
        except Exception:
            print(f"  !! {p}: summary fetch failed ({meta_resp.status_code})")
            continue
        src = meta.get("thumbnail", {}).get("source")
        if not src:
            print(f"  !! no thumbnail for {p}")
            continue
        resp = _get(src)
        try:
            img = _circle_crop(Image.open(io.BytesIO(resp.content)))
        except Exception as e:
            print(f"  !! {p}: {resp.status_code} "
                  f"{resp.headers.get('content-type')} - {e}")
            continue
        img.save(path)
        print(f"  {p} -> {path.name}")


def data_uris(presidents: list[str]) -> dict[str, str]:
    """president -> data URI of their portrait (empty if missing)."""
    out = {}
    for p in presidents:
        path = PORTRAIT_DIR / f"{slug(p)}.png"
        if path.exists():
            b64 = base64.b64encode(path.read_bytes()).decode()
            out[p] = f"data:image/png;base64,{b64}"
    return out
