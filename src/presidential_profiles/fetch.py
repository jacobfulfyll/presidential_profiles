"""Fetch the Miller Center presidential speech corpus.

The Miller Center of Public Affairs (University of Virginia) publishes its
full speech corpus as a tarball of JSON files at data.millercenter.org.
This replaces the 2019 Selenium scraper — the official download includes
everything through the current presidency.
"""

import argparse
import html
import json
import re
import tarfile
from pathlib import Path

import pandas as pd
import requests

CORPUS_URL = "https://data.millercenter.org/miller_center_speeches.tgz"

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PARQUET_PATH = REPO_ROOT / "data" / "speeches.parquet"


def download(force: bool = False) -> Path:
    """Download the corpus tarball to data/raw/, skipping if present."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tgz_path = RAW_DIR / "miller_center_speeches.tgz"
    if tgz_path.exists() and not force:
        print(f"Using cached tarball: {tgz_path}")
        return tgz_path
    print(f"Downloading {CORPUS_URL} ...")
    resp = requests.get(CORPUS_URL, timeout=120)
    resp.raise_for_status()
    tgz_path.write_bytes(resp.content)
    print(f"Saved {len(resp.content):,} bytes to {tgz_path}")
    return tgz_path


_TAG_RE = re.compile(r"<[^>]+>")
_STAGE_RE = re.compile(
    r"[\(\[]\s*(?:applause|laughter|cheers|booing|crosstalk|inaudible|laughter and applause)"
    r"[^\)\]]*[\)\]]",
    re.IGNORECASE,
)
_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                            "–": "-", "—": " - ", "\xa0": " "})


def clean_transcript(text: str) -> str:
    """Strip HTML residue, stage directions, and normalize punctuation."""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _STAGE_RE.sub(" ", text)
    text = text.translate(_QUOTE_MAP)
    return re.sub(r"\s+", " ", text).strip()


def parse(tgz_path: Path) -> pd.DataFrame:
    """Parse every speech JSON in the tarball into one tidy DataFrame."""
    records = []
    with tarfile.open(tgz_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".json"):
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            doc = json.load(fh)
            records.append(
                {
                    # doc_name (the site slug) is the stable unique key;
                    # uuid is absent from nearly all corpus files.
                    "doc_name": doc.get("doc_name"),
                    "president": doc.get("president"),
                    "date": doc.get("date"),
                    "title": doc.get("title"),
                    "transcript": clean_transcript(doc.get("transcript") or ""),
                    "introduction": doc.get("introduction") or "",
                }
            )
    df = pd.DataFrame(records)
    # Dates carry mixed historical UTC offsets; keep the calendar date only.
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
    df["year"] = df["date"].dt.year
    df["word_count"] = df["transcript"].str.split().str.len()
    df = df.dropna(subset=["doc_name", "president", "date"])
    df = df[df["word_count"] > 0]
    df = df.drop_duplicates(subset="doc_name")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the Miller Center speech corpus")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    tgz_path = download(force=args.force)
    df = parse(tgz_path)
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, index=False)
    print(
        f"Wrote {len(df):,} speeches ({df['date'].min():%Y-%m-%d} to "
        f"{df['date'].max():%Y-%m-%d}) to {PARQUET_PATH}"
    )


if __name__ == "__main__":
    main()
