"""Load the speech corpus and attach president-level metadata."""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PARQUET_PATH = DATA_DIR / "speeches.parquet"

# Party affiliation while in office, keyed by the exact corpus name strings.
PARTY = {
    "George Washington": "Unaffiliated",
    "John Adams": "Federalist",
    "Thomas Jefferson": "Democratic-Republican",
    "James Madison": "Democratic-Republican",
    "James Monroe": "Democratic-Republican",
    "John Quincy Adams": "Democratic-Republican",
    "Andrew Jackson": "Democratic",
    "Martin Van Buren": "Democratic",
    "William Harrison": "Whig",
    "John Tyler": "Whig",
    "James K. Polk": "Democratic",
    "Zachary Taylor": "Whig",
    "Millard Fillmore": "Whig",
    "Franklin Pierce": "Democratic",
    "James Buchanan": "Democratic",
    "Abraham Lincoln": "Republican",
    "Andrew Johnson": "Democratic",
    "Ulysses S. Grant": "Republican",
    "Rutherford B. Hayes": "Republican",
    "James A. Garfield": "Republican",
    "Chester A. Arthur": "Republican",
    "Grover Cleveland": "Democratic",
    "Benjamin Harrison": "Republican",
    "William McKinley": "Republican",
    "Theodore Roosevelt": "Republican",
    "William Taft": "Republican",
    "Woodrow Wilson": "Democratic",
    "Warren G. Harding": "Republican",
    "Calvin Coolidge": "Republican",
    "Herbert Hoover": "Republican",
    "Franklin D. Roosevelt": "Democratic",
    "Harry S. Truman": "Democratic",
    "Dwight D. Eisenhower": "Republican",
    "John F. Kennedy": "Democratic",
    "Lyndon B. Johnson": "Democratic",
    "Richard M. Nixon": "Republican",
    "Gerald Ford": "Republican",
    "Jimmy Carter": "Democratic",
    "Ronald Reagan": "Republican",
    "George H. W. Bush": "Republican",
    "Bill Clinton": "Democratic",
    "George W. Bush": "Republican",
    "Barack Obama": "Democratic",
    "Donald Trump": "Republican",
    "Joe Biden": "Democratic",
}


def load() -> pd.DataFrame:
    """Load speeches with party metadata, sorted chronologically."""
    df = pd.read_parquet(PARQUET_PATH)
    df["party"] = df["president"].map(PARTY)
    df["decade"] = (df["year"] // 10) * 10
    return df.sort_values("date").reset_index(drop=True)


def president_order(df: pd.DataFrame) -> list[str]:
    """Presidents ordered by date of their first speech in the corpus."""
    return list(df.groupby("president")["date"].min().sort_values().index)
