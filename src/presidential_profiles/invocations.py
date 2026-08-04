"""Auditable, resumable extraction and annotation workflow for presidential mentions.

This intentionally does not alter the legacy 101-row tone artifact.  V2 rows
have stable content-derived ids and retain excluded candidates for review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from . import corpus, trends

ROOT = corpus.DATA_DIR / "invocations_v2"
CANDIDATES = ROOT / "candidates.parquet"
CLASSIFICATIONS = ROOT / "classifications.parquet"
EDGES = ROOT / "edges.parquet"
FUNCTIONS = {"legacy/inheritance", "institutional precedent", "policy inheritance",
             "historical comparison", "contemporary rivalry", "ceremonial/biographical", "other/unclear"}
STANCES = {"positive", "negative", "mixed", "neutral", "unclear"}
RUBRIC_VERSION = "invocation-v2-rubric-1"


def alias_registry() -> dict[str, list[str]]:
    """Deterministic full-name/title/surname aliases for every corpus president."""
    out = {}
    for name in corpus.PARTY:
        parts = name.replace(".", "").split()
        surname = parts[-1]
        initialed = " ".join([f"{p[0]}." if len(p) == 1 else p for p in parts])
        aliases = {name, surname, f"President {surname}", f"Mr. {surname}", initialed}
        # Common public variants that are not generated mechanically.
        extras = {"William Harrison": {"William Henry Harrison", "W. H. Harrison"},
                  "Theodore Roosevelt": {"Teddy Roosevelt", "TR"}, "Franklin D. Roosevelt": {"FDR", "Franklin Roosevelt"},
                  "John F. Kennedy": {"JFK", "John Kennedy"}, "Lyndon B. Johnson": {"LBJ", "Lyndon Johnson"},
                  "Richard M. Nixon": {"Richard Nixon"}, "George H. W. Bush": {"George Bush", "George H.W. Bush"},
                  "George W. Bush": {"George W Bush", "George W. Bush", "Bush 43"},
                  "Bill Clinton": {"William Jefferson Clinton", "William Clinton"},
                  "Barack Obama": {"President Obama"},
                  "Donald Trump": {"Trump"}, "Joe Biden": {"Biden", "Joseph Biden"}}
        aliases |= extras.get(name, set())
        out[name] = sorted(aliases, key=lambda x: (-len(x), x))
    return out


def _sentences(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r"[^.!?]+[.!?]?(?:\s|$)", text)]


def _status_on(target: str, speech_date: str, presidents: pd.DataFrame) -> str:
    when = pd.Timestamp(speech_date)
    rows = presidents[presidents.president == target]
    if rows.empty:
        return "unknown"
    first, last = pd.to_datetime(rows.date).min(), pd.to_datetime(rows.date).max()
    if when < first: return "not_yet_president"
    if when > last: return "former_president"
    return "serving_or_contemporary"


def _resolve_ambiguous(targets: list[str], alias: str, speech_date: str,
                       speaker: str, speeches: pd.DataFrame) -> list[str]:
    """Resolve shared surnames by chronology; full-name aliases stay exact."""
    if len(targets) <= 1:
        return targets
    # An explicit multiword alias normally identifies one target; overlapping
    # generated aliases such as George Bush are settled by exact registry hit.
    exact = [target for target in targets
             if alias.casefold() == target.casefold()]
    if len(exact) == 1:
        return exact
    when = pd.Timestamp(speech_date)
    first = speeches.groupby("president").date.min().map(pd.Timestamp)
    eligible = [target for target in targets if first.get(target, pd.Timestamp.max) <= when]
    pool = eligible or targets
    # In a later president's own speech, a bare shared surname can be a
    # self-reference; retain it for the exclusion audit rather than forcing it
    # onto the earlier namesake.
    if speaker in pool:
        return [speaker]
    return [max(pool, key=lambda target: first.get(target, pd.Timestamp.min))]


def _source_role(context: str) -> tuple[str, str]:
    prefix = context.lstrip()[:80]
    if re.match(r"(?i)(q:|question:|reporter:|the press:)", prefix):
        return "reporter", "reporter_speech"
    if re.match(r"(?i)(audience:|audience member:|crowd:)", prefix):
        return "audience", "audience_speech"
    if re.match(r"(?i)(moderator:)", prefix):
        return "moderator", "moderator_speech"
    return "presidential_speech", ""


def _is_bare_surname(target: str, raw: str) -> bool:
    surname = target.replace(".", "").split()[-1]
    return raw.strip().casefold() == surname.casefold()


def _washington_is_person(context: str) -> bool:
    """Conservative cues that bare ``Washington`` denotes the president.

    Full-name and titled forms are already unambiguous. Bare uses default to
    place/institution because the modern corpus overwhelmingly uses Washington
    for the capital, state, government, press, or political establishment.
    """
    person_cues = (
        r"\bWashington(?:['’]s)\s+"
        r"(?:time|day|days|life|death|birthday|legacy|example|warning|words|"
        r"vision|leadership|presidency|principles|policy|experience|spirit|army)\b",
        r"\bWashington\s+(?:said|wrote|warned|believed|understood|urged|led|"
        r"fought|served|stood|sought|opposed|favored|called|"
        r"became|lived|died|took|established)\b",
        r"\b(?:like|unlike|since|under|after|before)\s+Washington\b",
        r"\b(?:warning|example|words|farewell|legacy|vision|life|memory|"
        r"experience|principles)\s+of\s+Washington\b",
        r"\bWashington\s+(?:and|or)\s+(?:Adams|Jefferson|Madison|Jackson|"
        r"Lincoln|Franklin|Hamilton)\b",
    )
    return any(re.search(pattern, context, re.I) for pattern in person_cues)


AMBIGUOUS_BARE_SURNAMES = {
    "Adams", "Arthur", "Bush", "Carter", "Cleveland", "Ford", "Garfield",
    "Grant", "Harrison", "Hayes", "Hoover", "Jackson", "Johnson", "Kennedy",
    "Lincoln", "Madison", "Obama", "Pierce", "Polk", "Reagan", "Roosevelt",
    "Taylor", "Tyler", "Washington", "Wilson",
}


def _named_person_collision(target: str, raw: str, context: str) -> bool:
    """Whether a bare surname is visibly attached to somebody else's name."""
    surname = re.escape(raw.strip())
    allowed = {
        token.casefold() for token in target.replace(".", "").split()[:-1]
        if len(token) > 1
    }
    for match in re.finditer(rf"\b([A-Z][a-z]+)\s+{surname}\b", context):
        if match.group(1).casefold() not in allowed:
            return True
    return bool(re.search(
        rf"\b(?:Prime Minister|Senator|Representative|Congresswoman|"
        rf"Congressman|Governor|Justice|Judge|Secretary|General)\s+{surname}\b",
        context,
    ))


def _institutional_surname_use(raw: str, context: str) -> bool:
    surname = re.escape(raw.strip())
    return bool(re.search(
        rf"\b{surname}\s+(?:Center|Centre|Space Center|Airport|School|University|"
        rf"Institute|Foundation|Library|County|Street|Avenue|Square|Heights|"
        rf"Motor|administration building|Dam|Hole)\b"
        rf"|\b(?:city|county|state|town|port|university|center|airport|dam)\s+"
        rf"(?:of\s+)?{surname}\b",
        context,
        re.I,
    ))


def _ambiguous_surname_is_person(raw: str, context: str) -> bool:
    surname = re.escape(raw.strip())
    cues = (
        rf"\b{surname}['’]s\s+(?:administration|presidency|policy|policies|"
        rf"legacy|words|warning|example|life|death|birthday|leadership|vision|"
        rf"record|decision|message|term|time|day|era)\b",
        rf"\b{surname}\s+(?:said|wrote|warned|believed|understood|urged|led|"
        rf"fought|served|stood|sought|opposed|favored|called|became|lived|died|"
        rf"signed|issued|appointed|vetoed|told|asked|argued|was|had)\b",
        rf"\b(?:under|since|after|before|like|unlike|from)\s+{surname}\b",
        rf"\b(?:President|General|Governor)\s+{surname}\b",
        rf"\b(?:patriotic|resolute|former|late|great)\s+{surname}\b",
        rf"\b{surname}\s+(?:and|or)\s+(?:Adams|Jefferson|Madison|Jackson|Lincoln|"
        rf"Grant|Roosevelt|Wilson|Truman|Eisenhower|Kennedy|Reagan|Bush|Obama|Biden)\b",
    )
    return any(re.search(pattern, context, re.I) for pattern in cues)


def _mention_exclusion(
    target: str, raw: str, context: str, target_status: str
) -> str:
    """Return an auditable semantic exclusion for common false aliases."""
    if _is_bare_surname(target, raw) and target_status == "not_yet_president":
        return "anachronistic_bare_surname"
    if target == "George Washington" and _is_bare_surname(target, raw):
        if not _washington_is_person(context):
            return "place_or_institution"
    if target == "Ulysses S. Grant" and _is_bare_surname(target, raw):
        if raw != "Grant" or re.search(
            r"\b(?:to|we|shall|may|do|does|did|will|hereby|can|could)\s+grant\b"
            r"|\bgrant(?:ed|ing|s)\b",
            context,
            re.I,
        ):
            return "lexical_surname"
    if _is_bare_surname(target, raw):
        if _named_person_collision(target, raw, context):
            return "different_named_person"
        if _institutional_surname_use(raw, context):
            return "place_or_institution"
        surname = target.replace(".", "").split()[-1]
        if surname in AMBIGUOUS_BARE_SURNAMES and not _ambiguous_surname_is_person(raw, context):
            return "ambiguous_bare_surname"
    return ""


def extract_candidates(speeches: pd.DataFrame | None = None,
                       paragraphs: pd.DataFrame | None = None) -> pd.DataFrame:
    speeches = corpus.load() if speeches is None else speeches.copy()
    paragraphs = pd.read_parquet(corpus.DATA_DIR / "paragraphs.parquet") if paragraphs is None else paragraphs.copy()
    meta = speeches.set_index("doc_name")
    aliases = alias_registry()
    # One combined matcher keeps a full-corpus extraction practical.  A
    # sequential target loop is ~45× slower and makes resumability academic.
    alias_targets: dict[str, list[str]] = {}
    for target, variants in aliases.items():
        for variant in variants:
            alias_targets.setdefault(variant.casefold(), []).append(target)
    matcher = re.compile(r"(?<!\w)(" + "|".join(re.escape(x) for x in
                         sorted(alias_targets, key=lambda x: (-len(x), x))) + r")(?!\w)", re.I)
    records = []
    for p in paragraphs.itertuples(index=False):
        if p.doc_name not in meta.index: continue
        doc = meta.loc[p.doc_name]
        text = str(p.text)
        sent = _sentences(text)
        for match in matcher.finditer(text):
            possible = alias_targets[match.group().casefold()]
            targets = _resolve_ambiguous(possible, match.group(), str(doc.date),
                                         str(doc.president), speeches)
            for target in targets:
                    context = next((s for a,b,s in sent if a <= match.start() < b), text)
                    raw = match.group()
                    candidate_id = hashlib.sha256(f"{p.doc_name}|{p.para_idx}|{match.start()}|{target}".encode()).hexdigest()[:24]
                    role, role_exclusion = _source_role(context)
                    status = _status_on(target, str(doc.date), speeches)
                    excluded = "self_reference" if target == doc.president else role_exclusion
                    if not excluded:
                        excluded = _mention_exclusion(target, raw, context, status)
                    quote = bool(re.search(rf"[“\"](?=[^”\"]*{re.escape(raw)})|{re.escape(raw)}[^”\"]*[”\"]",
                                           context, re.I))
                    records.append({"candidate_id": candidate_id, "doc_name": p.doc_name, "para_idx": int(p.para_idx),
                        "char_start": int(match.start()), "char_end": int(match.end()), "target": target, "speaker": doc.president,
                        "speech_date": str(doc.date), "raw_mention": raw, "context": context.strip(),
                        "quotation_status": "deliberate_quotation" if quote else "narration",
                        "source_role": role, "target_status": status,
                        "excluded_reason": excluded})
    return pd.DataFrame(records).drop_duplicates("candidate_id").sort_values(["doc_name", "para_idx", "char_start"]).reset_index(drop=True)


def _fingerprint(candidates: pd.DataFrame) -> str:
    return hashlib.sha256("\n".join(candidates.candidate_id).encode()).hexdigest()


def write_candidates(frame: pd.DataFrame, root: Path = ROOT) -> None:
    root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(root / "candidates.parquet", index=False)
    (root / "manifest.json").write_text(json.dumps({"version": "invocation-v2", "corpus_fingerprint": _fingerprint(frame),
        "rubric_version": RUBRIC_VERSION,
        "annotation_standard": "exploratory AI classification; not human-validated", "unknown_cost_usd": True}, indent=2))


def next_batch(size: int = 25, root: Path = ROOT) -> list[dict]:
    if not 1 <= size <= 50:
        raise ValueError("batch size must be between 1 and 50")
    candidates = pd.read_parquet(root / "candidates.parquet")
    classified = pd.read_parquet(root / "classifications.parquet") if (root / "classifications.parquet").exists() else pd.DataFrame(columns=["candidate_id"])
    todo = candidates[(candidates.excluded_reason == "") & ~candidates.candidate_id.isin(classified.candidate_id)].head(size)
    return todo.to_dict("records")


def batch_jsonl(size: int = 25, root: Path = ROOT) -> str:
    rows = next_batch(size, root)
    candidates = pd.read_parquet(root / "candidates.parquet")
    meta = {"_meta": {"corpus_fingerprint": _fingerprint(candidates),
                      "rubric_version": RUBRIC_VERSION,
                      "expected_ids": [row["candidate_id"] for row in rows]}}
    return "\n".join(json.dumps(row) for row in [meta, *rows])


def import_batch(rows: list[dict], root: Path = ROOT) -> pd.DataFrame:
    candidates = pd.read_parquet(root / "candidates.parquet")
    if not rows: raise ValueError("batch is empty")
    if "_meta" in rows[0]:
        meta = rows.pop(0)["_meta"]
        if meta.get("corpus_fingerprint") != _fingerprint(candidates):
            raise ValueError("corpus-fingerprint drift")
        expected = meta.get("expected_ids", [])
        actual = [row.get("candidate_id") for row in rows]
        if actual != expected:
            raise ValueError("incomplete or reordered batch")
    incoming = pd.DataFrame(rows)
    required = {"candidate_id", "function", "stance", "evidence_span", "rationale"}
    missing = required - set(incoming.columns)
    if missing: raise ValueError(f"missing fields: {sorted(missing)}")
    if incoming.candidate_id.duplicated().any(): raise ValueError("duplicate candidate_id in batch")
    if not incoming.candidate_id.isin(candidates.candidate_id).all(): raise ValueError("unknown candidate_id")
    if not set(incoming["function"]).issubset(FUNCTIONS): raise ValueError("unknown function enum")
    if not set(incoming.stance).issubset(STANCES): raise ValueError("unknown stance enum")
    contexts = candidates.set_index("candidate_id").context
    for row in incoming.itertuples(index=False):
        if not str(row.evidence_span).strip() or str(row.evidence_span) not in contexts[row.candidate_id]:
            raise ValueError(f"evidence span absent from candidate context: {row.candidate_id}")
    old = pd.read_parquet(root / "classifications.parquet") if (root / "classifications.parquet").exists() else pd.DataFrame()
    if not old.empty and incoming.candidate_id.isin(old.candidate_id).any(): raise ValueError("candidate already classified")
    incoming["rubric_version"] = incoming.get("rubric_version", RUBRIC_VERSION)
    combined = pd.concat([old, incoming], ignore_index=True)
    combined.to_parquet(root / "classifications.parquet", index=False)
    return combined


def classify_candidate(row: pd.Series) -> dict:
    """Fixed-rubric exploratory AI-assisted classification from supplied evidence."""
    text = str(row["context"])
    low = text.casefold()
    ceremonial = r"\b(honor|tribute|birthday|born|died|death|memory|life of|eulogy|mourn)\b"
    policy = r"\b(policy|program|act|law|reform|plan|initiative|medicare|social security|tax|treaty)\b"
    precedent = r"\b(precedent|constitution|executive order|veto|congress|administration|office)\b"
    legacy = r"\b(legacy|inherit|footsteps|tradition|example|vision|unfinished work|carry forward|built on)\b"
    comparison = r"\b(like|unlike|compared|as .* did|since the days|reminds us|in the time of)\b"
    rivalry = r"\b(defeat|failed|failure|wrong|opponent|campaign|party|democrat|republican|blame|attack)\b"
    if re.search(ceremonial, low): function = "ceremonial/biographical"
    elif re.search(legacy, low): function = "legacy/inheritance"
    elif re.search(policy, low): function = "policy inheritance"
    elif re.search(precedent, low): function = "institutional precedent"
    elif row.get("target_status") in {"serving_or_contemporary", "not_yet_president"} and re.search(rivalry, low):
        function = "contemporary rivalry"
    elif re.search(comparison, low): function = "historical comparison"
    else: function = "other/unclear"

    positive = bool(re.search(r"\b(great|honor|courage|wisdom|leadership|success|admire|proud|good|right)\b", low))
    negative = bool(re.search(r"\b(fail|wrong|disgrace|corrupt|weak|bad|disaster|mistake|blame|dangerous)\w*\b", low))
    stance = "mixed" if positive and negative else "positive" if positive else "negative" if negative else "neutral"
    return {"candidate_id": row["candidate_id"], "function": function, "stance": stance,
            "evidence_span": text, "rationale": "Fixed-rubric classification from the supplied sentence context.",
            "rubric_version": RUBRIC_VERSION, "evidence_status": "AI-classified; not human-validated"}


def classify_all(root: Path = ROOT) -> pd.DataFrame:
    candidates = pd.read_parquet(root / "candidates.parquet")
    eligible = candidates[candidates.excluded_reason == ""].copy()
    rows = [classify_candidate(row) for _, row in eligible.iterrows()]
    labels = pd.DataFrame(rows)
    labels.to_parquet(root / "classifications.parquet", index=False)
    prompt_text = "fixed enum invocation rubric v1; classify function and stance from supplied context"
    checksums = [hashlib.sha256("\n".join(labels.candidate_id.iloc[i:i+50]).encode()).hexdigest()
                 for i in range(0, len(labels), 50)]
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({"classification_rows": len(labels), "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
                     "batch_checksums": checksums, "provenance": "Codex chat implementation pass",
                     "classification_date": "2026-07-23", "unknown_cost_usd": True})
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return labels


def audit(root: Path = ROOT) -> pd.DataFrame:
    candidates = pd.read_parquet(root / "candidates.parquet")
    labels = pd.read_parquet(root / "classifications.parquet")
    merged = candidates.merge(labels, on="candidate_id", how="left", validate="one_to_one")
    merged["audit_reason"] = ""
    merged.loc[(merged.excluded_reason == "") & merged["function"].isna(), "audit_reason"] = "unresolved"
    merged.loc[merged["function"].eq("other/unclear"), "audit_reason"] = "unclear_function"
    merged.loc[merged.stance.eq("unclear"), "audit_reason"] = "unclear_stance"
    merged.loc[merged.raw_mention.str.casefold().isin({"bush", "roosevelt", "harrison", "johnson", "adams"}),
               "audit_reason"] += "|ambiguous_surname"
    queue = merged[merged.audit_reason != ""].copy()
    queue.to_parquet(root / "audit_queue.parquet", index=False)
    return queue


def build_edges(root: Path = ROOT) -> pd.DataFrame:
    candidates, labels = pd.read_parquet(root / "candidates.parquet"), pd.read_parquet(root / "classifications.parquet")
    x = candidates.merge(labels, on="candidate_id", validate="one_to_one")
    x = x[x.excluded_reason == ""]
    x["era"] = pd.to_datetime(x.speech_date).dt.year.map(
        lambda year: next(
            (name for name, start, end in trends.ERAS if start <= int(year) <= end),
            "outside corpus",
        )
    )
    grouped = x.groupby(["speaker", "target"], as_index=False).agg(raw_mentions=("candidate_id", "size"),
        distinct_speeches=("doc_name", "nunique"), function_mix=("function", lambda s: s.value_counts().to_dict()),
        stance_mix=("stance", lambda s: s.value_counts().to_dict()),
        era_mix=("era", lambda s: s.value_counts().to_dict()))
    grouped["distinct_speakers"] = 1
    totals = corpus.load().groupby("president").doc_name.nunique()
    grouped["mentions_per_100_speeches"] = grouped.raw_mentions / grouped.speaker.map(totals) * 100
    grouped["evidence_status"] = "AI-classified; not human-validated"
    grouped.to_parquet(root / "edges.parquet", index=False)
    return grouped


def main() -> None:
    p = argparse.ArgumentParser(description="Invocation-v2 workflow")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("extract"); n = sub.add_parser("next-batch"); n.add_argument("--size", type=int, default=25)
    i = sub.add_parser("import-batch"); i.add_argument("path", type=Path)
    sub.add_parser("status"); sub.add_parser("audit"); sub.add_parser("classify-all"); sub.add_parser("build-edges")
    a = p.parse_args()
    if a.command == "extract": write_candidates(extract_candidates())
    elif a.command == "next-batch": print(batch_jsonl(a.size))
    elif a.command == "import-batch": import_batch([json.loads(x) for x in a.path.read_text().splitlines() if x.strip()])
    elif a.command == "classify-all": print(len(classify_all()))
    elif a.command == "build-edges": print(len(build_edges()))
    else:
        c = pd.read_parquet(CANDIDATES); done = pd.read_parquet(CLASSIFICATIONS) if CLASSIFICATIONS.exists() else pd.DataFrame()
        if a.command == "status":
            eligible = int((c.excluded_reason == "").sum())
            invalid = int((~done["function"].isin(FUNCTIONS) | ~done.stance.isin(STANCES)).sum()) if len(done) else 0
            print(json.dumps({"candidates": len(c), "excluded": len(c) - eligible,
                              "completed": len(done), "unresolved": max(eligible - len(done), 0),
                              "invalid": invalid, "disputed": 0}))
        else: print(json.dumps({"audit_rows": len(audit())}))


if __name__ == "__main__":
    main()
