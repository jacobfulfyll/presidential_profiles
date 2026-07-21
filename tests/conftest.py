"""Shared fixtures for the test suite, in two independent halves.

**Annotation provenance** (top half) — global guardrails, a fake Anthropic SDK,
and forged run state for ``annotate.py`` / ``llm_annotations.py``.

**register.py synthetic builders** (bottom half, below the second banner) —
``register_taxonomy`` / ``register_corpus`` / ``register_panel``, which build
hand-countable stand-ins for the seven on-disk tables and for a rolled-up
``SpeechPanel``. They live here rather than in a helper module because ``tests/``
is not a package, so a test module cannot import from a sibling. They construct
no client, touch no path constant, and register no autouse fixture, so they
cannot affect the annotation tests above.

Every test in this suite runs OFFLINE and spends $0. Three guardrails enforce it:

* ``_no_anthropic_creds`` (autouse) deletes ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN
  so a stray real client construction fails loudly instead of leaking a paid call.
* ``redirect_annotation_dirs`` (autouse) repoints the module's write-side path
  constants at ``tmp_path`` so no test ever touches the real
  ``data/llm_annotations/`` tree. Read-only corpus paths (speeches/paragraphs
  parquet) are deliberately left pointing at the real data — tests read them, and
  the migration/coverage assertions depend on the real corpus.
* ``_frozen_data_artifacts`` (autouse) fails any test that WRITES to a committed
  data artifact several tests read live, which would otherwise make those tests
  silently order-dependent.

Because ``redirect_annotation_dirs`` moves ``ANNOTATIONS_DIR``, anything that
must read a real frozen artifact reads it by absolute worktree path instead
(see ``test_register_taxonomy_index.py``).

The fake Anthropic client here forges SDK *outputs*; it never talks to the network.
We patch ``anthropic.Anthropic`` in place (the functions under test do
``import anthropic; anthropic.Anthropic()``), which leaves ``anthropic.types``
— needed at import time by annotate.py — untouched.
"""

from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import pytest

from presidential_profiles import annotate as A
from presidential_profiles import llm_annotations as ann

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# global guardrails (autouse)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_anthropic_creds(monkeypatch):
    """No credentials in the environment for any test. A real client
    construction must fail loudly rather than silently authenticate."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


# Committed, provenance-stamped artifacts that tests READ live off `data/`:
#   * `topic_display_names.json` — `profiles_site.DISCOVERED_LABELS` is
#     IMPORT-TIME state read from this file, and
#     `test_profiles_site_labels_come_from_the_names_file` compares that snapshot
#     against a fresh read. That comparison is correct today only because every
#     names-file write in the suite is redirected at `tmp_path`.
#   * `issues_meta.json` — `TestCheckedInIssuesMeta` regression-checks the
#     report's §3/§4 coherence figures against it.
#   * `llm_annotations/taxonomy_v1.json` and `llm_annotations/crosswalk_v1.json`
#     — read live by `test_triangulate.py` (the 50 canonical level-2 names, the
#     16 crosswalk keys) and by `test_taxonomy.py`. They are also frozen,
#     provenance-stamped outputs of a PAID run, so a test that rewrote one would
#     be destroying an artifact that cannot be cheaply regenerated.
# A test that wrote to any of these would make the readers order-dependent:
# passing or failing according to what ran before them. This turns the
# convention into an enforced invariant that fails in the test that broke it,
# not downstream.
FROZEN_DATA_ARTIFACTS = (
    "topic_display_names.json",
    "issues_meta.json",
    "llm_annotations/taxonomy_v1.json",
    "llm_annotations/crosswalk_v1.json",
)


def _artifact_digests() -> dict[str, str]:
    from presidential_profiles.corpus import DATA_DIR

    out = {}
    for name in FROZEN_DATA_ARTIFACTS:
        path = DATA_DIR / name
        if path.exists():
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


@pytest.fixture(autouse=True)
def _frozen_data_artifacts():
    """Fail the test that mutates a committed data artifact, not its successor.

    Digests rather than raw bytes so a failure reads as the filename plus a
    short hash, not a 10 KB byte diff.
    """
    before = _artifact_digests()

    yield

    after = _artifact_digests()
    changed = sorted(k for k in before if after.get(k) != before[k])
    assert not changed, (
        f"committed data artifact(s) modified by this test: {changed}. They are "
        f"read-only in this suite — redirect the write at tmp_path (monkeypatch "
        f"topic_quality.NAMES_PATH / issues.ISSUES_META_PATH). Leaving one "
        f"modified makes every later test that reads it order-dependent."
    )


@pytest.fixture(autouse=True)
def redirect_annotation_dirs(tmp_path, monkeypatch):
    """Repoint every write-side path constant at tmp_path. Both modules read
    these off the ``llm_annotations`` module at call time, so patching here
    covers annotate.py's ``ann.RUNS_DIR`` / ``ann.write_manifest`` usage too."""
    root = tmp_path / "llm_annotations"
    monkeypatch.setattr(ann, "ANNOTATIONS_DIR", root)
    monkeypatch.setattr(ann, "MANIFESTS_DIR", root / "manifests")
    monkeypatch.setattr(ann, "RUNS_DIR", root / "runs")
    monkeypatch.setattr(ann, "INVOCATION_TONE_PATH", root / "invocation_tone.parquet")
    return root


# ---------------------------------------------------------------------------
# fake Anthropic SDK (forged outputs, zero network)
# ---------------------------------------------------------------------------


class FakeResult:
    """One line of a batches.results() stream. Only needs ``to_json`` — that is
    all _fetch_results calls on each streamed result."""

    def __init__(self, payload: dict):
        self._payload = payload

    def to_json(self, indent=None) -> str:
        return json.dumps(self._payload)


class FakeBatches:
    def __init__(self, *, results_source=None, create_batch=None, retrieve_batch=None):
        self._results_source = results_source
        self._create_batch = create_batch
        self._retrieve_batch = retrieve_batch

    def results(self, batch_id):
        src = self._results_source
        if callable(src):
            return src(batch_id)
        return iter(src or [])

    def create(self, requests):
        if self._create_batch is None:
            raise AssertionError("create() called but no create_batch configured")
        return self._create_batch

    def retrieve(self, batch_id):
        if self._retrieve_batch is None:
            raise AssertionError("retrieve() called but no retrieve_batch configured")
        return self._retrieve_batch


class FakeMessages:
    def __init__(self, batches):
        self.batches = batches


class FakeAnthropic:
    def __init__(self, batches):
        self.messages = FakeMessages(batches)


@pytest.fixture
def patch_anthropic(monkeypatch):
    """Install a FakeAnthropic whose batches behaviour the test configures.

    Patches the attribute on the real module so ``import anthropic`` inside the
    functions under test resolves to the same (patched) module object."""
    import anthropic

    def _install(*, results_source=None, create_batch=None, retrieve_batch=None):
        batches = FakeBatches(
            results_source=results_source,
            create_batch=create_batch,
            retrieve_batch=retrieve_batch,
        )
        client = FakeAnthropic(batches)
        monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: client)
        return client

    return _install


@pytest.fixture
def anthropic_construction_bomb(monkeypatch):
    """Make ``anthropic.Anthropic(...)`` explode if it is ever constructed.

    Used to prove a money-gate refuses BEFORE any client is built. A SystemExit
    / ValueError from the gate must surface instead of this AssertionError."""
    import anthropic

    def _bomb(*a, **k):
        raise AssertionError("anthropic.Anthropic() was constructed — a gate failed to fire first")

    monkeypatch.setattr(anthropic, "Anthropic", _bomb)
    return _bomb


# ---------------------------------------------------------------------------
# forged run state / results builders
# ---------------------------------------------------------------------------


def _succeeded_line(cid: str, annotations, usage: dict) -> dict:
    """A succeeded paragraph-unit result line as ingest expects to parse it."""
    text = json.dumps({"annotations": [{"para_idx": i, "label": lbl} for i, lbl in annotations]})
    return {
        "custom_id": cid,
        "result": {
            "type": "succeeded",
            "message": {"usage": usage, "content": [{"type": "text", "text": text}]},
        },
    }


def _errored_line(cid: str, error_type: str) -> dict:
    return {
        "custom_id": cid,
        "result": {"type": "errored", "error": {"type": error_type}},
    }


@pytest.fixture
def forge_run(redirect_annotation_dirs):
    """Write a run directory (state.json, requests_index.json, results.jsonl)
    under the redirected RUNS_DIR and return the run_id.

    ``specs``/``model``/``corpus_fingerprint`` go into state so cmd_ingest never
    reaches for the real corpus — the ingest tests stay fully offline."""

    def _make(
        run_id: str,
        *,
        index: dict,
        result_lines: list[dict] | None = None,
        results_text: str | None = None,
        batch_id: str = "batch_test",
        results_batch_id: str | None = "batch_test",
        submitted_custom_ids: list[str] | None = None,
        n_requests: int | None = None,
        specs: list[str] | None = None,
        model: str = "claude-sonnet-5",
    ) -> Path:
        run_dir = ann.RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "requests_index.json").write_text(json.dumps(index, indent=2, sort_keys=True))

        if results_text is not None:
            (run_dir / "results.jsonl").write_text(results_text)
        elif result_lines is not None:
            (run_dir / "results.jsonl").write_text(
                "".join(json.dumps(l) + "\n" for l in result_lines)
            )

        state = {
            "batch_id": batch_id,
            "results_batch_id": results_batch_id,
            "model": model,
            "specs": specs if specs is not None else ["placeholder"],
            "submitted_custom_ids": (
                submitted_custom_ids
                if submitted_custom_ids is not None
                else sorted(index)
            ),
            "n_requests": n_requests if n_requests is not None else len(index),
            "corpus_fingerprint": {"n_speeches": 1, "n_paragraphs": 1, "doc_name_sha256": "x"},
        }
        (run_dir / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True))
        return run_dir

    _make.succeeded_line = staticmethod(_succeeded_line)
    _make.errored_line = staticmethod(_errored_line)
    return _make


class Args:
    """Minimal stand-in for the argparse.Namespace the cmd_* functions read."""

    def __init__(self, **kw):
        self.run_id = None
        self.spec = None
        self.limit = None
        self.force = False
        self.yes = False
        self.max_cost_usd = A.MAX_COST_USD
        self.count_tokens = False
        self.refetch = False
        # New surfaces (pilot sampling, sealing, qa report path). The cmd_*
        # functions read these via getattr with a default so the pre-existing
        # tests still pass, but new tests set them explicitly.
        self.pilot = False
        self.resubmit_sealed = False
        self.out = None
        self.__dict__.update(kw)


@pytest.fixture
def args():
    return Args


# ---------------------------------------------------------------------------
# register.py synthetic builders
# ---------------------------------------------------------------------------
#
# register.py keeps load and compute apart, so every function under test takes
# DataFrames. These builders make tiny hand-countable stand-ins for the seven
# on-disk tables (and for the rolled-up SpeechPanel) so no test ever reads the
# real 36k-row corpus. They live in conftest because `tests/` is not a package
# — a test module cannot import a helper from a sibling test module.


# A five-topic / three-domain taxonomy with the SAME shape as taxonomy_v1: two
# non-policy domains, and one level-2 name containing a lowercase small word
# ("the War on Terror") so the case-variant resolution path has a real target.
REGISTER_TAXONOMY = {
    "level1": [
        {"name": "Economy", "definition": "d", "kind": "policy"},
        {"name": "Security", "definition": "d", "kind": "policy"},
        {"name": "Ceremonial", "definition": "d", "kind": "non-policy"},
        {"name": "Personal Narrative", "definition": "d", "kind": "non-policy"},
    ],
    "level2": [
        {"name": "Jobs & Wages", "definition": "d", "level1": "Economy"},
        {"name": "Trade & Tariffs", "definition": "d", "level1": "Economy"},
        {"name": "the War on Terror", "definition": "d", "level1": "Security"},
        {"name": "Holidays & Tributes", "definition": "d", "level1": "Ceremonial"},
        {"name": "Reflection on Office", "definition": "d", "level1": "Personal Narrative"},
    ],
}

_STYLE_DEFAULTS = {
    "n_tokens": 200.0,
    "n_sents": 10.0,
    "i_count": 2.0,
    "we_count": 8.0,
    "fk_grade": 9.0,
    "n_words": 200.0,
}


def _build_register_corpus(specs):
    """The seven register.py input tables, from a compact per-speech spec.

    Each spec is ``{doc_name, year, president, speech_type, paras: [...]}`` where
    each paragraph is ``{legacy: [issue names], topics: [level-2 labels],
    pv: proposal_values, words: int}``. Style/stat columns take the defaults
    above unless overridden on the spec.
    """
    import pandas as pd

    from presidential_profiles.register import STYLE_MARKERS
    from presidential_profiles.taxonomy import LEGACY_ISSUES

    para_rows, issue_rows, ann_rows = [], [], []
    speech_rows, type_rows, stat_rows, marker_rows = [], [], [], []

    for spec in specs:
        doc = spec["doc_name"]
        year = spec["year"]
        president = spec.get("president", "P")
        for idx, para in enumerate(spec["paras"]):
            words = para.get("words", 100)
            para_rows.append({
                "doc_name": doc,
                "para_idx": idx,
                "text": "w " * words,
                "word_count": words,
            })
            fired = set(para.get("legacy", ()))
            issue_rows.append({
                "doc_name": doc,
                "para_idx": idx,
                "president": president,
                "year": para.get("year", year),
                **{issue: issue in fired for issue in LEGACY_ISSUES},
            })
            ann_rows.append({
                "doc_name": doc,
                "para_idx": idx,
                "topics": list(para.get("topics", ())),
                "proposal_values": para.get("pv", "neither"),
            })
        speech_rows.append({"doc_name": doc, "president": president, "year": year})
        type_rows.append({"doc_name": doc, "speech_type": spec["speech_type"]})
        stat_rows.append({
            "doc_name": doc,
            **{k: float(spec.get(k, v)) for k, v in _STYLE_DEFAULTS.items()
               if k != "n_words"},
        })
        marker_rows.append({
            "doc_name": doc,
            "n_words": float(spec.get("n_words", _STYLE_DEFAULTS["n_words"])),
            **{m: float(spec.get(m, 4.0)) for m in STYLE_MARKERS},
        })

    return types.SimpleNamespace(
        paragraphs=pd.DataFrame(para_rows),
        issues=pd.DataFrame(issue_rows),
        annotations=pd.DataFrame(ann_rows),
        speeches=pd.DataFrame(speech_rows),
        speech_annotations=pd.DataFrame(type_rows),
        stats=pd.DataFrame(stat_rows),
        markers=pd.DataFrame(marker_rows),
    )


_SCALAR_DEFAULTS = {
    "n_paragraphs": 4.0,
    "para_words": 400.0,
    "llm_labels": 8.0,
    "legacy_labels": 4.0,
    "llm_non_policy_paras": 0.0,
    "llm_zero_paras": 0.0,
    "legacy_zero_paras": 0.0,
    "pv_proposal": 2.0,
    "pv_values": 1.0,
    "pv_mixed": 0.0,
    "pv_neither": 1.0,
    "n_tokens": 400.0,
    "n_sents": 20.0,
    "i_count": 2.0,
    "we_count": 8.0,
    "fk_x_tokens": 3600.0,
    "n_words": 400.0,
}


def _build_register_panel(rows, topic_counts=None, topic_names=None):
    """A SpeechPanel built directly from per-speech scalars.

    Bypasses `build_speech_panel` on purpose: the measure / bootstrap / trend
    tests need exact, hand-computable sufficient statistics, and building them
    through a paragraph frame would only obscure where a number came from.
    """
    import numpy as np
    import pandas as pd

    from presidential_profiles.register import STYLE_MARKERS
    from presidential_profiles.taxonomy import ERA_SPAN

    speeches = pd.DataFrame([
        {
            "doc_name": r["doc_name"],
            "president": r.get("president", "P"),
            "year": r["year"],
            "speech_type": r["speech_type"],
            "era": (r["year"] // ERA_SPAN) * ERA_SPAN,
        }
        for r in rows
    ])
    scalars = pd.DataFrame([
        {
            **{k: float(r.get(k, v)) for k, v in _SCALAR_DEFAULTS.items()},
            **{m: float(r.get(m, 4.0)) for m in STYLE_MARKERS},
        }
        for r in rows
    ])
    if topic_counts is None:
        topic_counts = {"legacy15": np.ones((len(rows), 3), dtype=float)}
    if topic_names is None:
        topic_names = {
            name: tuple(f"t{i}" for i in range(matrix.shape[1]))
            for name, matrix in topic_counts.items()
        }
    from presidential_profiles.register import SpeechPanel

    return SpeechPanel(
        speeches=speeches,
        scalars=scalars,
        topic_counts={k: np.asarray(v, dtype=float) for k, v in topic_counts.items()},
        topic_names=topic_names,
    )


@pytest.fixture
def register_taxonomy():
    """A fresh copy of the synthetic taxonomy dict (tests mutate it)."""
    import copy

    return copy.deepcopy(REGISTER_TAXONOMY)


@pytest.fixture
def register_corpus():
    return _build_register_corpus


@pytest.fixture
def register_panel():
    return _build_register_panel

# synthetic corpus builder (combat.py suite)
# ---------------------------------------------------------------------------
#
# ``combat.load_frame`` accepts all five of its inputs as injected DataFrames,
# which is what lets the whole module — merge guards, era mapping, bootstrap,
# genre standardization — run on a dozen hand-authored rows instead of the real
# 36,229-row corpus (the repo's testing convention, CLAUDE.md). These two
# helpers build those frames from a compact spec so each test states only the
# structure it actually cares about.


def _combat_inputs(specs: list[dict]) -> dict:
    """Build the five injectable frames from a list of speech specs.

    Each spec is ``{"doc", "year", "type", "paras": [...]}``; each paragraph is
    a dict of flag overrides plus optional ``word_count`` / ``text`` /
    ``adversaries`` (``[(entity, type)]``, stance=adversarial) / ``entities``
    (``[(entity, type, stance)]`` for non-adversarial stances).
    """
    import pandas as pd

    from presidential_profiles import combat as C

    speeches, speech_anns, paragraphs, annotations, entities = [], [], [], [], []
    for spec in specs:
        doc = spec["doc"]
        speeches.append(
            {
                "doc_name": doc,
                "president": spec.get("president", "A President"),
                "party": spec.get("party", "Whig"),
                "date": f"{spec['year']}-01-01",
                "year": spec["year"],
                "title": spec.get("title", f"Address {doc}"),
            }
        )
        speech_anns.append(
            {
                "doc_name": doc,
                "speech_type": spec.get("type", C.SOTU_TYPE),
                "audience": spec.get("audience", "public"),
                "medium": spec.get("medium", "written"),
            }
        )
        for i, para in enumerate(spec["paras"]):
            paragraphs.append(
                {
                    "doc_name": doc,
                    "para_idx": i,
                    "text": para.get("text", f"{doc} paragraph {i}. " + "word " * 30),
                    "word_count": para.get("word_count", 40),
                }
            )
            annotations.append(
                {
                    "doc_name": doc,
                    "para_idx": i,
                    "run_id": spec.get("run_id", "run-test"),
                    **{f: bool(para.get(f, False)) for f in C.FLAGS},
                }
            )
            for name, kind in para.get("adversaries", []):
                entities.append(
                    {
                        "doc_name": doc,
                        "para_idx": i,
                        "entity": name,
                        "type": kind,
                        "stance": "adversarial",
                    }
                )
            for name, kind, stance in para.get("entities", []):
                entities.append(
                    {
                        "doc_name": doc,
                        "para_idx": i,
                        "entity": name,
                        "type": kind,
                        "stance": stance,
                    }
                )

    return {
        "annotations": pd.DataFrame(annotations),
        "speech_annotations": pd.DataFrame(speech_anns),
        "paragraphs": pd.DataFrame(paragraphs),
        "speeches": pd.DataFrame(speeches),
        "entities": pd.DataFrame(
            entities, columns=["doc_name", "para_idx", "entity", "type", "stance"]
        ),
    }


def _combat_paras(n: int, n_flagged: int = 0, flag: str = "party_attack", **extra):
    """``n`` paragraphs of which the first ``n_flagged`` carry ``flag``."""
    return [{flag: i < n_flagged, **extra} for i in range(n)]


@pytest.fixture
def combat_inputs():
    """Factory for the five frames ``combat.load_frame`` accepts."""
    return _combat_inputs


@pytest.fixture
def combat_paras():
    """Factory for a run of paragraphs with a known flag count."""
    return _combat_paras
