"""Shared fixtures for the annotation-provenance test suite.

Every test in this suite runs OFFLINE and spends $0. Two guardrails enforce it:

* ``_no_anthropic_creds`` (autouse) deletes ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN
  so a stray real client construction fails loudly instead of leaking a paid call.
* ``redirect_annotation_dirs`` (autouse) repoints the module's write-side path
  constants at ``tmp_path`` so no test ever touches the real
  ``data/llm_annotations/`` tree. Read-only corpus paths (speeches/paragraphs
  parquet) are deliberately left pointing at the real data — tests read them, and
  the migration/coverage assertions depend on the real corpus.

The fake Anthropic client here forges SDK *outputs*; it never talks to the network.
We patch ``anthropic.Anthropic`` in place (the functions under test do
``import anthropic; anthropic.Anthropic()``), which leaves ``anthropic.types``
— needed at import time by annotate.py — untouched.
"""

from __future__ import annotations

import json
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
