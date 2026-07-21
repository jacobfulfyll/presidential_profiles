---
name: money-path-verification
description: How to structurally verify a module has no paid Anthropic path in this repo — the highest-value security check here
metadata:
  type: project
---

`presidential_profiles` makes PAID Anthropic Batches-API calls (`taxonomy.py`, `annotate.py`; the
full-corpus run cost $38.56). For any new analysis module the single highest-value security check is
**"can this construct a client?"** — grep alone is insufficient because the import chain is transitive.

**Why:** Modules routinely import `taxonomy.py` (for `_require_full_merge`, `LEGACY_ISSUES`) and
`issues.py`, either of which could drag a client-constructing path in. A regression here costs real
money, not just correctness.

**How to apply — the three-layer check, in increasing strength:**
1. `grep -rn anthropic src/` — locate every reference. Known-good state: `taxonomy.py:~820` `_client()`
   holds a **lazy** `import anthropic` inside the function body; `annotate.py:39-40` has **module-level**
   `from anthropic.types...` imports. So *importing `annotate` loads `anthropic`; importing `taxonomy`
   does not.* A module is clean iff its chain never reaches `annotate`.
2. Structural proof (do this, it is cheap and decisive):
   `env -u ANTHROPIC_API_KEY PYTHONPATH=$PWD/src arch -x86_64 .venv/bin/python -c "import sys; import presidential_profiles.<mod>; print('anthropic' in sys.modules)"`
   → must print `False`.
3. Confirm `tests/conftest.py`'s autouse `_no_anthropic_creds` (deletes `ANTHROPIC_API_KEY` /
   `ANTHROPIC_AUTH_TOKEN`) is unsubverted — grep the new test file for `setenv|ANTHROPIC|api_key`.
   A test file that re-adds a key would silently reopen the paid path.

**Calibration:** `requests`/`urllib3`/`socket` DO land in `sys.modules` for most modules here, via
`issues.py → fetch.py`'s module-level `import requests`. That is pre-existing and is a library import,
not egress — no module-level `requests.get` exists in `fetch.py`/`indices.py`/`portraits.py`. Do not
book it as a network-egress finding. See [[project-attack-surface]].
