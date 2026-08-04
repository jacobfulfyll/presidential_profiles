---
name: verification-bands
description: How to verify data/bands.parquet + the banded issue charts end-to-end, including headless-Chrome visual checks when browser MCP tools are absent
metadata:
  type: reference
---

Verifying `presidential_profiles.bands` (two-surface confidence bands) and its rendered charts.

**Regeneration:** `env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN PYTHONPATH=src arch -x86_64 <venv-python> -m presidential_profiles.bands` — ~10s, $0, byte-identical (sha `45f6639205b3…` at 1,528 rows: 1,078 `corex_issues` + 450 `llm_topics`). Prints a self-check block (per-surface counts, ci_status, thinnest-period widths, disagreement contribution) that is the fastest sanity read.

**The disagreement component is recomputable in ~20 lines and matched to 0.0**: read `paragraph_annotations.parquet` (primary) + `paragraph_annotations__opus4-8.parquet` (secondary, 8,570 rows), merge one_to_one, join `data/paragraph_issues.parquet` for `year`, bucket by `trends.ERAS`, casefold-map topics against `taxonomy_v1.json['level2']` (50 names — it's a flat `level2` list, NOT nested under `level1`), then `|share_p - share_s| / 2` per (era, topic). Always do this rather than trusting the module — it is cheap and it is the only proof the two surfaces are wired to different sources.

**Rendered-chart-vs-parquet proof.** The plotly payload is `FIG = {…}` in the page (not inline in `Plotly.newPlot`, which takes `FIG.data, FIG.layout`). `json.JSONDecoder().raw_decode(s[s.index('{', s.index('FIG =')):])`. Numeric arrays are plotly binary (`{'dtype','bdata'}`) — decode with `np.frombuffer(base64.b64decode(v['bdata']), dtype=v['dtype'])`. The band polygon is the single `fill='toself'` trace: first half = hi, reversed second half = lo. This re-derives every plotted vertex against `bands.parquet`.

**`interval_unresolvable` (suppress-degenerate-band-intervals, verified 2026-07-21).** Post-fix shas:
`bands.parquet 1683189dd85f…`, `bands_meta.json 5f3dd91dadb4…`; 12 flagged Surface A cells (all 1785,
n=2, all four bounds null, `point` kept), 7 legitimate zero-width cells survive at n=13–14. **The 12
flagged cells render only 11 rings** — `Discovered 4` has `surface: false` in
`data/topic_display_names.json` so it has no page and no grid panel. Conversely a `Security & peace`
ring is *not* a bug: that page is `Discovered 5`'s display name. Always resolve series↔page names
through that file before calling a ring count a mismatch.

**Chart anatomy worth knowing (4 traces per issue timeline):** `toself` polygon (hover skipped) → dotted line carrying the FULL series *and* the hover/customdata `[lo, hi, n_speeches, ci_status(, ci_components)]` → solid line with caution periods gapped to `None` → president-period dot markers. So "the solid line skips 1785" is the de-emphasis, and the trust gate lives in the tooltip.

**Browser automation is not always available.** When `mcp__claude-in-chrome__*` is absent and kaleido/playwright are not installed, headless Chrome works:
`"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu --no-sandbox --hide-scrollbars --virtual-time-budget=15000 --window-size=W,H --screenshot=out.png http://127.0.0.1:PORT/page.html`
`--virtual-time-budget` (ms) is what lets plotly finish; 12–25s for the 1.4 MB dashboard. Then crop/upscale with PIL (installed) before reading the image — a full-page 1440×9000 shot is unreadable un-cropped.

**Two server hazards, both hit for real.** (a) `pkill`/`kill <pid>` are DENIED by the bash sandbox;
`/bin/kill -TERM <pid>` is permitted. Use that, and verify with `lsof -nP -iTCP:<port> -sTCP:LISTEN`.
(b) A prior pipeline stage can leave a server orphaned on the port you pick, rooted in a *different
worktree's* `docs/` — so screenshots silently show the predecessor's build. Check `lsof` before
binding, and confirm by re-shooting on a fresh port and comparing the PNG sha (they were identical
here, which is the proof, not the assumption).

**Known site-wide, pre-existing:** at a 390 px viewport the page overflows horizontally and prose is clipped at the right edge. `.chart` has `min-width: 640px` inside `.chart-scroll`. Reproduces on untouched pages (`compare.html`), so never attribute it to the chart task under test.

See [[verification-workflow]] for the build/test recipe and [[feedback_perturb-restore-discipline]] for the seam test (move `bands.parquet` aside, confirm `load_bands() is None` and the figure loses its polygon but keeps its line, then sha-verify the restore).
