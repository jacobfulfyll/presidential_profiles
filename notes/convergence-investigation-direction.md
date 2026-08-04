# Direction: Is the present anomalous? — the convergence investigation

*Brainstorm handoff, 2026-07-13. This is a direction doc, not a spec. It captures the
shape of a session so a planning/implementation pass can pick it up cold. Nothing here
is locked code; the "Open questions" section lists what still needs a decision.*

> **SUPERSEDED IN PART — read `TASKS.md` and `.pipeline/*/task.md` for the current plan.**
> Later the same day, task planning and an adversarial review of the convergence design
> (13 confirmed / 9 partial findings, 0 rejected) changed several decisions recorded below:
> - The **15-issue taxonomy is retired as primary** — it is blind to 28.7% of the corpus
>   (Indian affairs, Reconstruction, civil service reform), and blind in a time-varying way.
>   A corpus-native taxonomy is discovered first (`.pipeline/discover-corpus-taxonomy/`).
>   CorEx and the 15 are kept **frozen** as a bias-independent lexical cross-check.
> - **NMF and LDA are dropped** as triangulation arms (evidence in
>   `.pipeline/topic-method-comparison/task.md`). Embedding clustering is reassigned to discovery.
> - The **convergence measure was rebuilt from scratch** — the original design provably
>   manufactures a false convergence finding (rho = -0.605 on a null with no convergence in it).
>   See `.pipeline/convergence-analysis/task.md` before touching that analysis.
> - **Convergence now runs LAST.** The descriptive work (breadth/depth, issue attention,
>   combativeness, era atlas) runs first — the corpus supports description far better than
>   it supports this particular inference.
> The framing, the falsification principle (§2), the provenance design (§6), and the bias
> analysis (§7) all stand unchanged and remain load-bearing.

---

## 1. Motivation & driving question

The live question, in the user's words: **"is right now actually unique, or does it just
feel that way from inside?"** Living inside the present is the bias. A 240-year corpus is
the instrument that lets you step outside it.

Corpus-native framing: this project **cannot do history** (a formal-speech corpus doesn't
tell you what happened, only what presidents chose to say officially). But it is *uniquely*
suited to one question — **"is the present anomalous relative to the full 240-year
distribution?"** That is the question to build around. Every finding is "where does *now*
land against the whole baseline," not "what does now mean."

The formal-register point is a **feature, not a caveat**: the Miller Center corpus is the
buttoned-up register (inaugurals, annual messages, major addresses). It's the *last* place
combativeness or informality should surface. So if those things have leaked into even this
register, that's a strong signal, not a weak one. Don't apologize for the formal-register
limit — weaponize it.

## 2. Operating principle: falsification (user value — treat as load-bearing)

The user asked for this explicitly: *"I have biases for sure, but I am willing to be proven
wrong by the data and I want that to be your attitude as well."*

Applied concretely:
- **State the expected result up front** for each analysis, then genuinely try to kill it.
- **Be willing to publish the null** — "no, now is NOT unique / no, he's NOT an outlier /
  no, they're NOT interchangeable" are all acceptable, publishable outcomes.
- **Method triangulation is the disconfirmation test.** A finding must survive multiple
  measurement methods (e.g. NMF + CorEx + LLM issue labels) to count. A signal that appears
  under only one model is an artifact of that model, not a fact about history.
- **Point the same attitude at the LLM annotator itself** (see §7): design so annotator
  bias would *show up* if present, rather than assuming it's absent.

## 3. The three clusters

### C — Convergence vs. individuality  *(LOCKED — primary focus)*
Are modern presidents "not really their own people"? Do they all run the same nationalized
media agenda, distinctive in voice but interchangeable in substance?

The hook is a **paradox against the project's own existing finding**: the README reports
Trump is the *most rhetorically distinct* president in the corpus (embedding similarity),
yet his *issue emphasis vs. his era* is *small* next to Lincoln's and FDR's. **Most
distinctive voice, least distinctive agenda, at the same time.**

This reframes the whole "is now a rupture" story: maybe "now is unique" isn't a rupture in
*style* — it's a **collapse in individuality**, presidents converging on one shared agenda
while shouting it in ever more personal voices. The rupture story and the convergence story
may be the same animal from two sides.

Measurable as: **within-era issue-mix dispersion across presidents, tracked over 240 years.**
Shrinking dispersion = convergence confirmed. Flat dispersion = a beautiful idea killed
(fine, per §2). A possible *mechanism* to test alongside it: convergence via drift from
concrete proposals toward shared soft "values-talk" (see proposal-vs-values field, §5).

### B — Combativeness triad  *(scored in the same annotation pass as C)*
"Right now feels uniquely combative in my lifetime" — but the user knows the past was hot
(Sumner was caned; there was a Civil War). So the only interesting result is **where today
lands against historical peaks (1860s, 1930s)** on the same axis. Three *separate* measures,
because "they/them" pronouns and "I'll name the radical left" are different beasts:
- **Party-attack** — attacking the other party *by name* (nothing built yet).
- **Enemy-naming** — naming specific enemies/opponents (feeds on the named-entities field).
- **Zero-sum framing** — "us wins / they lose." The user's strongest energy signal
  ("incredibly interesting"); plausibly the one that surfaces in the formal register when
  nothing else does.

### A — Ruptures across 240 years  *(parked — future phase)*
Generalized (by the user) from "Trump vs. before-Trump" to **"find ALL the ruptures, rank
them, and ask: has this happened before, and when?"** — the cleaner, less presentist frame.
Key falsification move to preserve: **the pull-Trump-out test** — remove his speeches and
re-run; if the 2000s–2010s inflection survives, he's the symptom not the cause (the better
story). Techniques for later: distance-from-predecessor curves, rolling-window anomaly /
changepoint detection. Don't lose this; it's phase two, and C's convergence result may
reframe it before it's even built.

## 4. The one-pipeline insight

Better topic modeling, running multiple methods, and LLM annotation are **not three
projects — they're one dependency chain serving C.** C's convergence claim is only as
trustworthy as the topic labels beneath it (currently half-sludge — Discovered 1–7 are part
real themes, part procedural noise like "secretary, report, department, senate"). So C
*requires* a topic-model upgrade; "run multiple methods and compare" *is* the falsification
test for C; and LLM issue-labeling is simply *one more method* to triangulate against the
statistical models. One pipeline.

## 5. LLM enrichment schema

Every field is a **calibration debt** — an unvalidated annotation is a liability, not an
asset (§2, §7). Keep the schema tight.

**Core fields (build these):**
- **Per-speech issue labels** — LLM labeling as one triangulation method for C, against
  NMF/CorEx.
- **Speech-type / occasion tags** — inaugural / SOTU / crisis / eulogy / campaign / etc.
  Also the primary control for the speech-type confound (§9).
- **Audience + medium** — written-to-Congress / radio / TV / rally. Sharpens the register
  story and is the other half of the confound control (audience drives register).
- **Named entities** — people, nations, institutions named. Infrastructure for B
  (enemy-naming, party-attack); relatively factual, low-bias to extract.
- **B-triad scores** — party-attack, enemy-naming, zero-sum (see §3-B).
- **Proposal-density vs. values-talk** — concrete asks vs. abstract appeals.
  **CONFIRMED core (user decision, 2026-07-13).** Rationale: may *explain* C (convergence
  as everyone drifting to the same values-talk), turning C from a bare correlation into a
  mechanism.

**Later list (only if cheap in the same pass):**
- Allusions / quotations (Bible, Constitution, Declaration, prior presidents) — the
  "canon-forming" thread; nearly free to grab alongside named entities.
- Crisis-context flags; domestic-vs-foreign focus. Both drift toward the history side the
  user explicitly deprioritized ("others can fill in the history portion").

## 6. Provenance design (hard requirement)

*"It should be clear what data is LLM added."*
- LLM-derived fields live in a **separate layer / separate parquet files**, never silently
  merged into the Miller Center source (`data/speeches.parquet` stays pristine).
- Each derived field **stamped with model + prompt version + date**.
- UI **visually marks** derived data as LLM-generated.
- Design so the pass is **re-runnable with a different model** and the outputs diffable —
  which doubles as a falsification tool (inter-model agreement, §7).

## 7. Bias & validation design (falsification pointed at the annotator)

Three failure modes, in rough order of scariness for *this* corpus:

1. **Anachronism proper** — 19th-century political speech was florid and brutal by modern
   standards; period-normal invective reads as extreme to a 2026 model. "Combative" needs an
   *era-anchored* yardstick, not a modern one.
2. **Prior-knowledge contamination (worst, and corpus-specific)** — the model *knows who
   these people are* and can't un-know that a passage is the Gettysburg Address or that it's
   reading Trump. Risk: it scores *reputation* instead of *text*. Worse, the model's priors
   are strongest exactly at the famous peaks — which is exactly where the headline findings
   will live.
3. **Rubric drift** — over a 1,057-speech run, the model's internal "7 out of 10" wanders.

Mitigations (measure the bias, don't pretend it's gone):
- **Era-anchored rubrics** — few-shot calibration examples drawn from *each period*.
- **Ordinal / pairwise scoring** — "is A more zero-sum than B?" is far more robust than
  "rate A a 6.4"; within-era ranking sidesteps much of the anachronism drift.
- **Mask speaker + date where feasible** — but honestly: this only helps the *obscure
  middle* (~900 forgettable speeches). Treat the famous ~30 as known-compromised.
- **Inter-model agreement** as a cheap smoke test.
- **Hand-graded, cross-era calibration set — DEFERRED (user decision, 2026-07-13).**
  This run trusts the LLM annotator fully, with no human ground-truth pass up front. A
  hand-graded truth run happens in a **future phase** and gets compared against this run's
  results retroactively. For that comparison to be meaningful, this run's annotations must
  be preserved exactly as produced — which the provenance stamping (§6: model + prompt
  version + date, separate layer) already guarantees. Findings published from this run
  should carry a caveat that annotations are LLM-only, pending human validation.
- **Publish disagreement rather than hiding it** — where models disagree about how
  combative 1850 was, the confidence band gets *wide*. Annotation uncertainty literally
  becomes part of the confidence bands (ties to §8), and that honesty is itself a finding.

Framing: we can't eliminate these biases; we can *measure* them and design so they'd show up
if present.

## 8. Presentation upgrades

Current trend charts are "quite smoothed" and don't teach you about specific presidents. Wanted:
- **Confidence bands on the topic/issue trend charts** (not clean curves that hide
  variance — and see §7, annotation disagreement can feed these).
- **Per-president dots on those same topic/trend charts** — each president visible as a
  point against the band, not absorbed into a smoothed line.
- **Less smoothing** generally.
- On profiles: show **both raw issue attention AND era-relative distinctiveness**, with a
  **"this was just the topic of the day" flag** for issues that aren't above/below the era.
- **Sample-size honesty** — a president with 8 speeches is not the same as one with 40; don't
  present them with equal confidence.

## 9. Known confounds

- **Speech-type mix shifting over 240 years** is the thing most likely to *fake* a
  convergence signal: early corpus is dense written annual messages; modern corpus is
  inaugurals / crisis / TV addresses. Modern presidents may look "converged" partly because
  they're all giving the *same kinds of speeches to the same audiences*, while Lincoln/FDR
  span a wider range of occasions. **Control via the occasion + medium tags (§5)** before
  believing any convergence curve.

## 10. Open questions for planning

- Confirm whether **proposal-vs-values** stays in the core schema (recommended yes — §5).
- Which **topic-model methods** to triangulate (NMF, CorEx, LDA, LLM labels, embeddings-based
  clustering — pick the set that makes the disconfirmation test meaningful).
- **Annotation model(s) and budget** — how many models for the inter-model agreement check;
  cost of the full 1,057-speech (or 36k-paragraph) pass.
- Unit of annotation: whole speeches vs. the existing ~36k paragraph chunks.
- *(Future phase, not this run:)* design of the **hand-graded truth run** — sampling per era,
  blind-grading protocol, and how its results get compared against this run's stored
  LLM annotations.
