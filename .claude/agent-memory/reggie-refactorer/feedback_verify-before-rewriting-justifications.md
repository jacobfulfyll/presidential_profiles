---
name: verify-before-rewriting-justifications
description: In presidential_profiles notes/, never swap one prose justification for another without recomputing the underlying numbers first
metadata:
  type: feedback
---

When fixing a wrong *justification* in a `notes/*.md` report, recompute the numbers
from the real artifacts before writing the replacement — and prefer a replacement that
some existing test or committed artifact already pins.

**Why:** These reports' entire thesis is method honesty; the user's stated hard value is
falsification ("publishing the null is a success"). A confidently-worded but unverified
replacement is the same failure as the original wrong justification, just fresher. In the
`topic-method-comparison` fix I found the obvious replacement argument was *also* wrong —
the sweep curve does pass through the prior published value at a different term count, so
"distance to the prior figure" cannot rule anything out in either direction. The argument
that actually holds came from the *other* topic (the one that reproduced), not the one
under suspicion.

**How to apply:** Write a throwaway read-only script in the scratchpad that recomputes the
quantity across the relevant parameter (term count, stop list, era grid), run it, then
write prose that matches the printout. Never call `run()`-style entry points that persist
to `data/` — call the pure functions and assemble the pipeline yourself. State the
falsifying observation explicitly in the report rather than only the conclusion.

Related: [[run-commands]]
