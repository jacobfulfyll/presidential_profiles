# Memory Index

- [Project attack surface](project-attack-surface.md) — presidential_profiles is an offline static-site generator; scope security reviews to deps + data integrity
- [Dependency CVE calibration](dependency-cve-calibration.md) — how to weigh dev-only / unbounded-constraint CVEs in this repo
- [Inter-model agreement task](project-inter-model-agreement.md) — agreement.py is no-network/no-paid; annotate.py --model/--sample keep gate ordering + add resume-identity guards --force can't bypass
- [Derived-analysis-module recipe](derived-analysis-module-recipe.md) — how to actually PROVE the $0 guard + frozen-artifact integrity for agreement.py/combat.py-shaped modules (two gotchas that fake a clean result)
- [HTML render sink inventory](html-render-sink-inventory.md) — which docs/ sinks escape and which don't; plotly JSON + issue_slug are safe by construction, don't re-flag them
