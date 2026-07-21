# Completed Tasks

- [x] fix-paragraph-issues-key Give paragraph_issues.parquet a real (doc_name, para_idx) key -- 2026-07-14
- [x] build-annotation-provenance-layer LLM derived-data layer, manifests, pp-annotate CLI, invocation_tone migration -- 2026-07-14
- [x] build-word-families Group word forms into one node in the word graph; audit trail in notes/word-families-audit-* -- 2026-07-16
- [x] discover-corpus-taxonomy Embedding-cluster half: embed_topics.py, k=40/k=15 assignments keyed (doc_name, para_idx); LLM taxonomy half re-backlogged as derive-corpus-taxonomy -- 2026-07-19
- [x] profile-issue-views Raw attention + era-relative + "topic of the day" flag on profiles -- 2026-07-19
- [x] derive-corpus-taxonomy LLM taxonomy discovery + crosswalk: frozen taxonomy_v1 (17 domains / 50 topics, 5 non-policy) + crosswalk_v1, 100% held-out coverage, $2.62 actual spend -- 2026-07-20
- [x] run-llm-annotation-pass Annotate 36,229 paragraphs + 1,057 speeches via Sonnet 5 batches; 100% coverage via convergence ladder, all QA gates green, $38.56 actual of $50 -- 2026-07-21
- [x] combativeness-over-time Where today lands vs the 1860s and 1930s: present-era party_attack 4.81x [2.35, 14.32] the Civil War era on annual messages; zero_sum NOT distinguishable from the founding; 1930s peak is two campaigns, not the presidency -- 2026-07-21
- [x] issue-attention-over-time Topic lifecycles across 240 years: attention.py + topic_lifecycles.parquet (201 rows, byte-reproducible) + notes/attention-findings-v1.md; 17 died / 14 born / 12 persistent / 7 revived -- 2026-07-21
- [x] topic-method-comparison LLM vs CorEx triangulation: coherence + naming registry, per-issue/per-era agreement, rename-vs-death; pre-registered low-coherence prediction FALSIFIED and published -- 2026-07-21
- [x] breadth-depth-register Breadth/depth/register trends under 3 genre treatments + 3 taxonomies; headline mostly falsified, null published -- 2026-07-21
- [x] inter-model-agreement-check Opus 4.8 second pass on persisted 25% sample (266 speeches, 94.7% para coverage at pre-registered bail); agreement_v1.parquet + notes/agreement-report-v1.md; party_attack kappa 0.0 pre-1848 = predicted anachronism made visible; $30.52 actual -- 2026-07-22
- [x] era-atlas Era fingerprints across 3 grains + raw/detrended similarity + data-driven periodization + LLM portraits ($0.06): present era detrended-nearest to Civil War & Reconstruction (0.315, mutual); 6/8 canonical boundaries recovered at bin grain; opponents leave-one-out clean -- 2026-07-21
