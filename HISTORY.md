# Completed Tasks

- [x] fix-paragraph-issues-key Give paragraph_issues.parquet a real (doc_name, para_idx) key -- 2026-07-14
- [x] build-annotation-provenance-layer LLM derived-data layer, manifests, pp-annotate CLI, invocation_tone migration -- 2026-07-14
- [x] build-word-families Group word forms into one node in the word graph; audit trail in notes/word-families-audit-* -- 2026-07-16
- [x] discover-corpus-taxonomy Embedding-cluster half: embed_topics.py, k=40/k=15 assignments keyed (doc_name, para_idx); LLM taxonomy half re-backlogged as derive-corpus-taxonomy -- 2026-07-19
- [x] profile-issue-views Raw attention + era-relative + "topic of the day" flag on profiles -- 2026-07-19
- [x] derive-corpus-taxonomy LLM taxonomy discovery + crosswalk: frozen taxonomy_v1 (17 domains / 50 topics, 5 non-policy) + crosswalk_v1, 100% held-out coverage, $2.62 actual spend -- 2026-07-20
- [x] run-llm-annotation-pass Annotate 36,229 paragraphs + 1,057 speeches via Sonnet 5 batches; 100% coverage via convergence ladder, all QA gates green, $38.56 actual of $50 -- 2026-07-21
