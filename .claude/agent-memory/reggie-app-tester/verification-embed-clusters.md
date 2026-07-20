---
name: verification-embed-clusters
description: How to verify embed_topics.build_clusters (paragraph embedding clusters) end-to-end for presidential_profiles
metadata:
  type: reference
---

Verifying `presidential_profiles.embed_topics.build_clusters(force=True)` (paragraph clusters, model2vec + MiniBatchKMeans). Complements [[verification-workflow]] (that one is the static-site build).

- **Runs fully offline.** Model `minishlab/potion-base-8M` is in the HF cache; setting `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and it still completes — a good positive proof that no network fetch was introduced.
- **Timing:** ~3-4s wall clock for the whole real 36,229-paragraph run (embed + both k). Not slow; safe to run `force=True` live during VERIFY rather than trusting the cached parquet.
- **Deterministic under seed 42 on the REAL model:** two `force=True` runs yield byte-identical `cluster_k40`/`cluster_k15` vectors (sort by `(doc_name, para_idx)` first). Unit tests can't prove this; only the real run does.
- **Artifacts (leave them regenerated — they are what the task ships):** `data/paragraph_clusters.parquet` shape (36229, 4), cols `[doc_name, para_idx, cluster_k40, cluster_k15]`; `data/paragraph_clusters_meta.json` with `model/n_paragraphs/random_state/clusters.{k40,k15}` where each k's cluster sizes partition to 36229.
- **NPMI separates signal from sludge as designed:** top k40 clusters (npmi ~0.3-0.4) are crisp topics (fiscal/treasury, tax/budget, currency/banks/gold, navy, agriculture, health/medicare, nuclear/soviet); bottom (npmi <0.08) are filler (think/believe/right, great/national/prosperity). Blind-spot check: coinage/currency and naval-shipbuilding get their own k40 clusters; terrorism folds into a war/attacks cluster; Indian affairs does NOT surface as a distinct cluster at k40 (only "territory" bleeds into constitution/foreign-relations clusters).
