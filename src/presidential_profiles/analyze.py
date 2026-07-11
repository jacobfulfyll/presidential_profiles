"""Run the full analysis pipeline: stats, topics, embeddings, and all charts.

Usage:  uv run pp-analyze [--force]
"""

import argparse

from . import corpus, figures, rhetoric, similarity, topics, trends


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full analysis pipeline")
    parser.add_argument(
        "--force", action="store_true", help="recompute cached intermediate tables"
    )
    args = parser.parse_args()

    print("Loading corpus...")
    df = corpus.load()
    print(f"  {len(df):,} speeches, {df['word_count'].sum():,} words, "
          f"{df['year'].min()}-{df['year'].max()}")

    print("Computing per-speech linguistic stats (spaCy)...")
    stats = rhetoric.build_stats(df, force=args.force)

    print("Fitting topic model (TF-IDF + NMF)...")
    doc_topics, topic_terms = topics.build_topics(df, force=args.force)

    print("Embedding speeches (model2vec)...")
    emb = similarity.build_embeddings(df, force=args.force)
    sim = similarity.similarity_matrix(emb)

    print("Computing keyword trends...")
    kw = trends.keyword_trends(df)
    distinctive = trends.distinctive_terms(df)

    print("Rendering figures...")
    figures.modal_verbs(stats)
    figures.pronouns(stats)
    figures.readability(stats)
    figures.topics_small_multiples(doc_topics, topic_terms)
    figures.keyword_small_multiples(kw)
    figures.distinctive_terms_chart(distinctive)
    figures.president_map(emb)
    figures.similarity_heatmap(sim)

    print("Done. See outputs/figures and outputs/interactive.")


if __name__ == "__main__":
    main()
