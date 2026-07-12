"""President similarity via modern static embeddings (model2vec).

Replaces the 2019 ELMo + TensorFlow-1 pipeline. model2vec's potion models are
distilled static embeddings: no torch, near-instant inference, and strong
enough for document-level similarity.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from model2vec import StaticModel
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

from .corpus import DATA_DIR, load, president_order

EMB_PATH = DATA_DIR / "president_embeddings.parquet"
SPEECH_EMB_PATH = DATA_DIR / "speech_embeddings.parquet"
ADJ_PATH = DATA_DIR / "president_embeddings_adjusted.parquet"

MODEL_NAME = "minishlab/potion-base-8M"

# Window for the era baseline: a president's voice is compared against
# presidents whose first speech falls within this many years of their own.
ERA_WINDOW = 24


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def build_embeddings(df: pd.DataFrame | None = None, force: bool = False) -> pd.DataFrame:
    """Mean speech embedding per president, with 2D PCA coordinates."""
    if EMB_PATH.exists() and not force:
        return pd.read_parquet(EMB_PATH)

    if df is None:
        df = load()

    model = StaticModel.from_pretrained(MODEL_NAME)
    speech_vecs = _normalize(np.asarray(model.encode(df["transcript"].tolist())))

    speech_emb = pd.DataFrame(
        speech_vecs, columns=[f"e{j}" for j in range(speech_vecs.shape[1])]
    )
    speech_emb.insert(0, "doc_name", df["doc_name"].values)
    speech_emb.to_parquet(SPEECH_EMB_PATH, index=False)

    order = president_order(df)
    pres_vecs = np.vstack(
        [speech_vecs[(df["president"] == p).values].mean(axis=0) for p in order]
    )
    pres_vecs = _normalize(pres_vecs)

    coords = PCA(n_components=2, random_state=42).fit_transform(pres_vecs)

    out = pd.DataFrame(
        {
            "president": order,
            "party": [df[df["president"] == p]["party"].iloc[0] for p in order],
            "n_speeches": [int((df["president"] == p).sum()) for p in order],
            "first_year": [int(df[df["president"] == p]["year"].min()) for p in order],
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
        }
    )
    vec_df = pd.DataFrame(pres_vecs, columns=[f"e{j}" for j in range(pres_vecs.shape[1])])
    out = pd.concat([out, vec_df], axis=1)
    out.to_parquet(EMB_PATH, index=False)
    return out


def similarity_matrix(emb: pd.DataFrame) -> pd.DataFrame:
    """Cosine similarity, presidents in chronological order."""
    vec_cols = [c for c in emb.columns if c.startswith("e")]
    m = _normalize(emb[vec_cols].to_numpy())
    sim = m @ m.T
    return pd.DataFrame(sim, index=emb["president"], columns=emb["president"])


def build_adjusted(emb: pd.DataFrame | None = None, force: bool = False) -> pd.DataFrame:
    """Era-adjusted president embeddings: each vector minus the mean of
    contemporaries (first speech within ERA_WINDOW years, excluding self).

    Raw similarity is ~0.68 correlated with temporal proximity — it mostly
    measures the era's shared language. The residual is what distinguishes a
    president from their contemporaries, making cross-era comparison
    meaningful (top pair: Lincoln <-> FDR)."""
    if ADJ_PATH.exists() and not force:
        return pd.read_parquet(ADJ_PATH)

    if emb is None:
        emb = build_embeddings()
    vec_cols = [c for c in emb.columns if c.startswith("e")]
    V = _normalize(emb[vec_cols].to_numpy())
    years = emb["first_year"].to_numpy()

    adjusted = np.zeros_like(V)
    for i in range(len(V)):
        mask = (np.abs(years - years[i]) <= ERA_WINDOW) & (np.arange(len(V)) != i)
        if mask.sum() < 2:
            nearest = np.argsort(np.abs(years - years[i]))[1:5]
            mask = np.zeros(len(V), dtype=bool)
            mask[nearest] = True
        adjusted[i] = V[i] - V[mask].mean(axis=0)
    adjusted = _normalize(adjusted)

    # MDS on the actual cosine distances: with 45 points it preserves the
    # pairwise structure far better than PCA (which scattered the top pair,
    # Lincoln <-> FDR, to opposite regions).
    dist = 1.0 - adjusted @ adjusted.T
    np.fill_diagonal(dist, 0.0)
    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
                 n_init=8, normalized_stress=False).fit_transform(dist)
    out = emb[["president", "party", "n_speeches", "first_year"]].copy()
    out["pc1"] = coords[:, 0]
    out["pc2"] = coords[:, 1]
    vec_df = pd.DataFrame(adjusted, columns=vec_cols)
    out = pd.concat([out.reset_index(drop=True), vec_df], axis=1)
    out.to_parquet(ADJ_PATH, index=False)
    return out
