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

from .corpus import DATA_DIR, load, president_order

EMB_PATH = DATA_DIR / "president_embeddings.parquet"

MODEL_NAME = "minishlab/potion-base-8M"


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
