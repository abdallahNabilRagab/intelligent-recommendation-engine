# ==========================================
# RECSYS_PROJECT/src/inference.py
# Production Recommender Engine
# Streamlit Cloud Stable • Arrow Proof • No LargeUtf8
# ==========================================

import pickle
import faiss
import gdown
import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
import logging

from src.als.recommend import ALSRecommender
from src.content_based.search import ContentSearcher
from src.hybrid.hybrid import HybridRecommender

# ==========================================
# Logging
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ==========================================
# Runtime Cache
# ==========================================

CACHE_DIR = Path("/tmp/recsys_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# Google Drive Files
# ==========================================

GDRIVE_FILES = {
    "faiss.index": "https://drive.google.com/uc?id=1Ls_O-Mk4HcVD8rwK-eETDpOCCUFTsb3A",
    "item_features.npy": "https://drive.google.com/uc?id=1tWBmtp0SoO7t0ef_Wei55-ekvbv-DeAA",
    "clean_interactions.parquet": "https://drive.google.com/uc?id=1THrU5p0QebN5eTnLwV3iR8s8b77vdhdu",
    "clean_movies.parquet": "https://drive.google.com/uc?id=1m_IZTPLvf7AKvS3tay6jj1ZKZKRdJyT6",
}

# ==========================================
# Download Helper
# ==========================================

def download_if_needed(filename: str, force: bool = False):

    path = CACHE_DIR / filename

    if force and path.exists():
        path.unlink()

    if not path.exists():
        logger.info(f"⬇️ Downloading {filename}")
        gdown.download(
            GDRIVE_FILES[filename],
            str(path),
            quiet=False,
            fuzzy=True
        )

    if not path.exists() or path.stat().st_size < 1024:
        raise RuntimeError(f"Invalid file: {filename}")

    return path

# ==========================================
# 🔥 HARD Arrow Killer
# ==========================================

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    df = pd.DataFrame(df).copy()

    for col in df.columns:

        if "string" in str(df[col].dtype).lower():
            df[col] = df[col].astype(str)

        elif str(df[col].dtype).startswith("large"):
            df[col] = df[col].astype(str)

        elif df[col].dtype == "object":
            df[col] = df[col].astype(str)

    df = df.convert_dtypes(dtype_backend="numpy_nullable")

    return df


def safe_read_parquet(path: Path):

    df = pd.read_parquet(path)

    return sanitize_dataframe(df)

# ==========================================
# Numpy Loader
# ==========================================

def load_npz_safe(path: Path):

    with np.load(path) as data:
        if "data" in data:
            return data["data"]

    raise RuntimeError("Invalid NPZ format")

# ==========================================
# Recommender Engine
# ==========================================

class RecommenderEngine:

    def __init__(self):

        logger.info("🚀 Initializing Engine")

        self._load_data()
        self._load_models()
        self._build_engines()

        logger.info("✅ Engine Ready")

    # --------------------------------------
    # Data
    # --------------------------------------
    def _load_data(self):

        movies_path = download_if_needed("clean_movies.parquet")

        self.movies = safe_read_parquet(movies_path)

    # --------------------------------------
    # Models
    # --------------------------------------
    def _load_models(self):

        BASE_DIR = Path(__file__).resolve().parents[1]
        MODELS_DIR = BASE_DIR / "models"

        def load_pickle(name):

            path = MODELS_DIR / name

            if not path.exists():
                raise FileNotFoundError(name)

            with open(path, "rb") as f:
                return pickle.load(f)

        self.als_model = load_pickle("als_model.pkl")
        self.tfidf = load_pickle("tfidf.pkl")
        self.mlb = load_pickle("mlb.pkl")
        self.item_map = load_pickle("item_map.pkl")
        self.user_map = load_pickle("user_map.pkl")
        self.movieId_to_index = load_pickle("movieId_to_index.pkl")

        self.inv_item_map = {v: k for k, v in self.item_map.items()}

        faiss_path = download_if_needed("faiss.index")
        self.faiss_index = faiss.read_index(str(faiss_path))

        features_path = download_if_needed("item_features.npy")
        self.item_features = load_npz_safe(features_path)

        sparse_path = MODELS_DIR / "X_sparse.npz"

        if not sparse_path.exists():
            raise FileNotFoundError("X_sparse.npz missing")

        self.X_sparse = sparse.load_npz(sparse_path)

    # --------------------------------------
    # Engines
    # --------------------------------------
    def _build_engines(self):

        interactions_path = download_if_needed("clean_interactions.parquet")
        train_df = safe_read_parquet(interactions_path)

        self.als_engine = ALSRecommender(
            self.als_model,
            self.X_sparse,
            self.user_map,
            self.item_map,
            self.inv_item_map
        )

        self.content_engine = ContentSearcher(
            train_df,
            self.item_features,
            self.faiss_index,
            self.movieId_to_index,
            {v: k for k, v in self.movieId_to_index.items()}
        )

        self.hybrid_engine = HybridRecommender(
            self.als_engine,
            self.content_engine,
            train_df
        )

    # --------------------------------------
    # Formatter
    # --------------------------------------
    def _format_output(self, recs):

        if recs is None or len(recs) == 0:
            return pd.DataFrame()

        df = (
            pd.DataFrame(recs, columns=["movieId", "score"])
            .merge(self.movies, on="movieId", how="left")
        )

        return sanitize_dataframe(df)

    # --------------------------------------
    # APIs
    # --------------------------------------
    def recommend_als(self, user_id, top_k=10):
        return self._format_output(
            self.als_engine.recommend_als(user_id, top_k)
        )

    def recommend_content(self, movie_id, top_k=10):
        return self._format_output(
            self.content_engine.recommend(movie_id, top_k)
        )

    def recommend_hybrid(self, user_id, top_k=10, alpha=0.7):
        return self._format_output(
            self.hybrid_engine.recommend_weighted(user_id, top_k, alpha)
        )
