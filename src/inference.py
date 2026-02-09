# ==========================================
# RECSYS_PROJECT/src/inference.py
# Robust Production Inference Engine
# Arrow / LargeUtf8 SAFE – Cloud Stable Edition
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
# Runtime Cache Directory
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
# Helpers
# ==========================================

def _safe_remove(path: Path):
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def download_if_needed(filename: str, force: bool = False) -> Path:
    path = CACHE_DIR / filename

    if force:
        _safe_remove(path)

    if not path.exists():
        logger.info(f"⬇️ Downloading {filename}")
        gdown.download(
            GDRIVE_FILES[filename],
            str(path),
            quiet=False,
            fuzzy=True
        )

    if not path.exists() or path.stat().st_size < 1024:
        raise RuntimeError(f"❌ Invalid or empty file: {filename}")

    return path


# ==========================================
# 🔥 Arrow / LargeUtf8 Killer
# ==========================================

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        dtype_name = str(df[col].dtype)

        if dtype_name in ["string", "large_string"]:
            df[col] = df[col].astype("object")

        elif "string" in dtype_name.lower():
            df[col] = df[col].astype("object")

    return df


def safe_read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return sanitize_dataframe(df)


# ==========================================
# NPZ Loader
# ==========================================

def load_npz_as_npy_safe(path: Path) -> np.ndarray:
    try:
        with np.load(path) as data:
            if "data" not in data:
                raise ValueError("Missing 'data' key in NPZ")
            return data["data"]
    except Exception:
        logger.warning(f"⚠️ Corrupted NPZ detected: {path.name} → retrying")
        _safe_remove(path)
        path = download_if_needed(path.name, force=True)
        with np.load(path) as data:
            return data["data"]

# ==========================================
# Recommender Engine
# ==========================================

class RecommenderEngine:

    def __init__(self):
        logger.info("🚀 Initializing Recommender Engine")

        self._load_data()
        self._load_models()
        self._build_als_engine()
        self._build_content_engine()
        self._build_hybrid_engine()

        logger.info("✅ Recommender Engine Ready")

    # --------------------------------------
    # Load Display Data
    # --------------------------------------
    def _load_data(self):
        movies_path = download_if_needed("clean_movies.parquet")
        self.movies = safe_read_parquet(movies_path)

    # --------------------------------------
    # Load Models
    # --------------------------------------
    def _load_models(self):

        logger.info("🔄 Loading models")

        BASE_DIR = Path(__file__).resolve().parents[1]
        MODELS_DIR = BASE_DIR / "models"

        def load_pickle(name: str):
            path = MODELS_DIR / name
            if not path.exists():
                raise FileNotFoundError(f"❌ Missing model file: {name}")
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
        self.item_features = load_npz_as_npy_safe(features_path)

        sparse_path = MODELS_DIR / "X_sparse.npz"
        if not sparse_path.exists():
            raise FileNotFoundError("❌ X_sparse.npz not found")

        self.X_sparse = sparse.load_npz(sparse_path)

        logger.info("✅ Models loaded successfully")

    # --------------------------------------
    # ALS Engine
    # --------------------------------------
    def _build_als_engine(self):
        self.als_engine = ALSRecommender(
            model=self.als_model,
            X=self.X_sparse,
            user_map=self.user_map,
            item_map=self.item_map,
            inv_item_map=self.inv_item_map
        )

    # --------------------------------------
    # Content Engine
    # --------------------------------------
    def _build_content_engine(self):

        train_path = download_if_needed("clean_interactions.parquet")
        train_df = safe_read_parquet(train_path)

        self.content_engine = ContentSearcher(
            train_df=train_df,
            item_features=self.item_features,
            faiss_index=self.faiss_index,
            movieId_to_index=self.movieId_to_index,
            index_to_movieId={v: k for k, v in self.movieId_to_index.items()}
        )

    # --------------------------------------
    # Hybrid Engine
    # --------------------------------------
    def _build_hybrid_engine(self):

        train_path = download_if_needed("clean_interactions.parquet")
        train_df = safe_read_parquet(train_path)

        self.hybrid_engine = HybridRecommender(
            als_recommender=self.als_engine,
            content_searcher=self.content_engine,
            train_df=train_df
        )

    # --------------------------------------
    # Output Formatter
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

    def recommend_content(self, user_id, top_k=10):
        return self._format_output(
            self.content_engine.recommend(user_id, top_k)
        )

    def recommend_hybrid(self, user_id, top_k=10, alpha=0.7):
        return self._format_output(
            self.hybrid_engine.recommend_weighted(user_id, top_k, alpha)
        )
