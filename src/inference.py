# ==========================================
# RECSYS_PROJECT/src/inference.py
# Lightweight Production Inference Engine
# Google Drive Streaming Version
# ==========================================

import pickle
import faiss
import gdown
import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path

from src.als.recommend import ALSRecommender
from src.content_based.search import ContentSearcher
from src.hybrid.hybrid import HybridRecommender

# ==========================================
# Runtime Cache Directory (Streamlit Safe)
# ==========================================

CACHE_DIR = Path("/tmp/recsys_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# Google Drive Files
# ==========================================

GDRIVE_FILES = {
    "faiss.index": "https://drive.google.com/uc?id=1Ls_O-Mk4HcVD8rwK-eETDpOCCUFTsb3A",
    "item_features.npy": "https://drive.google.com/uc?id=1IzwqjBdVytPmuBgPPFIYdMH2JUfbaq7w",
    "clean_interactions.parquet": "https://drive.google.com/uc?id=1THrU5p0QebN5eTnLwV3iR8s8b77vdhdu",
    "clean_movies.parquet": "https://drive.google.com/uc?id=1m_IZTPLvf7AKvS3tay6jj1ZKZKRdJyT6",
}

# ==========================================
# Helpers
# ==========================================

def download_if_needed(filename: str) -> Path:
    path = CACHE_DIR / filename
    if not path.exists():
        print(f"⬇️ Downloading {filename} ...")
        gdown.download(GDRIVE_FILES[filename], str(path), quiet=False)
    return path

# ==========================================
# Recommender Engine
# ==========================================

class RecommenderEngine:

    def __init__(self):
        self._load_data()
        self._load_models()
        self._build_als_engine()
        self._build_content_engine()
        self._build_hybrid_engine()

    # --------------------------------------
    # Load Display Data
    # --------------------------------------
    def _load_data(self):
        movies_path = download_if_needed("clean_movies.parquet")
        self.movies = pd.read_parquet(movies_path)

    # --------------------------------------
    # Load Models
    # --------------------------------------
    def _load_models(self):
        print("🔄 Loading models...")

        BASE_DIR = Path(__file__).resolve().parents[1]
        MODELS_DIR = BASE_DIR / "models"

        with open(MODELS_DIR / "als_model.pkl", "rb") as f:
            self.als_model = pickle.load(f)

        with open(MODELS_DIR / "tfidf.pkl", "rb") as f:
            self.tfidf = pickle.load(f)

        with open(MODELS_DIR / "mlb.pkl", "rb") as f:
            self.mlb = pickle.load(f)

        with open(MODELS_DIR / "item_map.pkl", "rb") as f:
            self.item_map = pickle.load(f)

        with open(MODELS_DIR / "user_map.pkl", "rb") as f:
            self.user_map = pickle.load(f)

        with open(MODELS_DIR / "movieId_to_index.pkl", "rb") as f:
            self.movieId_to_index = pickle.load(f)

        self.inv_item_map = {v: k for k, v in self.item_map.items()}

        # FAISS
        faiss_path = download_if_needed("faiss.index")
        self.faiss_index = faiss.read_index(str(faiss_path))

        # Features
        features_path = download_if_needed("item_features.npy")
        self.item_features = np.load(features_path)

        # Sparse Matrix (small enough → keep in repo)
        self.X_sparse = sparse.load_npz(MODELS_DIR / "X_sparse.npz")

        print("✅ Models loaded successfully")

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
        train_df = pd.read_parquet(train_path)

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
        train_df = pd.read_parquet(train_path)

        self.hybrid_engine = HybridRecommender(
            als_recommender=self.als_engine,
            content_searcher=self.content_engine,
            train_df=train_df
        )

    # --------------------------------------
    # Recommendations
    # --------------------------------------
    def recommend_als(self, user_id, top_k=10):
        recs = self.als_engine.recommend_als(user_id, top_k)
        if recs is None:
            return pd.DataFrame()
        return pd.DataFrame(recs, columns=["movieId", "score"]).merge(
            self.movies, on="movieId", how="left"
        )

    def recommend_content(self, user_id, top_k=10):
        recs = self.content_engine.recommend(user_id, top_k)
        if recs is None:
            return pd.DataFrame()
        return pd.DataFrame(recs, columns=["movieId", "score"]).merge(
            self.movies, on="movieId", how="left"
        )

    def recommend_hybrid(self, user_id, top_k=10, alpha=0.7):
        recs = self.hybrid_engine.recommend_weighted(user_id, top_k, alpha)
        if recs is None:
            return pd.DataFrame()
        return pd.DataFrame(recs, columns=["movieId", "score"]).merge(
            self.movies, on="movieId", how="left"
        )
