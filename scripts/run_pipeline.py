# ==========================================
# Main Recommendation Pipeline
# Production-Safe + Relational Sampling + Parquet Saving
# ==========================================

import logging
import time
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import faiss

# =====================
# Import Project Modules
# =====================

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.features import FeatureBuilder
from src.split import TemporalSplitter
from src.content_based.vectorize import ContentVectorizer
from src.content_based.search import ContentSearcher
from src.hybrid.hybrid import HybridRecommender
from src.evaluation import Evaluator
from src.als.train import ALSTrainer
from src.als.recommend import ALSRecommender
from src.als.evaluate import ALSEvaluator

# =====================
# Logging
# =====================

logger = logging.getLogger("recommender_pipeline")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# =====================
# Paths
# =====================

ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SPLITS = ROOT / "data" / "splits"
MODELS_DIR = ROOT / "models"
EVAL_DIR = DATA_PROCESSED / "evaluation"

for p in [DATA_PROCESSED, DATA_SPLITS, MODELS_DIR, EVAL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# =====================
# Helpers
# =====================

def save_pickle(obj, path):
    if not path.exists():
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Saved pickle: {path.name}")

def save_parquet(df, path):
    if not path.exists():
        df.to_parquet(path, index=False)
        logger.info(f"Saved parquet: {path.name}")

# =====================
# Main
# =====================

def main():
    start_time = time.time()
    logger.info("🚀 Starting Recommendation Pipeline...")

    loader = DataLoader()
    preprocessor = DataPreprocessor()
    builder = FeatureBuilder()
    splitter = TemporalSplitter()
    vectorizer = ContentVectorizer()
    evaluator = Evaluator()

    # ---------------------
    # Load Raw Data
    # ---------------------
    logger.info("📥 Loading raw datasets...")
    data = loader.load_data()

    ratings = data["ratings"]
    movies = data["movies"]
    tags = data["tags"]
    links = data["links"]

    # ---------------------
    # Relational Sampling
    # ---------------------
    SAMPLE_RATIO = 0.20
    logger.info(f"📉 Relational sampling {SAMPLE_RATIO*100:.0f}% from ratings...")

    ratings = ratings.sample(frac=SAMPLE_RATIO, random_state=42)

    valid_movie_ids = set(ratings["movieId"].unique())

    movies = movies[movies["movieId"].isin(valid_movie_ids)]
    tags = tags[tags["movieId"].isin(valid_movie_ids)]
    links = links[links["movieId"].isin(valid_movie_ids)]

    logger.info(f"Ratings after sampling: {ratings.shape}")
    logger.info(f"Movies after relational filter: {movies.shape}")

    # ---------------------
    # Preprocessing
    # ---------------------
    logger.info("🧹 Preprocessing data...")

    user_activity, movie_activity = preprocessor.compute_activity(ratings)

    filtered = preprocessor.filter_interactions(
        ratings,
        user_activity,
        movie_activity
    )

    filtered = preprocessor.process_timestamps(filtered)
    filtered = preprocessor.clean_ratings(filtered)
    filtered = preprocessor.remove_duplicate_interactions(filtered)

    movies = preprocessor.clean_movies(movies)
    tags = preprocessor.clean_tags(tags)
    links = preprocessor.clean_links(links)

    save_parquet(filtered, DATA_PROCESSED / "clean_interactions.parquet")
    save_parquet(movies, DATA_PROCESSED / "clean_movies.parquet")
    save_parquet(tags, DATA_PROCESSED / "clean_tags.parquet")

    # ---------------------
    # Feature Engineering
    # ---------------------
    logger.info("🛠️ Building features...")

    features_path = DATA_PROCESSED / "features.pkl"

    if features_path.exists():
        with open(features_path, "rb") as f:
            features = pickle.load(f)
    else:
        features = builder.build_features(filtered, movies, tags)
        save_pickle(features, features_path)

    # ---------------------
    # Train/Test Split
    # ---------------------
    logger.info("📆 Temporal split...")

    train_path = DATA_SPLITS / "train.parquet"
    test_path = DATA_SPLITS / "test.parquet"

    if train_path.exists() and test_path.exists():
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
    else:
        train_df, test_df = splitter.split(filtered)
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)

    # ---------------------
    # ALS
    # ---------------------
    logger.info("💪 Preparing ALS pipeline...")

    if "confidence" not in train_df.columns:
        train_df["confidence"] = 1 + 40 * train_df["rating"]

    user_map_path = MODELS_DIR / "user_map.pkl"
    item_map_path = MODELS_DIR / "item_map.pkl"

    if user_map_path.exists():
        user_map = pickle.load(open(user_map_path, "rb"))
        item_map = pickle.load(open(item_map_path, "rb"))
    else:
        user_map = {u: i for i, u in enumerate(train_df["userId"].unique())}
        item_map = {m: i for i, m in enumerate(train_df["movieId"].unique())}
        save_pickle(user_map, user_map_path)
        save_pickle(item_map, item_map_path)

    inv_item_map = {i: m for m, i in item_map.items()}

    train_df["u"] = train_df["userId"].map(user_map)
    train_df["i"] = train_df["movieId"].map(item_map)

    from scipy.sparse import save_npz, load_npz

    als_model_path = MODELS_DIR / "als_model.pkl"
    X_sparse_path = MODELS_DIR / "X_sparse.npz"

    als_trainer = ALSTrainer(factors=64, iterations=20)

    if als_model_path.exists():
        als_model_raw = pickle.load(open(als_model_path, "rb"))
        X_sparse = load_npz(X_sparse_path)
    else:
        als_model_raw, X_sparse = als_trainer.fit(train_df, user_map, item_map)
        save_pickle(als_model_raw, als_model_path)
        save_npz(X_sparse_path, X_sparse)

    als_recommender = ALSRecommender(
        als_model_raw,
        X_sparse,
        user_map,
        item_map,
        inv_item_map
    )

    # ---------------------
    # ALS Evaluation
    # ---------------------
    als_results_path = EVAL_DIR / "als_results.parquet"

    if not als_results_path.exists():
        als_eval = ALSEvaluator(als_recommender)
        test_user_items = test_df.groupby("userId")["movieId"].apply(set)
        preds = als_eval.precompute(test_user_items, k=10)

        results = pd.DataFrame([{
            "precision@10": als_eval.precision(test_user_items, preds, 10),
            "recall@10": als_eval.recall(test_user_items, preds, 10),
            "ndcg@10": als_eval.ndcg(test_user_items, preds, 10),
        }])

        results.to_parquet(als_results_path, index=False)

    # ---------------------
    # Content-Based
    # ---------------------
    logger.info("📚 Building content-based system...")

    mlb_path = MODELS_DIR / "mlb.pkl"
    tfidf_path = MODELS_DIR / "tfidf.pkl"
    movieId_to_index_path = MODELS_DIR / "movieId_to_index.pkl"
    item_features_path = MODELS_DIR / "item_features.npy"
    faiss_index_path = MODELS_DIR / "faiss.index"

    if all(p.exists() for p in [
        mlb_path, tfidf_path,
        movieId_to_index_path,
        item_features_path,
        faiss_index_path
    ]):
        mlb = pickle.load(open(mlb_path, "rb"))
        tfidf = pickle.load(open(tfidf_path, "rb"))
        movieId_to_index = pickle.load(open(movieId_to_index_path, "rb"))
        item_features = np.load(item_features_path)
        faiss_index = faiss.read_index(str(faiss_index_path))
        index_to_movieId = {i: m for m, i in movieId_to_index.items()}
    else:
        item_features, mlb, tfidf = vectorizer.build_item_features(movies, tags)
        save_pickle(mlb, mlb_path)
        save_pickle(tfidf, tfidf_path)
        np.save(item_features_path, item_features)

        faiss_index = vectorizer.build_faiss_index(item_features)
        faiss.write_index(faiss_index, str(faiss_index_path))

        movieId_to_index, index_to_movieId = vectorizer.build_id_mappings(movies)
        save_pickle(movieId_to_index, movieId_to_index_path)

    content_searcher = ContentSearcher(
        train_df,
        item_features,
        faiss_index,
        movieId_to_index,
        index_to_movieId
    )

    # ---------------------
    # Hybrid
    # ---------------------
    hybrid_model = HybridRecommender(
        als_recommender,
        content_searcher,
        train_df
    )

    # ---------------------
    # Hybrid Evaluation
    # ---------------------
    logger.info("📊 Evaluating Hybrid...")

    sampled_users = test_df["userId"].drop_duplicates().sample(
        min(100, test_df["userId"].nunique()),
        random_state=42
    )

    test_df_sampled = test_df[test_df["userId"].isin(sampled_users)]

    precision, recall, ndcg, evaluated_users = evaluator.evaluate_model(
        hybrid_model,
        test_df_sampled,
        k=10
    )

    hybrid_results = pd.DataFrame([{
        "precision@10": precision,
        "recall@10": recall,
        "ndcg@10": ndcg,
        "evaluated_users": evaluated_users
    }])

    hybrid_results.to_parquet(
        EVAL_DIR / "hybrid_results.parquet",
        index=False
    )

    logger.info("🎉 Pipeline finished successfully")
    logger.info(f"⏳ Runtime: {time.time()-start_time:.2f} sec")


if __name__ == "__main__":
    main()
