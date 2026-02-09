# ==========================================
# RECSYS_PROJECT/app/streamlit_app.py
# Cloud-Safe Production Streamlit Interface
# Arrow / LargeUtf8 FINAL & GUARANTEED FIX
# ==========================================

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# ==========================================
# Page Configuration (MUST be first)
# ==========================================
st.set_page_config(
    page_title="Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# ==========================================
# Resolve Project Root (Cloud & Local Safe)
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ==========================================
# Safe Imports
# ==========================================
try:
    from src.inference import RecommenderEngine
except Exception as e:
    st.error("❌ Failed to load Recommender Engine.")
    st.exception(e)
    st.stop()

# ==========================================
# Load Recommender Engine (Cached)
# ==========================================
@st.cache_resource(show_spinner="🔄 Loading recommendation engine...")
def load_engine():
    return RecommenderEngine()

engine = load_engine()

# ==========================================
# Validate Engine Assets
# ==========================================
if not hasattr(engine, "movies") or engine.movies is None:
    st.error("❌ Movies metadata not found in the engine.")
    st.stop()

# Sanitize movies metadata
movies = engine.movies.copy().astype(str)
movies["movieId"] = movies["movieId"].astype(str)
movies["title"] = movies["title"].astype(str)

# ==========================================
# 🚨 FINAL BULLETPROOF TABLE RENDERER
# (NO Arrow – NO LargeUtf8 – Cloud Safe)
# ==========================================
def render_table_html(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("No recommendations found.")
        return

    df = df.copy().reset_index(drop=True)

    # Force everything to string + HARD truncate
    df = df.astype(str)
    for col in df.columns:
        df[col] = df[col].str.slice(0, 300)

    html = df.to_html(
        index=False,
        escape=True,
        classes="recsys-table"
    )

    st.markdown(
        f"""
        <style>
        .recsys-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        .recsys-table th {{
            background-color: #111;
            color: white;
            padding: 8px;
            text-align: left;
        }}
        .recsys-table td {{
            padding: 8px;
            border-bottom: 1px solid #444;
            white-space: normal;
            word-break: break-word;
            max-width: 420px;
        }}
        </style>
        {html}
        """,
        unsafe_allow_html=True
    )

# ==========================================
# UI Header
# ==========================================
st.title("🎬 Hybrid Movie Recommendation System")
st.caption("Academic • Production-Grade • Research-Oriented")

# ==========================================
# Sidebar Controls
# ==========================================
with st.sidebar:
    st.header("⚙️ Recommendation Settings")
    mode = st.selectbox(
        "Recommender Type",
        ["ALS (User-Based)", "Content-Based", "Hybrid"]
    )

# ==========================================
# Main Panel
# ==========================================
st.markdown("### 🎯 Recommendation Interface")

# ------------------------------------------
# ALS Mode
# ------------------------------------------
if mode == "ALS (User-Based)":
    user_id = st.number_input("Enter User ID", min_value=1, step=1)

    if st.button("🎯 Get Recommendations"):
        with st.spinner("Generating recommendations..."):
            recs = engine.recommend_als(user_id)

        st.subheader("📌 Recommended Movies")
        render_table_html(recs)

# ------------------------------------------
# Content-Based Mode
# ------------------------------------------
elif mode == "Content-Based":
    movie_title = st.selectbox(
        "Select a Movie",
        movies["title"].sort_values().unique()
    )

    if st.button("🔍 Find Similar Movies"):
        movie_id = movies.loc[
            movies["title"] == movie_title, "movieId"
        ].iloc[0]

        with st.spinner("Finding similar movies..."):
            recs = engine.recommend_content(movie_id)

        st.subheader("📌 Similar Movies")
        render_table_html(recs)

# ------------------------------------------
# Hybrid Mode
# ------------------------------------------
elif mode == "Hybrid":
    user_id = st.number_input("Enter User ID", min_value=1, step=1)

    if st.button("🤝 Generate Hybrid Recommendations"):
        with st.spinner("Running hybrid inference..."):
            recs = engine.recommend_hybrid(user_id)

        st.subheader("📌 Hybrid Recommendations")
        render_table_html(recs)

# ==========================================
# Footer / Developer Info
# ==========================================
st.markdown("---")
with st.container():
    st.markdown("## 👨‍💻 Developer Information")
    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
            width=110
        )

    with col2:
        st.markdown("""
### 🧑‍💻 Abdallah Nabil Ragab

**🎓 M.Sc. in Business Information Systems**  
**💼 Data Scientist | Machine Learning Engineer | Software Engineer**

---

💬 **Feedback & Suggestions**

If you have ideas, feature requests, or found an issue,
your feedback is highly appreciated.

📩 **Email:**  
`abdallah.nabil.ragab94@gmail.com`
""")

st.markdown("---")
