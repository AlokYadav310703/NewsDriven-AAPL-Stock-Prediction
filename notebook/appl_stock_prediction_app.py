"""
AAPL Stock Price Prediction with News-Aware Correction Model
================================================================

DEPLOYMENT-READY VERSION (CORRECTED):
- All paths are relative to the script location, with robust fallback search
- Enhanced UI with financial-focused design
- Customizable configuration
- Error handling for missing resources
- FIXED: ChromaDB path resolution no longer silently falls back to an empty
  collection. The app now verifies the collection actually contains items
  and surfaces the real path + item count in the sidebar and main body.
- FIXED: ChromaDB query failures are no longer swallowed silently. The full
  exception + traceback is shown in an expander so failures (e.g. embedding
  dimension mismatches) are visible instead of just showing
  "No ChromaDB matches available".
- NEW: ChromaDB collection loaded from HuggingFace Hub with automatic caching
- NEW: yfinance data is validated for missing/NaN values. If Yahoo Finance
  returns nothing, too few rows, or any NaN OHLC values (common during
  outages, rate limiting, or before market data updates), the app now stops
  with a clear, specific error message instead of silently propagating NaNs
  into the models and showing "$nan" or a cryptic downstream exception.

Architecture:
  Stage 1: Ensemble (LSTM + CNN) predicts next-day price from OHLC
  Stage 2: Correction Model (sklearn regressor, e.g Random Forest) refines
           the Stage 1 prediction using today's news features and market
           technical indicators.

Run with:
    streamlit run stock_prediction_app_v2.py
"""

import os
import sys
import traceback
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import chromadb
import tensorflow as tf
import joblib
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import requests
from pathlib import Path
import shutil

# ══════════════════════════════════════════════════════════════════════════════
# PROJECT STRUCTURE & PATH CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

APP_DIR = Path(__file__).resolve().parent
CWD = Path.cwd().resolve()

_SEARCH_CANDIDATES = []
for _c in [APP_DIR, APP_DIR.parent, APP_DIR.parent.parent, CWD, CWD.parent]:
    if _c not in _SEARCH_CANDIDATES:
        _SEARCH_CANDIDATES.append(_c)


def _find_folder(folder_name: str) -> Path:
    """Return the first existing candidate/folder_name, else APP_DIR/folder_name."""
    for candidate in _SEARCH_CANDIDATES:
        target = candidate / folder_name
        if target.exists():
            return target
    return APP_DIR / folder_name


MODEL_DIR = _find_folder("models")

# HuggingFace configuration
HF_REPO_ID = "alokyadav310703/Similarity"
HF_COLLECTION_NAME = "aapl_memory_v2"
LOCAL_CHROMA_CACHE = Path.home() / ".cache" / "aapl_stock_predictor" / "chroma"

TEXT_CHAR_CAP = 2000

# Hardcoded for demo purposes only. In a real deployment this should come
# from an environment variable or secrets manager, not be committed to source.
NEWSAPI_KEY = "1671260b3d8341d59df512e6cd64224f"

STAGE2_FEATURE_ORDER = [
    "sentiment_score",
    "impact_score",
    "event_weight",
    "return_1d",
    "return_5d",
]

OHLC_COLS = ["open", "high", "low", "close"]

# ══════════════════════════════════════════════════════════════════════════════
# HUGGINGFACE CHROMADB DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def download_chroma_from_huggingface(repo_id: str, collection_name: str, cache_dir: Path) -> Path:
    """
    Download ChromaDB collection from HuggingFace Hub.

    Assumes the ChromaDB collection is stored as a directory in the HF repo.
    The directory should contain the ChromaDB persistent database files
    (e.g., index files, metadata, etc.).

    Args:
        repo_id: HuggingFace repo ID (e.g., "alokyadav310703/Similarity")
        collection_name: Name of the collection folder in the repo
        cache_dir: Local cache directory

    Returns:
        Path to the local ChromaDB directory
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_chroma_path = cache_dir / collection_name

    try:
        st.info(f"📥 Downloading ChromaDB collection from HuggingFace ({repo_id})...")

        # Try to download the directory from HuggingFace
        # Note: This uses the huggingface_hub library which you may need to install
        try:
            from huggingface_hub import snapshot_download

            # Download the entire repo snapshot
            repo_path = snapshot_download(repo_id=repo_id)
            source_collection = Path(repo_path) / collection_name

            if not source_collection.exists():
                raise FileNotFoundError(
                    f"Collection '{collection_name}' not found in HuggingFace repo at {repo_id}"
                )

            # Copy to cache location
            if local_chroma_path.exists():
                shutil.rmtree(local_chroma_path)
            shutil.copytree(source_collection, local_chroma_path)

            st.success(f"✓ ChromaDB collection downloaded and cached at {local_chroma_path}")
            return local_chroma_path

        except ImportError:
            st.error(
                "⚠️ `huggingface_hub` library not found. Install it with:\n"
                "`pip install huggingface_hub`"
            )
            raise

    except Exception as e:
        st.error(f"Failed to download ChromaDB from HuggingFace: {e}")
        raise


def get_or_create_local_chroma_cache(repo_id: str, collection_name: str, cache_dir: Path) -> Path:
    """
    Get the local ChromaDB cache path.
    If not cached, download from HuggingFace.

    Args:
        repo_id: HuggingFace repo ID
        collection_name: Collection folder name
        cache_dir: Cache directory path

    Returns:
        Path to local ChromaDB directory
    """
    local_path = cache_dir / collection_name

    # If already cached, use it
    if local_path.exists():
        return local_path

    # Otherwise download from HuggingFace
    return download_chroma_from_huggingface(repo_id, collection_name, cache_dir)


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM UI STYLING
# ══════════════════════════════════════════════════════════════════════════════

def apply_custom_styling():
    """Apply custom CSS for financial dashboard aesthetic"""
    custom_css = """
    <style>
    /* Root Variables */
    :root {
        --primary-color: #0066cc;
        --success-color: #10b981;
        --danger-color: #ef4444;
        --warning-color: #f59e0b;
        --dark-bg: #0f172a;
        --light-bg: #f8fafc;
        --border-color: #e2e8f0;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
    }

    /* Overall Layout */
    .main {
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }

    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
        border-radius: 18px;
        padding: 1.5rem 1.75rem;
        color: white;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.2);
        margin-bottom: 1rem;
    }

    .hero-card h1 {
        color: white;
        background: none;
        -webkit-text-fill-color: white;
        margin-bottom: 0.35rem;
    }

    .hero-card p {
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 0;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        padding: 0.7rem 0.9rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.18);
        color: white;
        font-weight: 600;
        backdrop-filter: blur(8px);
    }

    .sidebar-card {
        background: linear-gradient(135deg, #1e293b 0%, #111827 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 0.9rem;
        margin-bottom: 1rem;
        color: #e2e8f0;
    }

    /* Headers */
    h1, h2, h3 {
        color: var(--text-primary);
        font-weight: 700;
    }

    h1 {
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    /* Metric Cards */
    [data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }

    [data-testid="metric-container"]:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border-color: #0066cc;
    }

    /* Containers with borders */
    [data-testid="column"] {
        gap: 1.5rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 102, 204, 0.2);
    }

    .stButton > button:hover {
        box-shadow: 0 4px 16px rgba(0, 102, 204, 0.4);
        transform: translateY(-2px);
    }

    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextInput input,
    .stSlider > div > div > div > input {
        border-radius: 8px;
        border: 1.5px solid var(--border-color);
        padding: 0.75rem;
    }

    /* Sidebar */
    .stSidebar, [data-testid="stSidebar"] {
        background: #0f172a !important;
        border-right: 1px solid #1e293b;
    }

    .stSidebar [data-testid="stSidebarContent"] {
        padding: 2rem 1.5rem;
    }

    /* Sidebar text & headers */
    .stSidebar, .stSidebar p, .stSidebar span, .stSidebar label,
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar .stMarkdown,
    .stSidebar .stCaption {
        color: #e2e8f0 !important;
    }

    .stSidebar h2, .stSidebar h3 {
        border-bottom: none;
        color: #f8fafc !important;
    }

    /* Sidebar inputs */
    .stSidebar .stTextInput input,
    .stSidebar .stTextInput > div > div > input {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155 !important;
    }

    .stSidebar .stTextInput input::placeholder {
        color: #64748b !important;
    }

    /* Sidebar slider */
    .stSidebar [data-testid="stSlider"] label {
        color: #e2e8f0 !important;
    }

    /* Sidebar divider */
    .stSidebar hr {
        background: linear-gradient(90deg, transparent, #334155, transparent);
    }

    /* Sidebar buttons */
    .stSidebar .stButton > button {
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        color: white;
    }

    /* Sidebar alert boxes stay legible on dark background */
    .stSidebar .stSuccess, .stSidebar .stError,
    .stSidebar .stWarning, .stSidebar .stInfo {
        color: #0f172a !important;
    }

    /* Info/Warning/Error boxes */
    .stInfo {
        background-color: #ecf0ff !important;
        border-left: 4px solid #0066cc !important;
        border-radius: 8px;
        padding: 1rem;
    }

    .stSuccess {
        background-color: #ecfdf5 !important;
        border-left: 4px solid #10b981 !important;
        border-radius: 8px;
        padding: 1rem;
    }

    .stWarning {
        background-color: #fffbeb !important;
        border-left: 4px solid #f59e0b !important;
        border-radius: 8px;
        padding: 1rem;
    }

    .stError {
        background-color: #fef2f2 !important;
        border-left: 4px solid #ef4444 !important;
        border-radius: 8px;
        padding: 1rem;
    }

    /* Dividers */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 2rem 0;
    }

    /* Captions and small text */
    .stCaption {
        color: var(--text-secondary);
        font-size: 0.875rem;
    }

    /* Select/Selectbox */
    .stSelectbox > div > div {
        border-radius: 8px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Subheaders */
    h2 {
        color: #1e293b;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #0066cc;
        padding-bottom: 0.5rem;
    }

    /* Price highlights */
    .price-up {
        color: #10b981;
        font-weight: 700;
    }

    .price-down {
        color: #ef4444;
        font-weight: 700;
    }

    /* Animation for loading */
    @keyframes shimmer {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }

    .loading {
        animation: shimmer 2s infinite;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES (Loaded once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    """Load SentenceTransformer for news embeddings"""
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Connecting to ChromaDB...")
def load_chromadb_collection(chroma_path: str):
    """Load ChromaDB collection with stored AAPL news events.

    FIX: previously this used get_or_create_collection() and returned
    success even when the resulting collection was empty (e.g. because the
    path was wrong, or the collection name didn't match what was actually
    on disk). Now it:
      1) Verifies the on-disk path exists before connecting.
      2) Lists ALL collections at that path so a name mismatch is visible.
      3) Reports the live item count so an empty result is never silent.

    Returns a dict: {"collection": <Collection|None>, "path": str,
                      "all_collections": [names], "count": int, "error": str|None}
    """
    result = {
        "collection": None,
        "path": str(chroma_path),
        "all_collections": [],
        "count": 0,
        "error": None,
    }

    try:
        chroma_path = Path(chroma_path)
        if not chroma_path.exists():
            result["error"] = f"ChromaDB path does not exist on disk: {chroma_path}"
            return result

        client = chromadb.PersistentClient(path=str(chroma_path))

        # Show every collection actually present at this path -- this is
        # what catches a "wrong collection name" bug immediately.
        existing = client.list_collections()
        result["all_collections"] = [c.name for c in existing]

        if "aapl_events" not in result["all_collections"]:
            result["error"] = (
                f"No collection named 'aapl_events' found at {chroma_path}. "
                f"Collections present: {result['all_collections'] or '(none)'}"
            )
            # Still create/return it so the rest of the app doesn't crash,
            # but the error above makes the root cause visible.
            collection = client.get_or_create_collection(
                name="aapl_events",
                metadata={"hnsw:space": "cosine"},
            )
        else:
            collection = client.get_collection(name="aapl_events")

        result["collection"] = collection
        result["count"] = collection.count()

        if result["count"] == 0 and result["error"] is None:
            result["error"] = (
                f"Collection 'aapl_events' exists at {chroma_path} but contains 0 items."
            )

        return result

    except Exception as e:
        result["error"] = f"{e}\n\n{traceback.format_exc()}"
        return result


@st.cache_resource(show_spinner="Loading ML models...")
def load_models(model_path, correction_model_filename: str):
    """
    Load all trained models.
    Stage 1 (LSTM, CNN, meta-learner) + Stage 2 sklearn correction model
    """
    try:
        model_path = Path(model_path)
        if not model_path.exists():
            st.error(f"Model directory not found: {model_path}")
            return None

        models = {}

        # Load base learners
        lstm_path = model_path / "lstm_model.keras"
        cnn_path = model_path / "cnn_model.keras"

        if not lstm_path.exists() or not cnn_path.exists():
            st.error(f"Required model files not found in {model_path}")
            return None

        models['lstm'] = tf.keras.models.load_model(str(lstm_path))
        models['cnn'] = tf.keras.models.load_model(str(cnn_path))

        # Load meta-learner (Stage 1 ensemble)
        try:
            models['meta_learner'] = joblib.load(model_path / "meta_model.pkl")
        except FileNotFoundError:
            models['meta_learner'] = joblib.load(model_path / "meta_learner.pkl")

        # Stage 2 correction model
        correction_model_path = model_path / correction_model_filename
        if not correction_model_path.exists():
            st.error(f"Correction model not found: {correction_model_filename}")
            return None

        models['correction_model'] = joblib.load(correction_model_path)

        # Scalers
        scaler_files = [
            "feature_scaler.pkl",
            "target_scaler.pkl",
            "feature_scaler_raw.pkl",
            "target_scaler_raw.pkl"
        ]

        for scaler_file in scaler_files:
            scaler_path = model_path / scaler_file
            if not scaler_path.exists():
                st.error(f"Scaler not found: {scaler_file}")
                return None
            models[scaler_file.replace('.pkl', '')] = joblib.load(scaler_path)

        return models
    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_apple_ohlc(lookback: int = 30) -> pd.DataFrame:
    """Fetch Apple OHLC data from yfinance for the last N days.

    NEW: Validates the data before returning it. Yahoo Finance can, without
    raising an exception, return:
      - an empty DataFrame (outage, rate limiting, ticker temporarily
        unavailable)
      - fewer rows than needed to compute a 5-day return
      - rows with NaN values in Open/High/Low/Close (partial trading days,
        data provider glitches, holidays that still get a placeholder row)

    Any of these previously flowed silently downstream and surfaced later as
    "$nan" in the UI or a cryptic scaler/model exception. Now each case is
    caught here with a specific, actionable st.error message, and the
    function returns None so callers can st.stop() cleanly.
    """
    try:
        df = yf.download("AAPL", period="6mo", progress=False)

        # Case 1: yfinance returned nothing at all
        if df is None or df.empty:
            st.error(
                "⚠️ **No price data available.** Yahoo Finance returned an "
                "empty response for AAPL. This usually means the data "
                "provider is temporarily down, rate-limiting requests, or "
                "the market data feed hasn't updated yet. Please wait a "
                "few minutes and try again."
            )
            return None

        df = df.tail(lookback + 6).reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = df.columns.str.lower()

        # Case 2: not enough rows to compute a 5-day return
        if len(df) < 6:
            st.error(
                f"⚠️ **Not enough price history available.** Only {len(df)} "
                "day(s) of AAPL data were returned, but at least 6 are "
                "needed to compute technical indicators. Yahoo Finance may "
                "be returning incomplete data right now — please try again shortly."
            )
            return None

        # Case 3: NaN values anywhere in the OHLC columns of the fetched window
        missing_cols = [c for c in OHLC_COLS if c not in df.columns]
        if missing_cols:
            st.error(
                f"⚠️ **Malformed price data.** Expected columns "
                f"{OHLC_COLS} but missing: {missing_cols}. Yahoo Finance may "
                "have changed its response format or returned partial data. "
                "Please try again later."
            )
            return None

        if df[OHLC_COLS].isnull().values.any():
            bad_rows = df[df[OHLC_COLS].isnull().any(axis=1)]
            bad_dates = (
                bad_rows["date"].dt.date.astype(str).tolist()
                if "date" in bad_rows.columns else []
            )
            date_str = f" Affected date(s): {', '.join(bad_dates)}." if bad_dates else ""
            st.error(
                "⚠️ **Incomplete price data (NaN values detected).** Yahoo "
                "Finance returned missing Open/High/Low/Close values for "
                "AAPL, so a reliable prediction can't be computed right "
                f"now.{date_str} This can happen before market close, on "
                "partial trading days, or during a temporary data provider "
                "issue. Please try again later."
            )
            return None

        # Case 4: specifically guard today's (most recent) row, since that
        # becomes `current_price` and feeds every downstream calculation.
        if df.iloc[-1][OHLC_COLS].isnull().any():
            st.error(
                "⚠️ **Today's AAPL price is unavailable (NaN).** This can "
                "happen before the market closes for the day or during a "
                "brief Yahoo Finance API hiccup. Please try again in a few minutes."
            )
            return None

        return df

    except Exception as e:
        st.error(f"⚠️ Failed to fetch OHLC data from Yahoo Finance: {e}")
        return None


def compute_market_returns(df: pd.DataFrame) -> dict:
    """Compute today's market technical indicators"""
    try:
        closes = df["close"].values.astype(float)

        if np.isnan(closes[-1]) or np.isnan(closes[-2]) or np.isnan(closes[-6]):
            st.error(
                "⚠️ Cannot compute market returns: one or more required "
                "closing prices are NaN. Please re-fetch the price data."
            )
            return {"return_1d": 0.0, "return_5d": 0.0}

        return_1d = (closes[-1] - closes[-2]) / closes[-2]
        return_5d = (closes[-1] - closes[-6]) / closes[-6]
        return {"return_1d": float(return_1d), "return_5d": float(return_5d)}
    except Exception as e:
        st.error(f"Error computing market returns: {e}")
        return {"return_1d": 0.0, "return_5d": 0.0}


def fetch_newsapi_articles(api_key: str, num_articles: int = 3) -> list:
    """Fetch latest Apple news from NewsAPI.org"""
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": "Apple AAPL",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": num_articles,
                "apiKey": api_key,
            },
            timeout=15,
        )

        data = resp.json()
        if data.get("status") != "ok":
            raise RuntimeError(data.get("message", "Unknown NewsAPI error"))

        articles = []
        for article in data.get("articles", []):
            title = article.get("title", "")
            body = article.get("description") or article.get("content", "")
            text = f"{title}. {body}".strip()[:TEXT_CHAR_CAP]

            articles.append({
                "title": title,
                "source": (article.get("source") or {}).get("name", "Unknown"),
                "published_at": article.get("publishedAt", "n/a"),
                "url": article.get("url", ""),
                "text": text,
            })

        return articles
    except Exception as e:
        st.error(f"NewsAPI request failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# SIMILARITY SEARCH & NEWS FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def find_similar_events(collection, encoder, query_text: str, top_k: int = 5):
    """Find similar stored events in ChromaDB.

    FIX: the original version caught every exception and returned (None, 0)
    with only `st.error(f"...: {e}")`, which is easy to miss and gives no
    actionable detail (e.g. an embedding-dimension mismatch between the
    query encoder and the stored vectors would look identical to "no data").
    This version surfaces the full traceback in an expander.
    """
    try:
        if collection is None:
            return None, 0

        count = collection.count()
        if count == 0:
            return None, 0

        query_embedding = encoder.encode(query_text[:TEXT_CHAR_CAP]).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count)
        )
        return results, count
    except Exception as e:
        st.error(f"ChromaDB query failed: {e}")
        with st.expander("🔍 Full error details (click to expand)"):
            st.code(traceback.format_exc())
        return None, 0


def extract_news_features(results) -> dict:
    """Extract and aggregate news features from ChromaDB results"""
    if results is None or not results['metadatas'] or not results['metadatas'][0]:
        return _default_news_features()

    metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    similarities = 1 - np.array(distances)
    total_sim = similarities.sum()

    if total_sim == 0:
        return _default_news_features()

    weights = similarities / total_sim

    aggregated = {
        'sentiment_score': np.average(
            [float(m.get('sentiment_score', 0)) for m in metadatas],
            weights=weights
        ),
        'impact_score': np.average(
            [float(m.get('impact_score', 0)) for m in metadatas],
            weights=weights
        ),
        'event_weight': np.average(
            [float(m.get('event_weight', 0)) for m in metadatas],
            weights=weights
        ),
        'news_count': len(metadatas),
        'has_supply_chain_event': int(any(m.get('event_type') == 'supply_chain' for m in metadatas)),
    }

    return aggregated


def _default_news_features() -> dict:
    """Default zero features when no news is available"""
    return {
        'sentiment_score': 0.0,
        'impact_score': 0.0,
        'event_weight': 0.0,
        'news_count': 0,
        'has_supply_chain_event': 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def prepare_ohlc_features(df: pd.DataFrame, scaler, lookback: int) -> np.ndarray:
    """Prepare OHLC data for base learner input"""
    try:
        ohlc = df[['open', 'high', 'low', 'close']].values.astype(float)

        if np.isnan(ohlc).any():
            st.error(
                "⚠️ Cannot prepare model input: NaN values found in OHLC "
                "data. Please re-fetch the price data before running a prediction."
            )
            return None

        if len(ohlc) < lookback:
            ohlc = np.vstack([np.zeros((lookback - len(ohlc), 4)), ohlc])
        else:
            ohlc = ohlc[-lookback:]

        ohlc_scaled = scaler.transform(ohlc)
        return ohlc_scaled.reshape(1, lookback, 4)
    except Exception as e:
        st.error(f"Error preparing OHLC features: {e}")
        return None


def get_stage1_prediction(models: dict, ohlc_for_lstm: np.ndarray, ohlc_for_cnn: np.ndarray):
    """Get Stage 1 ensemble prediction"""
    try:
        lstm_pred_scaled = models['lstm'].predict(ohlc_for_lstm, verbose=0)
        cnn_pred_scaled = models['cnn'].predict(ohlc_for_cnn, verbose=0)

        meta_input = np.hstack([lstm_pred_scaled, cnn_pred_scaled])
        stage1_pred_scaled = models['meta_learner'].predict(meta_input)
        stage1_pred_scaled = np.array(stage1_pred_scaled).reshape(-1, 1)

        lstm_pred_actual = models['target_scaler_raw'].inverse_transform(lstm_pred_scaled)[0, 0]
        cnn_pred_actual = models['target_scaler'].inverse_transform(cnn_pred_scaled)[0, 0]
        stage1_pred_actual = models['target_scaler'].inverse_transform(stage1_pred_scaled)[0, 0]

        return (
            float(stage1_pred_actual),
            {'lstm': float(lstm_pred_actual), 'cnn': float(cnn_pred_actual)},
            float(stage1_pred_scaled[0, 0]),
        )
    except Exception as e:
        st.error(f"Error in Stage 1 prediction: {e}")
        return None, None, None


def get_stage2_prediction(models: dict, stage1_pred_actual: float, news_features_raw: dict, market_returns: dict):
    """Get Stage 2 correction model prediction"""
    try:
        fused_features = {**news_features_raw, **market_returns}
        X = np.array([[fused_features.get(feat, 0.0) for feat in STAGE2_FEATURE_ORDER]])

        predicted_error = float(models['correction_model'].predict(X)[0])
        final_pred_actual = stage1_pred_actual - predicted_error
        correction_actual = final_pred_actual - stage1_pred_actual

        return float(final_pred_actual), float(correction_actual)
    except Exception as e:
        st.error(f"Error in Stage 2 prediction: {e}")
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# UI RENDERING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def render_news_results(article_idx: int, article: dict, results):
    """Render similar news events found in ChromaDB"""
    if results is None or not results['metadatas'] or not results['metadatas'][0]:
        st.info(f"ℹ️ No similar events found in ChromaDB for article {article_idx}")
        return None

    ids = results["ids"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    with st.container(border=True):
        st.subheader(f"📊 Similar Events: {len(ids)} found")

        for rank, (doc_id, meta, dist) in enumerate(zip(ids, metas, distances), start=1):
            similarity = round(1 - dist, 4)

            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.write(f"**#{rank}** {meta.get('title', 'Untitled')}")
                st.caption(f"📅 {meta.get('date', 'n/a')} • {meta.get('event_type', 'n/a')}")

            with col2:
                st.metric("Similarity", f"{similarity:.2%}")


            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sentiment", f"{meta.get('sentiment_score', 0):.2f}")
            with col2:
                st.metric("Impact", f"{meta.get('impact_score', 0):.3f}")
            with col3:
                st.metric("Weight", f"{meta.get('event_weight', 0):.3f}")
            with col4:
                direction = meta.get("direction", "NEUTRAL")
                direction_emoji = {"POSITIVE": "📈", "NEGATIVE": "📉", "NEUTRAL": "➡️"}.get(direction, "")
                st.metric("Direction", f"{direction_emoji} {direction}")

            st.divider()

    return True


def render_prediction_results(current_price, stage1_pred, base_preds, stage2_pred, correction, news_features, market_returns):
    """Render final prediction results with enhanced UI"""

    st.divider()
    st.markdown("## 📊 Prediction Results")

    # Main Price Prediction Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Current Price",
            f"${current_price:.2f}",
            help="Today's closing price"
        )

    with col2:
        stage1_delta = stage1_pred - current_price
        st.metric(
            "Stage 1 Prediction",
            f"${stage1_pred:.2f}",
            delta=f"${stage1_delta:+.2f}",
            help="Ensemble (LSTM + CNN) prediction"
        )

    with col3:
        stage2_delta = stage2_pred - current_price
        st.metric(
            "Final Prediction",
            f"${stage2_pred:.2f}",
            delta=f"${stage2_delta:+.2f}",
            delta_color="inverse" if stage2_delta < 0 else "normal",
            help="After news-aware correction"
        )

    st.divider()

    # Stage 1 Base Learners
    st.markdown("### 🧠 Stage 1: Base Learners")
    col1, col2 = st.columns(2)
    with col1:
        lstm_delta = base_preds['lstm'] - current_price
        st.metric("LSTM Model", f"${base_preds['lstm']:.2f}", delta=f"${lstm_delta:+.2f}")
    with col2:
        cnn_delta = base_preds['cnn'] - current_price
        st.metric("CNN Model", f"${base_preds['cnn']:.2f}", delta=f"${cnn_delta:+.2f}")

    st.divider()

    # Stage 2 Correction
    st.markdown("### 📰 Stage 2: News-Aware Correction")

    col1, col2 = st.columns([3, 1])
    with col1:
        correction_pct = (correction / current_price) * 100
        if correction > 0.01:
            st.success(f"✅ Bullish Adjustment: +${correction:.2f} ({correction_pct:+.2f}%)")
        elif correction < -0.01:
            st.error(f"⚠️ Bearish Adjustment: ${correction:.2f} ({correction_pct:+.2f}%)")
        else:
            st.info(f"➡️ Neutral Adjustment: ${correction:.2f}")

    with col2:
        st.write("")

    # News Features
    st.markdown("### 📰 News Sentiment Inputs")
    news_col1, news_col2, news_col3, news_col4 = st.columns(4)

    with news_col2:
        sentiment = news_features['sentiment_score']
        sentiment_text = "Positive 📈" if sentiment > 0.3 else "Negative 📉" if sentiment < -0.3 else "Neutral ➡️"
        st.metric("Sentiment", sentiment_text, f"{sentiment:.2f}")

    with news_col3:
        st.metric("Impact Score", f"{news_features['impact_score']:.2f}")

    with news_col4:
        st.metric("Event Weight", f"{news_features['event_weight']:.2f}")

    st.divider()

    # Market Returns
    st.markdown("### 📈 Market Technical Indicators")
    market_col1, market_col2, market_col3 = st.columns(3)

    with market_col1:
        ret_1d = market_returns['return_1d'] * 100
        ret_1d_color = "green" if ret_1d > 0 else "red" if ret_1d < 0 else "gray"
        st.metric("1-Day Return", f"{ret_1d:+.2f}%")

    with market_col2:
        ret_5d = market_returns['return_5d'] * 100
        st.metric("5-Day Return", f"{ret_5d:+.2f}%")

    with market_col3:
        st.metric("📰 News Count", int(news_features['news_count']))

    # Risk Indicators
    st.markdown("### 🚨 Risk Indicators")
    risk_col1, risk_col2 = st.columns(2)

    with risk_col1:
        if news_features['has_supply_chain_event']:
            st.warning("⚠️ Supply Chain Event Detected")
        else:
            st.success("✓ No Supply Chain Issues")


    st.divider()

    # Final Recommendation
    expected_move = stage2_pred - current_price
    expected_move_pct = (expected_move / current_price) * 100

    st.markdown("## 🎯 Trading Recommendation")

    if abs(expected_move_pct) < 0.5:
        st.info(
            "### 📊 HOLD\n"
            f"Minimal price movement expected ({expected_move_pct:+.2f}%)\n"
            "Market appears stable with balanced news sentiment."
        )
    elif expected_move_pct > 2:
        st.success(
            "### 📈 STRONG BUY\n"
            f"Expected upside: **{expected_move_pct:+.2f}%**\n"
            f"Target price: **${stage2_pred:.2f}**\n"
            "Positive news sentiment and strong technical indicators."
        )
    elif expected_move_pct > 0.5:
        st.success(
            "### 📈 BUY\n"
            f"Expected upside: **{expected_move_pct:+.2f}%**\n"
            f"Target price: **${stage2_pred:.2f}**"
        )
    elif expected_move_pct < -2:
        st.error(
            "### 📉 STRONG SELL\n"
            f"Expected downside: **{expected_move_pct:.2f}%**\n"
            f"Target price: **${stage2_pred:.2f}**\n"
            "Negative news sentiment detected."
        )
    else:
        st.error(
            "### 📉 SELL\n"
            f"Expected downside: **{expected_move_pct:.2f}%**\n"
            f"Target price: **${stage2_pred:.2f}**"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Page configuration
    st.set_page_config(
        page_title="AAPL Stock AI Predictor",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom styling
    apply_custom_styling()

    # Header
    st.markdown("""
    <div class="hero-card">
        <h1>📈 AAPL Stock AI Predictor</h1>
        <p>Combine technical momentum with live news context to get a sharper next-day outlook for Apple.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='status-pill'>⚡ Stage 1: LSTM + CNN ensemble</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='status-pill'>🧠 Stage 2: News-aware correction</div>", unsafe_allow_html=True)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # SIDEBAR CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────────

    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        model_path = MODEL_DIR
        correction_model_filename = "Best_Adjustment_Model.pkl"

        news_api_key = NEWSAPI_KEY

        st.markdown("### 🎚️ Parameters")
        top_k = st.slider(
            "Similar events to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of ChromaDB matches for news analysis"
        )

        st.divider()

        if st.button("🔎 Test Database Connection", use_container_width=True):
            try:
                with st.spinner("🔄 Connecting to HuggingFace and ChromaDB..."):
                    chroma_path = get_or_create_local_chroma_cache(
                        HF_REPO_ID,
                        HF_COLLECTION_NAME,
                        LOCAL_CHROMA_CACHE
                    )
                test_result = load_chromadb_collection(str(chroma_path))
                if test_result["error"]:
                    st.error(f"❌ {test_result['error']}")
                else:
                    st.success(
                        f"✓ Connected\n\n"
                        f"'aapl_events' item count: **{test_result['count']:,}**\n"
                        f"📍 Cached at: `{chroma_path}`"
                    )
            except Exception as e:
                st.error(f"Failed to download/connect: {e}")

        run_prediction = st.button(
            "🚀 RUN PREDICTION",
            type="primary",
            use_container_width=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN EXECUTION
    # ─────────────────────────────────────────────────────────────────────────

    if run_prediction:
        progress_placeholder = st.empty()

        # Load models
        with progress_placeholder.container():
            st.info("🔄 Loading ML models...")

        models = load_models(model_path, correction_model_filename)
        if models is None:
            st.stop()

        lookback_window = models['lstm'].input_shape[1]
        cnn_lookback_window = models['cnn'].input_shape[1]

        # Download/Load ChromaDB from HuggingFace
        with progress_placeholder.container():
            st.info("🔄 Downloading ChromaDB from HuggingFace...")

        try:
            chroma_path = get_or_create_local_chroma_cache(
                HF_REPO_ID,
                HF_COLLECTION_NAME,
                LOCAL_CHROMA_CACHE
            )
            chroma_result = load_chromadb_collection(str(chroma_path))
            collection = chroma_result["collection"]
        except Exception as e:
            st.error(f"Failed to load ChromaDB from HuggingFace: {e}")
            st.stop()

        # Surface the connection diagnostics in the main body
        if chroma_result["error"]:
            st.warning(f"⚠️ ChromaDB: {chroma_result['error']}")
            st.info("ℹ️ The app will continue but news similarity search may return no matches.")
        else:
            st.success(
                f"✓ ChromaDB loaded from HuggingFace: **{chroma_result['count']:,} items** "
                f"in collection 'aapl_events'\n\n"
                f"📍 Cached at: `{chroma_result['path']}`"
            )

        # Load embedding model
        with progress_placeholder.container():
            st.info("🔄 Loading embedding model...")

        encoder = load_embedding_model()

        # Fetch OHLC data
        with progress_placeholder.container():
            st.info("📊 Fetching Apple price data...")

        ohlc_df = fetch_apple_ohlc(lookback=lookback_window)

        # NEW: fetch_apple_ohlc already shows a specific st.error and
        # returns None for empty responses, too-few-rows, or any NaN OHLC
        # values, so we can just stop here.
        if ohlc_df is None:
            progress_placeholder.empty()
            st.stop()

        progress_placeholder.empty()

        current_price = float(ohlc_df.iloc[-1]['close'])
        last_date = ohlc_df.iloc[-1]['date']

        # NEW: belt-and-suspenders guard in case current_price ends up NaN
        # via any future code path that bypasses fetch_apple_ohlc's checks.
        if np.isnan(current_price):
            st.error(
                "⚠️ **Current AAPL price is unavailable (NaN).** Stopping "
                "before running predictions on invalid data. Please try "
                "again in a few minutes — this is usually a temporary "
                "Yahoo Finance data issue."
            )
            st.stop()

        # Display current price
        col1, col2 = st.columns([2, 1])
        with col1:
            st.success(f"✓ Current AAPL Price: **${current_price:.2f}**")
        with col2:
            st.caption(f"As of {last_date.date()}")

        st.divider()

        # Market returns
        market_returns = compute_market_returns(ohlc_df)

        # Prepare OHLC features
        ohlc_for_lstm = prepare_ohlc_features(ohlc_df, models['feature_scaler_raw'], lookback=lookback_window)
        ohlc_for_cnn = prepare_ohlc_features(ohlc_df, models['feature_scaler'], lookback=lookback_window)

        if ohlc_for_lstm is None or ohlc_for_cnn is None:
            st.stop()

        # Stage 1 Prediction
        st.markdown("### Stage 1️⃣: Ensemble Prediction")
        stage1_pred, base_preds, stage1_pred_scaled = get_stage1_prediction(models, ohlc_for_lstm, ohlc_for_cnn)

        if stage1_pred is None:
            st.stop()

        st.success(f"✓ Stage 1 Prediction: **${stage1_pred:.2f}**")

        st.divider()

        # News Analysis
        st.markdown("### 📰 News Analysis")

        articles = fetch_newsapi_articles(news_api_key, num_articles=3)

        if not articles:
            st.warning("⚠️ No articles found from NewsAPI. Using neutral news features.")
            aggregated_news_features = _default_news_features()
        else:
            st.success(f"✓ Fetched {len(articles)} latest Apple news articles")

            all_news_features = []

            for idx, article in enumerate(articles, start=1):
                with st.expander(f"📰 Article {idx}: {article['title'][:60]}...", expanded=(idx == 1)):
                    st.caption(f"🔗 {article['source']} • {article['published_at']}")

                    if article['url']:
                        st.markdown(f"[Read full article]({article['url']})")

                    with st.spinner(f"Searching ChromaDB..."):
                        results, count = find_similar_events(collection, encoder, article['text'], top_k=top_k)

                    if count > 0 and collection is not None:
                        st.info(f"📊 Found {count:,} similar events in ChromaDB")
                        render_news_results(idx, article, results)
                        news_feats = extract_news_features(results)
                        all_news_features.append(news_feats)
                    else:
                        st.info("ℹ️ No ChromaDB matches available for this article")
                        all_news_features.append(_default_news_features())

            if all_news_features:
                feature_keys = all_news_features[0].keys()
                aggregated_news_features = {}
                for key in feature_keys:
                    values = [f[key] for f in all_news_features]
                    aggregated_news_features[key] = np.mean(values)
            else:
                aggregated_news_features = _default_news_features()

        st.divider()

        # Stage 2 Prediction
        st.markdown("### Stage 2️⃣: News Correction")
        stage2_pred, correction = get_stage2_prediction(
            models,
            stage1_pred,
            aggregated_news_features,
            market_returns,
        )

        if stage2_pred is None:
            st.stop()

        st.success(f"✓ Stage 2 Final Prediction: **${stage2_pred:.2f}**")

        st.divider()

        # Render results
        render_prediction_results(
            current_price,
            stage1_pred,
            base_preds,
            stage2_pred,
            correction,
            aggregated_news_features,
            market_returns,
        )

        # Export options
        st.divider()
        st.markdown("## 💾 Export Results")

        export_data = {
            "Timestamp": datetime.now().isoformat(),
            "Current_Price": current_price,
            "Stage1_Prediction": stage1_pred,
            "Stage2_Prediction": stage2_pred,
            "Correction": correction,
            "Expected_Move_$": stage2_pred - current_price,
            "Expected_Move_%": ((stage2_pred - current_price) / current_price) * 100,
        }

        col1, col2 = st.columns(2)

        with col1:
            csv = pd.DataFrame([export_data]).to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"aapl_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            json_str = pd.DataFrame([export_data]).to_json(orient='records')
            st.download_button(
                label="📥 Download as JSON",
                data=json_str,
                file_name=f"aapl_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

    else:
        st.markdown("""
        ---
        ### 🚀 Getting Started

        1. Click **"🔎 Test Database Connection"** to confirm everything is
           wired up correctly before running a full prediction
        2. **Click** "RUN PREDICTION" to execute the full pipeline
        3. **Review** Stage 1 & Stage 2 predictions with news analysis
        4. **Export** results as CSV or JSON

        ### 📖 How It Works

        **Stage 1**: LSTM + CNN ensemble analyzes 30 days of OHLC price data
        **Stage 2**: News-aware correction refines prediction using:
        - Market technical indicators (1-day, 5-day returns)
        - Feature aggregation via vector similarity

        ### 📦 Data Source
        - **ChromaDB Collection**: Downloaded from HuggingFace Hub (`alokyadav310703/Similarity`)
        - **Caching**: Automatically cached locally for faster subsequent runs
        - **Price Data**: Fetched live from Yahoo Finance and validated — if
          the feed is down or returns incomplete/NaN data, you'll see a
          clear error message instead of a broken prediction.

        ---
        *Made with ❤️ for stock prediction enthusiasts*
        """)


if __name__ == "__main__":
    main()