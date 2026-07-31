## A Multi-Stage Deep Learning Framework for Stock Price Prediction Using LLM-Based News Intelligence and Error Correction

Live Demo: https://aaplstockprediction.streamlit.app/

A three-stage, memory-augmented, event-driven stock prediction pipeline for Apple Inc. (AAPL) that fuses LLM-extracted news intelligence, ensemble deep-learning price forecasting, and a learned residual error-correction model.

---

## Overview

Traditional stock prediction models rely on historical price data (OHLCV) and ignore the real-world events that actually drive price movements. This system takes a different approach: it separates **news-driven prediction** from **price-driven prediction**, and then learns to correct the price model's errors using the news signal.

Each trading day, the system:
1. Collects the previous day's Apple news and extracts structured event data via LLM
2. Embeds and stores each event in a persistent, importance-weighted vector database (ChromaDB)
3. Runs an ensemble of LSTM/CNN models on wavelet-denoised OHLC data to forecast next-day price
4. Fuses the day's news features with market technical indicators and the model's historical error record, and trains a Random Forest to predict — and correct — the residual error of the price forecast
5. Outputs a final adjusted price, a directional call (**HIGH** / **LOW** / **NEUTRAL**), and a natural-language explanation
6. After market close, fetches the actual outcome and retrains all stages incrementally

This design is validated end-to-end on a 121-day, fully out-of-sample 2025 test window, where the correction stage improved directional accuracy from 58.3% to 77.5% (see [Results](#evaluation-results)).

---

## Key Features

- **Three-stage architecture** — news processing, price forecasting, and error correction each operate on their native signal and are trained independently, then fused at inference time
- **Event-driven prediction** — forecasts based on news cause-effect patterns, not price autocorrelation alone
- **Importance-weighted memory** — ChromaDB stores not just what happened, but how much it moved the market (predicted impact score, event weight, time-decay)
- **Denoised ensemble forecasting** — wavelet-denoised (Coiflet-3) LSTM + CNN stacking ensemble for the base price prediction
- **Learned residual correction** — a benchmarked Random Forest Regressor corrects the base model's prediction error using fused news + market + historical-error features
- **Online / incremental learning** — all three stages are retrained on the accumulated dataset after every trading day
- **Explainability-first design** — every prediction includes retrieved past events, similarity scores, feature importances, and an LLM-generated reasoning paragraph
- **Fully automated daily pipeline** — APScheduler drives all phases from ingestion to retraining
- **No look-ahead bias** — all train/validation/test splits are strictly chronological (no shuffling) at every stage

---

## System Architecture

### Daily Pipeline (Production Schedule)
<!--
| Phase | Time | Description |
|---|---|---|
| 1 — Data Ingestion | 6:00 AM | Collect Apple news, SEC filings, AAPL OHLC price data |
| 2 — LLM Event Extraction | 6:00 AM | Parse articles into structured JSON events (event type, severity, affected product) |
| 3 — Vector Memory Update | 6:00 AM | Embed events with a sentence-transformer, store in ChromaDB with computed impact score & event weight |
| 4 — Stage 2 Price Forecast | 7:00 AM | Denoised LSTM/CNN stacking ensemble predicts next-day close |
| 5 — Stage 3 Error Correction | 7:00 AM | Random Forest fuses news + technical + historical-error features to correct the Stage 2 forecast |
| 6 — Online Learning | 5:00 PM | Fetch actual outcome, update memory and error dataset, retrain Stage 1/2/3 models |
| 7 — Dashboard | Always on | Streamlit UI for predictions, accuracy tracking, and log browsing |
-->
### Stage 1 — LLM News Extraction & Vector Storage

1. **Filtering & loading** — articles mentioning AAPL are pulled from batch sources and truncated to 2,000 characters
2. **Label generation** — each article is aligned to its trading day, and the forward price change is computed at T+1, T+3, and T+5 using the *positional* trading-day index (so weekends/holidays don't distort horizons). The T+1 change is thresholded into **HIGH** (>+0.5%), **LOW** (<−0.5%), or **NEUTRAL** (in between)
3. **Sentiment scoring** — a lightweight local polarity score (TextBlob) is computed as a baseline signal that needs no external API call
4. **LLM feature extraction** — an LLM returns a validated JSON object per article with `event_type` (one of 10 categories), `severity` (0–10), and `affected_product`
5. **Impact-score modeling** — an XGBoost Regressor (300 trees, max depth 5, early stopping) predicts a continuous impact score per article from accumulated historical data, trained incrementally each session
6. **Event weighting** — the predicted impact score is combined with an event-type multiplier and an exponential time-decay (365-day half-life) to produce the final `event_weight` used for retrieval ranking
7. **Vector storage** — articles are embedded with a sentence-transformer model, indexed in ChromaDB by cosine similarity, with full metadata (date, event type, severity, affected product, sentiment, impact score, event weight, T+1 label) attached; duplicate articles are deduplicated before insertion

### Stage 2 — Ensemble Time-Series Price Forecasting

Operates independently on historical OHLC data:

- **Denoising** — Coiflet-3 wavelet denoising smooths short-term noise from 2010–2023 daily price data while preserving trend
- **Scaling & windowing** — raw and denoised series are each MinMax-scaled and windowed with a 20-day lookback
- **Chronological split** — 60% train / 15% validation / 25% test, no shuffling
- **Base models**:
  - **LSTM** (raw data) — two stacked 64-unit LSTM layers, 10% dropout after each, 32-unit dense head, Adam @ lr 0.0001 — captures long-range temporal dependencies
  - **CNN** (denoised data) — two 1D conv layers (64 filters, kernel 2, causal padding), max-pool, dense head, Adam @ lr 0.001 — captures local patterns
- **Stacking ensemble** — a linear regression meta-learner combines LSTM-Raw, CNN-Raw, and CNN-Denoised predictions, automatically weighting whichever model fits current conditions best

### Stage 3 — Residual Error Correction via Fused-Feature Regression

No single model is perfect, so Stage 3 learns to predict *how wrong* Stage 2 will be, using a fused feature set:

- **News features** (from Stage 1): `severity`, `sentiment_score`, `impact_score`, `event_weight`
- **Market technical indicators**: `return_1d`, `return_5d`
- **Historical error dataset**: Stage 2's actual price, predicted price, and error for every past trading day

The regression target is the Stage-2 prediction error (`Actual Price − Stage-2 Predicted Price`). Data is split chronologically (80% train / 20% test, no shuffling).

**Model benchmark** — seven candidate regressors are trained and compared on the identical split: Linear Regression, Random Forest (300 est.), Gradient Boosting, Extra Trees (300 est.), HistGradientBoosting, k-NN (k=5), and XGBoost (300 est., lr 0.05, depth 5). The best performer by held-out R² is persisted as the production correction model.

**Winner: Random Forest Regressor** — test R² of **0.7008**. Feature importance:

| Feature | Importance |
|---|---|
| `return_1d` | 0.6367 |
| `return_5d` | 0.1856 |
| `event_weight` | 0.0785 |
| `impact_score` | 0.0442 |
| `sentiment_score` | 0.0399 |
| `severity` | 0.0151 |

Price momentum dominates (~83% combined), but the four LLM-derived news features contribute a non-trivial ~17% of complementary signal.

**Correction applied at inference**:

```
Adjusted Price = Stage-2 Predicted Price − Stage-3 Predicted Error
```

---

## Technology Stack

| Component | Tool |
|---|---|
| Scheduler | APScheduler (Python) |
| News Ingestion | NewsAPI, PRAW, requests |
| LLM Extraction | Claude Haiku (Anthropic API) / Llama-3.1-8B via Groq |
| Baseline Sentiment | TextBlob |
| Embeddings | Sentence-transformer (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| Impact-Score Model | XGBoost Regressor |
| Price Forecasting | LSTM + CNN ensemble (stacked, linear meta-learner) |
| Signal Denoising | Coiflet-3 wavelet transform |
| Residual Correction | Random Forest Regressor (benchmarked vs. 6 alternatives) |
| Price Data | yfinance / Yahoo Finance |
| Log Storage | SQLite |
| Dashboard | Streamlit |
| Experiment Tracking | MLflow |
| Deployment | Railway.app / Render.com |

---

## Data Sources

**Primary historical dataset (cold start):**
[FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) — ~180,000 AAPL-specific news records (1999–2023), pre-aligned with OHLCV price data. A minimum of 500 labelled events in ChromaDB is required before reliable predictions can be made.

**Historical price data:** Yahoo Finance, 2010–2023 (~3,500 trading days; ~3,480 windowed examples after wavelet denoising, split 2,088 train / 522 validation / 870 test).

**Live data sources:**

| Source | Role |
|---|---|
| NewsAPI / GNews | Daily Apple news headlines |
| yfinance | AAPL daily OHLCV for outcome labelling |

---

## Event Extraction Schema

Every article is parsed by the LLM into a validated JSON object:

| Field | Description |
|---|---|
| `event_type` | One of 10 categories: `earnings_beat`, `earnings_miss`, `product_launch`, `legal`, `supply_chain`, `analyst_upgrade`, `analyst_downgrade`, `ceo_statement`, `regulatory`, `general` |
| `severity` | Estimated significance, 0.0–10.0 |
| `affected_product` | iPhone, Mac, Services, App Store, Vision Pro, or General |
| `sentiment_score` | Float −1.0 to +1.0, computed locally via TextBlob (not LLM-derived) |
| `market_context` | `bull_market`, `bear_market`, `high_volatility`, `neutral` |
| `source_credibility` | Tier 1 (Reuters/WSJ), Tier 2 (CNBC), Tier 3 (blogs) |

---

## Importance-Weighted Memory

The vector memory stores three importance numbers alongside every event embedding:

- **`price_impact_pct`** — actual AAPL price change at T+1, T+3, T+5 days after the event
- **`impact_score`** — predicted by the Stage 1 XGBoost Regressor from article features
- **`event_weight`** — retrieval multiplier combining the predicted impact score with an event-type bonus and an exponential time-decay (default half-life: 365 days)

---

## Prediction Engine (Retrieval)

For each new article, the engine:
1. Embeds the article with the sentence-transformer model
2. Queries ChromaDB for the top-10 most similar past events by cosine similarity
3. Re-ranks using **60% cosine similarity + 40% normalised event_weight**
4. Feeds the top-5 retrieved events into the downstream models
5. Generates a natural-language explanation via LLM

---

## Evaluation Results

### Stage 2 — Base Price Forecast (2010–2023 data)

| Model | Data | RMSE | MAE | R² |
|---|---|---|---|---|
| LSTM | Raw | 0.0236 | 0.0188 | 0.9641 |
| LSTM | Denoised | 0.2452 | 0.2370 | −2.8685 |
| CNN | Raw | 0.0521 | 0.0411 | 0.8250 |
| CNN | Denoised | 0.0453 | 0.0353 | 0.8679 |
| **Stacking Ensemble** | Mixed | **0.0249** | 0.0197 | **0.9602** |

Denoising helped the CNN (13.1% RMSE reduction) but badly hurt the LSTM (negative R²) — wavelet smoothing removes short-term fluctuations the LSTM's recurrent structure relies on. The ensemble nearly matches the best single model while being more robust to any one architecture's failure mode.

### Stage 2 vs. Stage 2+3 — End-to-End, Out-of-Sample (121 trading days, Jan 2 – Jun 27, 2025)

| Metric | Stage 2 (Baseline) | Stage 2+3 (Corrected) | Improvement |
|---|---|---|---|
| MSE | 54.631 | 23.836 | −56.4% |
| RMSE | 7.391 | 4.882 | −34.0% |
| MAE | 5.568 | 3.486 | −37.4% |
| R² | 0.818 | 0.921 | +12.5% (relative) |
| MAPE | 2.63% | 1.66% | −37.0% |
| Directional Accuracy | 58.33% | 77.50% | +19.17 pts |
| Hit Rate @1% | 23.97% | 43.80% | +19.83 pts |
| Hit Rate @2% | 48.76% | 71.07% | +22.31 pts |

The residual correction stage improves **both** magnitude accuracy and directional accuracy simultaneously on a genuinely held-out 2025 window.

