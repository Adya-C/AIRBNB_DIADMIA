# 🏠 Airbnb Revenue Optimization Analytics Platform

> A data-driven analytics dashboard for Dubai short-term rental hosts — powered by machine learning, built with Streamlit.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://airbnbdiadmia.streamlit.app/)

---

## 📌 Project Overview

This platform analyses 3,500 synthetic Airbnb listings from the Dubai market and applies machine learning techniques to help hosts **maximize occupancy, optimize pricing, and forecast revenue**.

Built as a university data analytics project, the dashboard covers the full analytics pipeline — from exploratory data analysis through to classification, clustering, association rule mining, regression, and demand forecasting — all wrapped in an interactive, filter-driven interface with a live **Host Simulator**.

---

## 🚀 Live Dashboard

👉 **[https://airbnbdiadmia.streamlit.app/](https://airbnbdiadmia.streamlit.app/)**

---

## 📁 Repository Structure

```
├── app.py                  # Main Streamlit dashboard (11 pages)
├── data_generator.py       # Synthetic dataset generator (3,500 listings, seed=42)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .streamlit/
    └── config.toml         # Light theme configuration
```

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Home | KPI summary, market snapshot charts |
| 📋 Dataset Overview | Data preview, column types, distributions |
| 🔍 Exploratory Analysis | Interactive scatter, box, heatmap charts with live sidebar filters |
| 💰 Pricing Analytics | Neighborhood, seasonal, day-of-week pricing breakdowns |
| 🤖 Classification Models | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting |
| 🎯 Clustering Analysis | K-Means (K=4) market segmentation with radar profiles |
| 🔗 Association Rules | Apriori algorithm — high-lift amenity combination discovery |
| 📈 Regression Modeling | Ridge regression price prediction with feature coefficient analysis |
| 📅 Demand Forecasting | Monthly demand trends + 6-month polynomial forecast |
| 🚀 Revenue Optimization | Price-revenue curve, neighborhood heatmap, amenity ROI |
| 🧮 Host Simulator | Enter your listing → get revenue forecast, pricing verdict & strategy report |

---

## 🤖 Machine Learning Techniques

| Technique | Algorithm | Purpose |
|-----------|-----------|---------|
| Classification | Random Forest, Gradient Boosting, Decision Tree, Logistic Regression | Predict booking probability |
| Clustering | K-Means (K=4) | Segment listings into market archetypes |
| Association Rules | Apriori (MLxtend) | Discover high-ROI amenity bundles |
| Regression | Ridge Regression (L2) | Predict optimal nightly price |
| Forecasting | Polynomial Regression | 12-month demand projection |

---

## 📦 Dataset

- **Source:** Synthetic — generated programmatically using NumPy (seed=42)
- **Size:** 3,500 listings × 33 variables
- **Geography:** Dubai, UAE — 15 distinct neighborhoods
- **Target variable:** `Booking_Status` (Booked / Not Booked)

Key variable groups:
- Property characteristics (type, bedrooms, accommodates)
- Pricing (nightly rate, cleaning fee, minimum nights)
- Amenities (WiFi, pool, kitchen, parking, self check-in, air conditioning)
- Host quality (review score, response rate, Superhost status)
- Demand signals (season, month, day of week, traveller type, lead time)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io/) | Dashboard framework |
| [Pandas](https://pandas.pydata.org/) | Data manipulation |
| [NumPy](https://numpy.org/) | Numerical computing |
| [Scikit-learn](https://scikit-learn.org/) | ML models |
| [MLxtend](http://rasbt.github.io/mlxtend/) | Association rule mining |
| [Plotly](https://plotly.com/python/) | Interactive charts |
| [XGBoost](https://xgboost.readthedocs.io/) | Gradient boosting (optional) |

---

## ⚙️ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## ☁️ Deploy to Streamlit Cloud

1. Push all files to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Click **New app** → connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** — live in ~60 seconds

> ⚠️ Ensure `config.toml` is inside the `.streamlit/` subfolder, not the root.

---

## 📋 Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
mlxtend>=0.22.0
plotly>=5.15.0
xgboost>=1.7.0
scipy>=1.10.0
```

---

## 👤 Author

University Data Analytics Project — Dubai Airbnb Revenue Optimization  
Dashboard: [https://airbnbdiadmia.streamlit.app/](https://airbnbdiadmia.streamlit.app/)
