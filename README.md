# 🏠 Airbnb Revenue Optimization Analytics Dashboard

A professional, data-driven analytics platform for Dubai Airbnb hosts and property managers — built with Streamlit, Scikit-learn, and Plotly.

## 🎯 Project Overview

This dashboard answers the critical questions every Airbnb host must resolve:
- What **factors** drive bookings?
- What **price** maximizes revenue?
- Which **amenities** increase occupancy?
- What **traveler segments** exist?
- What amenity **combinations** correlate with performance?
- How can hosts **optimize listings** to increase revenue?

## 📊 Dashboard Pages

| Page | Technique | Description |
|------|-----------|-------------|
| 🏠 Home | KPI Overview | Key metrics, market snapshot |
| 📋 Dataset Overview | EDA | 3,500 synthetic Dubai listings |
| 🔍 Exploratory Analysis | Visualization | Correlations, distributions, heatmaps |
| 💰 Pricing Analytics | Business Analytics | Seasonal & neighborhood pricing |
| 🤖 Classification Models | ML | Predict Booking_Status (LR, DT, RF, XGBoost) |
| 🎯 Clustering Analysis | K-Means | Segment listings into archetypes |
| 🔗 Association Rule Mining | Apriori | Discover amenity combinations |
| 📈 Regression Modeling | ML | Predict Price_Per_Night |
| 📅 Demand Forecasting | Time-series | Monthly demand patterns + forecast |
| 🚀 Revenue Optimization | Analytics | Optimal pricing, amenity ROI |

## 🚀 Quick Start

### Local Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/airbnb-analytics.git
cd airbnb-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app auto-generates a synthetic dataset of 3,500 Dubai Airbnb listings on startup — **no external data files needed**.

### Deploy to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo, set **Main file path** to `app.py`
5. Click **Deploy** 🚀

## 🛠️ Tech Stack

- **Streamlit** — Dashboard framework
- **Pandas / NumPy** — Data manipulation
- **Scikit-learn** — ML models (classification, clustering, regression)
- **MLxtend** — Apriori association rule mining
- **Plotly** — Interactive visualizations
- **XGBoost** — Gradient boosting classifier

## 📁 Project Structure

```
airbnb-analytics/
├── app.py              # Main Streamlit app (all pages)
├── data_generator.py   # Synthetic dataset generator
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── config.toml     # Theme configuration
└── README.md
```

## 🎨 Dataset Features

The synthetic dataset includes 33 features across 5 categories:

- **Property**: Type, Bedrooms, Bathrooms, Accommodates, Room Type
- **Location**: Neighborhood, Distance to City Center, Distance to Metro
- **Amenities**: WiFi, Kitchen, Pool, Parking, Self Check-in, Air Con
- **Performance**: Occupancy Rate, Review Score, Number of Reviews
- **Time**: Season, Month, Day of Week, Booking Lead Time

**Target Variable**: `Booking_Status` (Booked / Not Booked)

## 📈 Analytics Techniques

1. **Synthetic Data Generation** — Realistic Dubai market patterns
2. **Exploratory Data Analysis** — Correlation heatmaps, distributions
3. **Classification** — LR, Decision Tree, Random Forest, XGBoost
4. **Clustering** — K-Means with 4 market segments
5. **Association Rule Mining** — Apriori on amenity combinations
6. **Regression** — Linear, Ridge, Lasso for price prediction
7. **Demand Forecasting** — Polynomial regression on monthly trends
8. **Revenue Optimization** — Price-revenue curve, amenity ROI

---

Built for a university data analytics project · Dubai Airbnb Market Analysis 2024
