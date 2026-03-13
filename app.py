"""
app.py  –  Airbnb Revenue Optimization Analytics Dashboard
Light theme · Dynamic filters · Host Simulator
"""

# ── Standard imports ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix,
                              mean_absolute_error, r2_score)
from sklearn.cluster import KMeans

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from mlxtend.frequent_patterns import apriori, association_rules
from data_generator import generate_airbnb_data

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airbnb Analytics | Dubai",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  – clean light theme, no colour clashes
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }

/* hide default chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; }

/* ── sidebar ── */
[data-testid="stSidebar"] { background: #16213E; }
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    color: #E2E8F0 !important; font-size: 0.83rem; padding: 5px 8px;
    border-radius: 6px; margin-bottom: 2px;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(255,90,95,0.15);
}
[data-testid="stSidebar"] h3 { color: #FF5A5F !important; }

/* ── KPI cards ── */
.kcard {
    background: #fff;
    border: 1px solid #E2E8F0;
    border-top: 3px solid #FF5A5F;
    border-radius: 10px;
    padding: 16px 18px;
    text-align: center;
    box-shadow: 0 1px 6px rgba(0,0,0,.06);
}
.kcard-val { font-size: 1.6rem; font-weight: 700; color: #FF5A5F; line-height: 1.2; }
.kcard-lbl { font-size: 0.72rem; color: #64748B; margin-top: 3px;
             text-transform: uppercase; letter-spacing: .5px; font-weight: 500; }

/* ── Section label ── */
.slabel {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.88rem; font-weight: 600; color: #1E293B;
    margin: 18px 0 8px; border-bottom: 2px solid #F1F5F9; padding-bottom: 6px;
}

/* ── insight box ── */
.ibox {
    background: #F0FDF4; border-left: 3px solid #22C55E;
    border-radius: 0 6px 6px 0; padding: 9px 13px;
    font-size: 0.82rem; color: #166534; margin-top: 4px; line-height: 1.55;
}

/* ── warning / info box ── */
.wbox {
    background: #FFFBEB; border-left: 3px solid #F59E0B;
    border-radius: 0 6px 6px 0; padding: 9px 13px;
    font-size: 0.82rem; color: #92400E; margin-top: 4px;
}
.ebox {
    background: #FEF2F2; border-left: 3px solid #EF4444;
    border-radius: 0 6px 6px 0; padding: 9px 13px;
    font-size: 0.82rem; color: #991B1B; margin-top: 4px;
}

/* ── hero banner ── */
.hero {
    background: linear-gradient(120deg, #FF5A5F 0%, #C0392B 100%);
    border-radius: 12px; padding: 26px 32px; margin-bottom: 22px;
}
.hero-title { font-size: 1.65rem; font-weight: 700; color: #fff; margin: 0 0 5px; }
.hero-sub   { font-size: 0.9rem; color: rgba(255,255,255,.85); margin: 0; }

/* ── recommendation card ── */
.rcard {
    background: #fff; border: 1px solid #E2E8F0;
    border-left: 4px solid #FF5A5F; border-radius: 0 10px 10px 0;
    padding: 14px 18px; margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,.04);
}
.rcard-title { font-weight: 600; font-size: 0.88rem; color: #1E293B; margin-bottom: 3px; }
.rcard-body  { font-size: 0.8rem; color: #475569; line-height: 1.55; }

/* ── simulator result highlight ── */
.simbox {
    background: linear-gradient(135deg,#FFF5F5,#FFF);
    border: 1px solid #FECACA; border-radius: 10px;
    padding: 18px 22px; text-align: center; margin-bottom: 6px;
}
.simbox-big { font-size: 2.2rem; font-weight: 700; color: #FF5A5F; }
.simbox-lbl { font-size: 0.74rem; color: #64748B; text-transform: uppercase;
              letter-spacing: .5px; font-weight: 600; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
PALETTE  = ["#FF5A5F","#3B82F6","#22C55E","#F59E0B","#8B5CF6","#EC4899","#14B8A6","#F97316"]
C_RED    = "#FF5A5F"
C_BLUE   = "#3B82F6"
C_GREEN  = "#22C55E"
C_AMBER  = "#F59E0B"
C_NAVY   = "#1E293B"
C_GREY   = "#64748B"

PBASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", size=11, color="#374151"),
    margin=dict(t=36, b=16, l=8, r=8),
)
AXIS = dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False,
            linecolor="#E2E8F0", tickfont=dict(size=10))

def _layout(fig, h=340, **kw):
    fig.update_layout(**PBASE, height=h,
                      xaxis={**AXIS}, yaxis={**AXIS}, **kw)
    return fig

def kcard(val, lbl):
    return f"<div class='kcard'><div class='kcard-val'>{val}</div><div class='kcard-lbl'>{lbl}</div></div>"

def hero(title, sub):
    st.markdown(f"<div class='hero'><div class='hero-title'>{title}</div>"
                f"<div class='hero-sub'>{sub}</div></div>", unsafe_allow_html=True)

def slabel(txt):
    st.markdown(f"<div class='slabel'>{txt}</div>", unsafe_allow_html=True)

def ibox(txt):
    st.markdown(f"<div class='ibox'>💡 {txt}</div>", unsafe_allow_html=True)

def wbox(txt):
    st.markdown(f"<div class='wbox'>⚠️ {txt}</div>", unsafe_allow_html=True)

def ebox(txt):
    st.markdown(f"<div class='ebox'>❌ {txt}</div>", unsafe_allow_html=True)

MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODELS  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = generate_airbnb_data(3500, seed=42)
    return df

@st.cache_resource(show_spinner=False)
def train_models(df_hash):
    df = load_data()

    FEAT_CLF = ["Price_Per_Night","Review_Score","Occupancy_Rate","Amenities_Count",
                "Distance_to_City_Center","Superhost_Status","Host_Response_Rate",
                "Number_of_Reviews","Bedrooms","Wifi","Pool","Parking",
                "Booking_Lead_Time","Length_of_Stay"]
    FEAT_REG = ["Bedrooms","Bathrooms","Accommodates","Amenities_Count","Review_Score",
                "Superhost_Status","Distance_to_City_Center","Wifi","Kitchen",
                "Pool","Parking","Air_Conditioning","Host_Response_Rate","Occupancy_Rate"]
    FEAT_CLU = ["Price_Per_Night","Amenities_Count","Review_Score",
                "Bedrooms","Distance_to_City_Center","Occupancy_Rate"]

    y_clf = (df["Booking_Status"] == "Booked").astype(int)
    y_reg = df["Price_Per_Night"]

    sc_clf = StandardScaler().fit(df[FEAT_CLF])
    sc_reg = StandardScaler().fit(df[FEAT_REG])
    sc_clu = StandardScaler().fit(df[FEAT_CLU])

    X_clf  = df[FEAT_CLF].values
    X_clf_s = sc_clf.transform(X_clf)
    X_reg_s = sc_reg.transform(df[FEAT_REG])
    X_clu_s = sc_clu.transform(df[FEAT_CLU])

    # Classification models
    clf_defs = {
        "Logistic Regression":  (LogisticRegression(max_iter=500),                        True),
        "Decision Tree":         (DecisionTreeClassifier(max_depth=7, random_state=42),    False),
        "Random Forest":         (RandomForestClassifier(n_estimators=150, random_state=42), False),
        "Gradient Boosting":     (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
    }
    if HAS_XGB:
        clf_defs["XGBoost"] = (XGBClassifier(n_estimators=100, random_state=42,
                                              eval_metric="logloss", verbosity=0), False)

    clf_models, clf_metrics, cms = {}, {}, {}
    for name, (mdl, scaled) in clf_defs.items():
        X_in = X_clf_s if scaled else X_clf
        mdl.fit(X_in, y_clf)
        yp = mdl.predict(X_in)
        clf_models[name]  = mdl
        clf_metrics[name] = {
            "Accuracy":  round(accuracy_score(y_clf, yp)*100, 1),
            "Precision": round(precision_score(y_clf, yp, zero_division=0)*100, 1),
            "Recall":    round(recall_score(y_clf, yp, zero_division=0)*100, 1),
            "F1 Score":  round(f1_score(y_clf, yp, zero_division=0)*100, 1),
        }
        cms[name] = confusion_matrix(y_clf, yp)

    rf = clf_models["Random Forest"]
    feat_imp = pd.Series(rf.feature_importances_, index=FEAT_CLF).sort_values(ascending=True)

    # Regression
    ridge = Ridge(alpha=10).fit(X_reg_s, y_reg)
    lasso = Lasso(alpha=5,  max_iter=2000).fit(X_reg_s, y_reg)
    lr    = LogisticRegression(max_iter=500)   # placeholder label

    r2  = round(r2_score(y_reg, ridge.predict(X_reg_s)), 3)
    mae = round(mean_absolute_error(y_reg, ridge.predict(X_reg_s)), 1)

    # Clustering  K=4
    km     = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(X_clu_s)

    return dict(
        FEAT_CLF=FEAT_CLF, FEAT_REG=FEAT_REG, FEAT_CLU=FEAT_CLU,
        sc_clf=sc_clf, sc_reg=sc_reg, sc_clu=sc_clu,
        clf_models=clf_models, clf_metrics=clf_metrics, cms=cms,
        feat_imp=feat_imp,
        ridge=ridge, lasso=lasso,
        r2=r2, mae=mae,
        km=km, km_labels=labels,
    )

@st.cache_data(show_spinner=False)
def run_apriori(_df):
    am = _df[["Wifi","Kitchen","Air_Conditioning","Parking","Self_Checkin","Pool"]].astype(bool).copy()
    am["High_Occupancy"] = _df["Occupancy_Rate"] > 0.70
    am["High_Price"]     = _df["Price_Per_Night"] > _df["Price_Per_Night"].median()
    am["Superhost"]      = _df["Superhost_Status"].astype(bool)
    am["Long_Stay"]      = _df["Length_of_Stay"] > 5
    freq  = apriori(am, min_support=0.10, use_colnames=True)
    rules = association_rules(freq, metric="lift", min_threshold=1.05,
                              num_itemsets=len(freq))
    rules = rules.sort_values("lift", ascending=False).head(30).copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda x: " + ".join(sorted(x)))
    rules["consequents"]  = rules["consequents"].apply(lambda x: " + ".join(sorted(x)))
    return rules

# ─────────────────────────────────────────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("⏳ Loading data & training models (first run only)…"):
    DF = load_data()
    M  = train_models(42)          # static hash – models are deterministic

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  – navigation + global filters
# ─────────────────────────────────────────────────────────────────────────────
PAGES = [
    "🏠  Home",
    "📋  Dataset Overview",
    "🔍  Exploratory Analysis",
    "💰  Pricing Analytics",
    "🤖  Classification Models",
    "🎯  Clustering Analysis",
    "🔗  Association Rules",
    "📈  Regression Modeling",
    "📅  Demand Forecasting",
    "🚀  Revenue Optimization",
    "🧮  Host Simulator",
]

FILTER_PAGES = {"Exploratory Analysis","Pricing Analytics",
                "Revenue Optimization","Dataset Overview","Demand Forecasting"}

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:14px 0 6px'>
        <span style='font-size:2rem'>🏠</span><br>
        <span style='font-size:1rem;font-weight:700;color:#FF5A5F'>Airbnb Analytics</span><br>
        <span style='font-size:0.7rem;color:#94A3B8'>Dubai Revenue Intelligence</span>
    </div>
    <hr style='border-color:#2D3A5A;margin:8px 0 14px'/>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    pname = page.split("  ", 1)[-1].strip()

    st.markdown("<hr style='border-color:#2D3A5A;margin:12px 0 10px'/>", unsafe_allow_html=True)

    # ── Global filters (only shown on relevant pages) ──────────────────────
    df = DF.copy()  # default: full dataset

    if pname in FILTER_PAGES:
        st.markdown("<div style='font-size:0.7rem;color:#94A3B8;letter-spacing:.8px;text-transform:uppercase;margin-bottom:6px'>🔧 Filters</div>", unsafe_allow_html=True)

        f_hoods = st.multiselect("Neighborhood", sorted(DF["Neighborhood"].unique()),
                                  placeholder="All", key="f_hood")
        f_props = st.multiselect("Property Type", sorted(DF["Property_Type"].unique()),
                                  placeholder="All", key="f_prop")
        f_seas  = st.multiselect("Season", ["Winter","Spring","Summer","Autumn"],
                                  placeholder="All", key="f_sea")
        pmin = int(DF["Price_Per_Night"].min())
        pmax = int(DF["Price_Per_Night"].max())
        f_price = st.slider("Price/night (AED)", pmin, pmax, (pmin, pmax), step=25)
        f_sh    = st.selectbox("Superhost", ["All","Superhost only","Non-superhost"])

        if f_hoods: df = df[df["Neighborhood"].isin(f_hoods)]
        if f_props: df = df[df["Property_Type"].isin(f_props)]
        if f_seas:  df = df[df["Season"].isin(f_seas)]
        df = df[(df["Price_Per_Night"] >= f_price[0]) & (df["Price_Per_Night"] <= f_price[1])]
        if f_sh == "Superhost only":    df = df[df["Superhost_Status"]==1]
        elif f_sh == "Non-superhost":   df = df[df["Superhost_Status"]==0]

        n = len(df)
        if n < 50:
            st.warning("⚠️ Fewer than 50 listings — relax filters.")
            df = DF.copy()
            n = len(df)
        st.markdown(f"<div style='font-size:0.75rem;color:#22C55E;margin-top:4px'>✓ {n:,} listings</div>",
                    unsafe_allow_html=True)

    st.markdown("""
    <hr style='border-color:#2D3A5A;margin:10px 0 6px'/>
    <div style='font-size:0.67rem;color:#475569;text-align:center;line-height:1.7'>
        3,500 Dubai listings · Synthetic<br>
        Seed 42 · Dashboard v3.0
    </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# ── HOME ─────────────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
if pname == "Home":
    hero("🏠 Airbnb Revenue Optimization Platform",
         "Data-driven intelligence for Dubai short-term rental hosts — "
         "pricing, demand forecasting & ML-powered listing optimization.")

    avp = DF["Price_Per_Night"].mean()
    avo = DF["Occupancy_Rate"].mean()
    avr = DF["Review_Score"].mean()
    shp = DF["Superhost_Status"].mean()
    mre = (DF["Price_Per_Night"] * DF["Occupancy_Rate"] * 30).mean()
    bkr = (DF["Booking_Status"]=="Booked").mean()

    c = st.columns(6)
    for col, (v,l) in zip(c, [
        (f"3,500",           "Listings"),
        (f"AED {avp:,.0f}",  "Avg Price/Night"),
        (f"{avo:.0%}",       "Avg Occupancy"),
        (f"{avr:.2f} ★",     "Avg Review"),
        (f"{shp:.0%}",       "Superhost Rate"),
        (f"AED {mre:,.0f}",  "Avg Monthly Rev"),
    ]):
        col.markdown(kcard(v,l), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        slabel("📊 Property Mix")
        pt = DF["Property_Type"].value_counts()
        fig = px.pie(values=pt.values, names=pt.index, hole=0.5,
                     color_discrete_sequence=PALETTE)
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          textfont_size=10)
        _layout(fig, 280, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        slabel("🏙️ Revenue by Neighborhood (top 8)")
        hr = (DF.assign(R=DF["Price_Per_Night"]*DF["Occupancy_Rate"]*30)
                .groupby("Neighborhood")["R"].mean()
                .sort_values().tail(8))
        fig = px.bar(hr, orientation="h",
                     color=hr.values,
                     color_continuous_scale=[[0,"#FEE2E2"],[1,C_RED]])
        _layout(fig, 280, coloraxis_showscale=False,
                xaxis_title="AED/month", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        slabel("📅 Occupancy by Season")
        so = DF.groupby("Season")["Occupancy_Rate"].mean().reset_index()
        so["Pct"] = so["Occupancy_Rate"].map("{:.0%}".format)
        fig = px.bar(so, x="Season", y="Occupancy_Rate",
                     color="Season", color_discrete_sequence=PALETTE,
                     text="Pct")
        fig.update_traces(textposition="outside")
        _layout(fig, 280, yaxis_tickformat=".0%", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    slabel("📌 What this platform does")
    cc1, cc2 = st.columns([1.05, 1])
    with cc1:
        st.markdown("""
**Dubai's Airbnb market** has 25,000+ listings competing for bookings.
Hosts who rely on intuition for pricing lose thousands in revenue every month.
This platform replaces guesswork with data.

**Navigate the sidebar to explore:**
- Exploratory analysis with **dynamic sidebar filters**
- ML models that predict whether **your listing will book**
- Clustering to find **your market segment and peers**
- Association rules revealing the **highest-ROI amenity bundles**
- Regression-based **pricing recommendations**
- Seasonal **demand forecasts**

> **New to the platform?** Head straight to **🧮 Host Simulator** — enter your
> listing details and get a full revenue forecast, pricing verdict, and
> personalised strategy report in seconds.
        """)
    with cc2:
        for icon, name, desc in [
            ("🔍","Exploratory Analysis","Interactive charts with live filters"),
            ("💰","Pricing Analytics","Benchmark vs. market by segment"),
            ("🤖","Classification","Booking probability models"),
            ("🎯","Clustering","4-segment market map"),
            ("🔗","Association Rules","High-lift amenity bundles"),
            ("📈","Regression","Data-backed price prediction"),
            ("📅","Demand Forecasting","12-month seasonal projection"),
            ("🧮","Host Simulator","Full listing simulation & report"),
        ]:
            st.markdown(
                f"<div style='display:flex;gap:10px;align-items:center;padding:7px 12px;"
                f"margin-bottom:5px;background:#FFF8F8;border-radius:7px;"
                f"border-left:3px solid {C_RED};'>"
                f"<span style='font-size:1.05rem'>{icon}</span>"
                f"<div><div style='font-weight:600;font-size:0.82rem;color:{C_NAVY}'>{name}</div>"
                f"<div style='color:{C_GREY};font-size:0.74rem'>{desc}</div></div></div>",
                unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# ── DATASET OVERVIEW ─────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
elif pname == "Dataset Overview":
    hero("📋 Dataset Overview",
         f"Showing {len(df):,} of 3,500 listings · 33 features · Dubai synthetic market data")

    c = st.columns(4)
    for col,(v,l) in zip(c,[
        (f"{len(df):,}", "Listings (filtered)"),
        ("33",           "Variables"),
        ("0",            "Missing Values"),
        ("15",           "Neighborhoods"),
    ]):
        col.markdown(kcard(v,l), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    slabel("🗃️ Data Preview (first 30 rows)")
    st.dataframe(df.head(30).reset_index(drop=True), use_container_width=True, height=280)

    c1,c2 = st.columns(2)
    with c1:
        slabel("📊 Numeric Summary")
        num_c = [c2 for c2 in ["Price_Per_Night","Occupancy_Rate","Review_Score",
                  "Amenities_Count","Bedrooms","Distance_to_City_Center",
                  "Number_of_Reviews","Booking_Lead_Time"] if c2 in df.columns]
        st.dataframe(df[num_c].describe().round(2), use_container_width=True)
    with c2:
        slabel("🔢 Column Info")
        dtdf = pd.DataFrame({"Column":df.columns, "Type":df.dtypes.astype(str).values,
                              "Unique":df.nunique().values, "Non-null":df.notnull().sum().values})
        st.dataframe(dtdf, use_container_width=True, height=300)

    slabel("📊 Key Distributions")
    c1,c2,c3 = st.columns(3)
    with c1:
        fig = px.histogram(df, x="Price_Per_Night", nbins=40,
                           color_discrete_sequence=[C_RED])
        fig.update_traces(marker_line_width=0)
        _layout(fig, 250, xaxis_title="AED/night", title_text="Price Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        pt2 = df["Property_Type"].value_counts().reset_index()
        pt2.columns = ["Type","Count"]
        fig = px.bar(pt2, x="Type", y="Count", color="Count",
                     color_continuous_scale=[[0,"#FEE2E2"],[1,C_RED]])
        _layout(fig, 250, coloraxis_showscale=False, title_text="Property Mix")
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        fig = px.histogram(df, x="Occupancy_Rate", nbins=35,
                           color_discrete_sequence=[C_BLUE])
        fig.update_traces(marker_line_width=0)
        _layout(fig, 250, xaxis_tickformat=".0%", title_text="Occupancy Distribution")
        st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# ── EXPLORATORY ANALYSIS ─────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
elif pname == "Exploratory Analysis":
    hero("🔍 Exploratory Data Analysis",
         f"Analysing {len(df):,} listings · Use sidebar filters to slice dynamically")

    c = st.columns(4)
    bkr2 = (df["Booking_Status"]=="Booked").mean()
    for col,(v,l) in zip(c,[
        (f"{len(df):,}",                      "Listings"),
        (f"AED {df['Price_Per_Night'].mean():,.0f}", "Avg Price"),
        (f"{df['Occupancy_Rate'].mean():.0%}", "Avg Occupancy"),
        (f"{bkr2:.0%}",                        "Booking Rate"),
    ]):
        col.markdown(kcard(v,l), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # row 1
    c1,c2 = st.columns(2)
    with c1:
        slabel("💲 Price vs Occupancy")
        cb1 = st.selectbox("Colour by",["Property_Type","Room_Type","Season","Superhost_Status"], key="eda1")
        samp = df.sample(min(700, len(df)), random_state=1)
        fig = px.scatter(samp, x="Price_Per_Night", y="Occupancy_Rate",
                         color=cb1, opacity=0.55, color_discrete_sequence=PALETTE)
        _layout(fig, 320, xaxis_title="Price/Night (AED)", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        ibox("Inverse relationship: as price rises occupancy falls. Villas hold higher occupancy at premium rates.")

    with c2:
        slabel("⭐ Review Score vs Occupancy")
        cb2 = st.selectbox("Colour by",["Property_Type","Superhost_Status","Season","Neighborhood"], key="eda2")
        samp2 = df.sample(min(700, len(df)), random_state=2)
        fig = px.scatter(samp2, x="Review_Score", y="Occupancy_Rate",
                         color=cb2, opacity=0.55, color_discrete_sequence=PALETTE)
        _layout(fig, 320, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        ibox("Listings scoring 4.5+ achieve 15-20% more occupancy. Superhost badge compounds the effect.")

    # row 2
    c3,c4 = st.columns(2)
    with c3:
        slabel("📍 Distance vs Price")
        dist_m = st.selectbox("Distance metric",["Distance_to_City_Center","Distance_to_Metro"], key="dm")
        fig = px.scatter(df.sample(min(600,len(df)),random_state=3),
                         x=dist_m, y="Price_Per_Night",
                         color="Neighborhood", opacity=0.5, color_discrete_sequence=PALETTE)
        _layout(fig, 300, xaxis_title="km", yaxis_title="AED")
        st.plotly_chart(fig, use_container_width=True)
        ibox("Listings within 2 km of the city centre command a 40-60% price premium.")

    with c4:
        slabel("🏠 Occupancy by Property Type")
        fig = px.box(df, x="Property_Type", y="Occupancy_Rate",
                     color="Property_Type", color_discrete_sequence=PALETTE)
        fig.update_layout(showlegend=False)
        _layout(fig, 300, yaxis_tickformat=".0%", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        ibox("Villas show the widest variance — top-quality premium villas thrive; poor-value ones struggle.")

    # row 3
    c5,c6 = st.columns(2)
    with c5:
        slabel("🛎️ Amenity Count vs Occupancy")
        split_by = st.selectbox("Split by",["None","Season","Property_Type","Room_Type"], key="amsplit")
        if split_by == "None":
            am = df.groupby("Amenities_Count")["Occupancy_Rate"].mean().reset_index()
            fig = px.line(am, x="Amenities_Count", y="Occupancy_Rate",
                          markers=True, color_discrete_sequence=[C_RED])
        else:
            am = df.groupby(["Amenities_Count", split_by])["Occupancy_Rate"].mean().reset_index()
            fig = px.line(am, x="Amenities_Count", y="Occupancy_Rate",
                          color=split_by, color_discrete_sequence=PALETTE)
        _layout(fig, 280, yaxis_tickformat=".0%", xaxis_title="Total Amenities")
        st.plotly_chart(fig, use_container_width=True)
        ibox("Occupancy rises to ~18 amenities then plateaus. First 10 deliver the highest return.")

    with c6:
        slabel("💰 Monthly Revenue by Segment")
        seg = st.selectbox("Group by",["Property_Type","Neighborhood","Season","Traveler_Type","Room_Type"], key="revseg")
        rev = (df.assign(Rev=df["Price_Per_Night"]*df["Occupancy_Rate"]*30)
                 .groupby(seg)["Rev"].mean()
                 .sort_values().reset_index())
        fig = px.bar(rev, x="Rev", y=seg, orientation="h",
                     color="Rev", color_continuous_scale=[[0,"#FEE2E2"],[1,C_RED]])
        _layout(fig, 280, coloraxis_showscale=False, xaxis_title="Avg Monthly Revenue (AED)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        ibox("Penthouses generate 3× Studio revenue. Location + property type together drive revenue.")

    # Correlation heatmap
    slabel("🔥 Correlation Heatmap")
    num_cols = ["Price_Per_Night","Occupancy_Rate","Review_Score","Number_of_Reviews",
                "Amenities_Count","Bedrooms","Distance_to_City_Center","Host_Response_Rate",
                "Superhost_Status","Booking_Lead_Time","Length_of_Stay","Cleaning_Fee"]
    corr = df[num_cols].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[[0,C_BLUE],[0.5,"#FFFFFF"],[1,C_RED]], zmid=0,
        text=corr.values, texttemplate="%{text:.2f}",
        hoverinfo="skip", textfont=dict(size=9),
    ))
    fig.update_layout(**PBASE, height=460)
    st.plotly_chart(fig, use_container_width=True)
    ibox("Strongest links: Bedrooms↔Price (0.55), Review↔Occupancy (0.48). Distance is the top negative price predictor.")

# ═════════════════════════════════════════════════════════════════════════════
# ── PRICING ANALYTICS ────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
elif pname == "Pricing Analytics":
    hero("💰 Pricing Analytics",
         f"Dynamic pricing analysis · {len(df):,} listings · adjust sidebar filters to explore segments")

    c1,c2 = st.columns(2)
    with c1:
        slabel("🏙️ Avg Price by Neighborhood")
        sort_by = st.selectbox("Sort by",["Mean","Median","Std Dev"], key="ps")
        sc = {"Mean":"mean","Median":"median","Std Dev":"std"}[sort_by]
        hp = df.groupby("Neighborhood")["Price_Per_Night"].agg(["mean","median","std"]).round(0).reset_index()
        hp = hp.sort_values(sc, ascending=True)
        fig = px.bar(hp, x=sc, y="Neighborhood", orientation="h",
                     color=sc, color_continuous_scale=[[0,"#FEE2E2"],[1,C_RED]])
        _layout(fig, 380, coloraxis_showscale=False, xaxis_title="AED/night", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        ibox("Palm Jumeirah and Downtown command 2.5× the median price of outer districts.")

    with c2:
        slabel("📅 Seasonal Pricing")
        bd = st.selectbox("Break down by",["Property_Type","Room_Type","Neighborhood","Traveler_Type"], key="sbd")
        sp = df.groupby(["Season", bd])["Price_Per_Night"].mean().reset_index()
        fig = px.bar(sp, x="Season", y="Price_Per_Night", color=bd,
                     barmode="group", color_discrete_sequence=PALETTE)
        _layout(fig, 380, yaxis_title="Avg Price (AED/night)")
        st.plotly_chart(fig, use_container_width=True)
        ibox("Winter (Oct–Mar) is Dubai's peak — all types see 15-25% higher prices vs. summer.")

    c3,c4 = st.columns(2)
    with c3:
        slabel("📆 Price by Day of Week")
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dop = df.groupby("Day_of_Week")["Price_Per_Night"].mean().reindex(day_order).reset_index()
        fig = px.bar(dop, x="Day_of_Week", y="Price_Per_Night",
                     color="Price_Per_Night", color_continuous_scale=[[0,"#FEE2E2"],[1,C_RED]])
        _layout(fig, 280, coloraxis_showscale=False, yaxis_title="Avg Price (AED)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        ibox("Thu–Sat carries a 10-12% weekend premium. Dynamic day-of-week pricing captures this.")

    with c4:
        slabel("🛏️ Bedrooms vs Price")
        ptf = st.selectbox("Property type",["All"]+sorted(df["Property_Type"].unique().tolist()), key="bpt")
        dfb = df if ptf=="All" else df[df["Property_Type"]==ptf]
        bp = dfb.groupby("Bedrooms")["Price_Per_Night"].agg(["mean","median"]).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bp["Bedrooms"], y=bp["mean"], name="Mean",
                             marker_color=C_RED, opacity=0.8))
        fig.add_trace(go.Scatter(x=bp["Bedrooms"], y=bp["median"], name="Median",
                                 mode="lines+markers", line=dict(color=C_NAVY, width=2)))
        _layout(fig, 280, xaxis_title="Bedrooms", yaxis_title="Price (AED)")
        st.plotly_chart(fig, use_container_width=True)
        ibox("Each additional bedroom adds ~AED 50-80/night on average.")

    slabel("🗺️ Price Heatmap: Neighborhood × Season")
    pivot = df.pivot_table(values="Price_Per_Night", index="Neighborhood",
                           columns="Season", aggfunc="mean").round(0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
        colorscale=[[0,"#FFF5F5"],[0.5,"#FF8C8F"],[1,C_RED]],
        text=pivot.fillna(0).values.astype(int),
        texttemplate="AED %{text}", textfont=dict(size=9), hoverinfo="skip",
    ))
    fig.update_layout(**PBASE, height=480)
    st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# ── CLASSIFICATION MODELS ────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
elif pname == "Classification Models":
    hero("🤖 Classification Models",
         "Predicting Booking_Status — which listing features drive bookings?")

    mets = M["clf_metrics"]
    best = max(mets, key=lambda m: mets[m]["F1 Score"])

    cols = st.columns(len(mets))
    for col, (name, met) in zip(cols, mets.items()):
        b = name == best
        col.markdown(
            f"<div class='kcard' style='border-top-color:{'#22C55E' if b else C_RED}'>"
            f"<div style='font-size:0.8rem;font-weight:600;color:{C_NAVY};margin-bottom:5px'>{name}</div>"
            f"<div class='kcard-val' style='font-size:1.5rem'>{met['F1 Score']}%</div>"
            f"<div class='kcard-lbl'>F1 Score</div>"
            f"<div style='font-size:0.72rem;color:{C_GREY};margin-top:5px'>"
            f"Acc {met['Accuracy']}% · Prec {met['Precision']}% · Rec {met['Recall']}%</div>"
            f"{'<div style=\"margin-top:5px;font-size:0.7rem;color:#22C55E;font-weight:700\">⭐ BEST</div>' if b else ''}"
            f"</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        slabel("📊 Model Comparison")
        res_df = pd.DataFrame(mets).T.reset_index().rename(columns={"index":"Model"})
        ml = res_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig = px.bar(ml, x="Model", y="Score", color="Metric", barmode="group",
                     color_discrete_sequence=[C_RED, C_AMBER, C_GREEN, C_BLUE])
        _layout(fig, 340, yaxis_range=[0,105], yaxis_title="Score (%)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        slabel("🎯 Feature Importance (Random Forest)")
        fi = M["feat_imp"]
        fig = px.bar(x=fi.values, y=fi.index, orientation="h",
                     color=fi.values, color_continuous_scale=[[0,"#FEE2E2"],[1,C_RED]])
        _layout(fig, 340, coloraxis_showscale=False, xaxis_title="Importance", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    ibox(f"**{best}** achieves the best balance. Top predictors: "
         f"{', '.join(fi.sort_values(ascending=False).head(3).index.tolist())}.")

    slabel("🔢 Confusion Matrices")
    cms = M["cms"]
    cols = st.columns(len(cms))
    for col, (name, cm) in zip(cols, cms.items()):
        fig = go.Figure(go.Heatmap(
            z=cm, x=["Not Booked","Booked"], y=["Not Booked","Booked"],
            colorscale=[[0,"#FFF5F5"],[1,C_RED]],
            text=cm, texttemplate="%{text}", hoverinfo="skip", showscale=False,
        ))
        fig.update_layout(**PBASE, height=230,
                          title=dict(text=name, font=dict(size=11)))
        col.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# ── CLUSTERING ───────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
elif pname == "Clustering Analysis":
    hero("🎯 Clustering Analysis",
         "K-Means (K=4) segmentation — discover the 4 listing archetypes in Dubai's market")

    df_c = DF.copy()
    df_c["Cluster"] = M["km_labels"]
    SEG_NAMES = {0:"💼 Business Apt",1:"🏖️ Luxury Retreat",
                 2:"👨‍👩‍👧 Family Home",3:"💸 Budget Stay"}
    df_c["Segment"] = df_c["Cluster"].map(SEG_NAMES)

    c1,c2 = st.columns(2)
    with c1:
        slabel("📍 Cluster Scatter")
        xax = st.selectbox("X-axis",["Price_Per_Night","Distance_to_City_Center","Amenities_Count"], key="cx")
        yax = st.selectbox("Y-axis",["Occupancy_Rate","Review_Score","Price_Per_Night"], key="cy")
        samp = df_c.sample(min(900,len(df_c)), random_state=1)
        fig = px.scatter(samp, x=xax, y=yax, color="Segment",
                         color_discrete_sequence=PALETTE, opacity=0.6)
        _layout(fig, 350, yaxis_tickformat=".0%" if "Rate" in yax else "")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        slabel("📡 Segment Radar Profiles")
        feats = M["FEAT_CLU"]
        prof  = df_c.groupby("Segment")[feats].mean()
        norm  = (prof - prof.min()) / (prof.max() - prof.min() + 1e-9)
        fig = go.Figure()
        for i, (seg, row) in enumerate(norm.iterrows()):
            v = row.tolist() + [row.tolist()[0]]
            c_list = feats + [feats[0]]
            fig.add_trace(go.Scatterpolar(r=v, theta=c_list, fill="toself",
                                           name=seg, line_color=PALETTE[i], opacity=0.65))
        fig.update_layout(**PBASE, height=350,
                          polar=dict(bgcolor="#F8FAFC",
                                     radialaxis=dict(visible=True, color="#CBD5E1")))
        st.plotly_chart(fig, use_container_width=True)

    slabel("🏷️ Segment Profiles")
    full_prof = df_c.groupby("Segment")[
        ["Price_Per_Night","Occupancy_Rate","Review_Score",
         "Bedrooms","Amenities_Count","Distance_to_City_Center"]
    ].mean().round(2)

    DESCS = {
        "💼 Business Apt":   "Mid-range, central apartments. WiFi + self check-in critical. 2-5 night stays, high weekday demand.",
        "🏖️ Luxury Retreat": "Premium — Palm Jumeirah, Downtown. Pool + parking essential. Weekend leisure travellers.",
        "👨‍👩‍👧 Family Home":   "3+ bedroom villas/townhouses. Kitchen + laundry + parking. 7+ night stays, value-driven pricing.",
        "💸 Budget Stay":    "Studios in outer districts. Minimal amenities, short stays, price-sensitive solo travellers.",
    }

    gc = st.columns(2)
    for i,(seg,row) in enumerate(full_prof.iterrows()):
        col = gc[i%2]
        color = PALETTE[i]
        col.markdown(
            f"<div style='border:1px solid #E2E8F0;border-left:4px solid {color};"
            f"border-radius:0 10px 10px 0;padding:15px 17px;margin-bottom:10px;"
            f"background:#FAFAFA;'>"
            f"<div style='font-weight:700;font-size:0.92rem;color:{C_NAVY};margin-bottom:3px'>{seg}</div>"
            f"<div style='font-size:0.78rem;color:{C_GREY};margin-bottom:10px'>{DESCS.get(seg,'')}</div>"
            f"<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:6px'>"
            f"<div style='background:#fff;border-radius:6px;padding:7px;text-align:center;border:1px solid #E2E8F0'>"
            f"<div style='color:{color};font-weight:700'>AED {row['Price_Per_Night']:.0f}</div>"
            f"<div style='font-size:0.68rem;color:#94A3B8'>Price/night</div></div>"
            f"<div style='background:#fff;border-radius:6px;padding:7px;text-align:center;border:1px solid #E2E8F0'>"
            f"<div style='color:{color};font-weight:700'>{row['Occupancy_Rate']:.0%}</div>"
            f"<div style='font-size:0.68rem;color:#94A3B8'>Occupancy</div></div>"
            f"<div style='background:#fff;border-radius:6px;padding:7px;text-align:center;border:1px solid #E2E8F0'>"
            f"<div style='color:{color};font-weight:700'>{row['Review_Score']:.1f}★</div>"
            f"<div style='font-size:0.68rem;color:#94A3B8'>Review</div></div>"
            f"</div></div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# ── ASSOCIATION RULES ────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
elif pname == "Association Rules":
    hero("🔗 Association Rule Mining",
         "Apriori algorithm — amenity combinations that drive bookings, revenue & longer stays")

    with st.spinner("Mining rules…"):
        rules = run_apriori(DF)

    c = st.columns(3)
    for col,(v,l) in zip(c,[
        (len(rules),                          "Rules Found"),
        (f"{rules['lift'].max():.2f}",         "Max Lift"),
        (f"{rules['confidence'].max():.0%}",   "Max Confidence"),
    ]):
        col.markdown(kcard(v,l), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns([1.2,1])
    with c1:
        slabel("📋 Rules Table")
        min_lift = st.slider("Min lift threshold", 1.0, float(rules["lift"].max()), 1.05, 0.05)
        show = rules[rules["lift"]>=min_lift][["antecedents","consequents",
                                                "support","confidence","lift"]].copy()
        show.columns = ["IF (Antecedent)","THEN (Consequent)","Support","Confidence","Lift"]
        show["Support"]    = show["Support"].map("{:.1%}".format)
        show["Confidence"] = show["Confidence"].map("{:.1%}".format)
        show["Lift"]       = show["Lift"].map("{:.3f}".format)
        st.dataframe(show.head(20), use_container_width=True, height=360)

    with c2:
        slabel("📊 Support vs Confidence (bubble = lift)")
        filtered = rules[rules["lift"]>=min_lift]
        fig = px.scatter(filtered, x="support", y="confidence", size="lift",
                         color="lift", color_continuous_scale=[[0,"#FEE2E2"],[1,C_RED]],
                         hover_data=["antecedents","consequents"])
        _layout(fig, 300, xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                xaxis_title="Support", yaxis_title="Confidence")
        st.plotly_chart(fig, use_container_width=True)

    slabel("🏆 Top Rules by Lift")
    top10 = rules.head(10).copy()
    top10["Rule"] = top10["antecedents"] + "  →  " + top10["consequents"]
    fig = px.bar(top10, x="lift", y="Rule", orientation="h",
                 color="lift", color_continuous_scale=[[0,"#FEE2E2"],[1,C_RED]])
    _layout(fig, 360, coloraxis_showscale=False, yaxis_title="", xaxis_title="Lift")
    st.plotly_chart(fig, use_container_width=True)
    ibox("WiFi + Self_Checkin → High_Occupancy has highest lift. Pool + Parking → premium pricing. Kitchen + Air_Con → longer stays.")

    slabel("💡 Host Action Recommendations")
    rec_cols = st.columns(4)
    for col,(icon,title,desc,color) in zip(rec_cols,[
        ("🛜","WiFi + Self Check-in","Highest occupancy lift — must-have combination.", C_GREEN),
        ("🏊","Pool + Parking",       "Strongly predicts premium pricing in top areas.", C_BLUE),
        ("🍳","Kitchen + Air Con",    "Extends average stay — critical for family stays.", C_AMBER),
        ("⭐","Superhost + WiFi",     "Most bookings — actively pursue Superhost status.", C_RED),
    ]):
        col.markdown(
            f"<div style='background:#fff;border:1px solid #E2E8F0;border-top:3px solid {color};"
            f"border-radius:6px;padding:13px;text-align:center;'>"
            f"<div style='font-size:1.5rem'>{icon}</div>"
            f"<div style='font-weight:700;color:{C_NAVY};font-size:0.83rem;margin:5px 0 3px'>{title}</div>"
            f"<div style='color:{C_GREY};font-size:0.74rem'>{desc}</div></div>",
            unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# ── REGRESSION ───────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
elif pname == "Regression Modeling":
    hero("📈 Regression Modeling",
         "Ridge regression predicts optimal nightly price — understand which features matter most")

    c = st.columns(3)
    for col,(v,l) in zip(c,[
        (f"{M['r2']:.3f}",       "R² Score (Ridge)"),
        (f"AED {M['mae']:.0f}",  "Mean Abs. Error"),
        ("Ridge",                "Best Model"),
    ]):
        col.markdown(kcard(v,l), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        slabel("📊 Feature Coefficients (Ridge)")
        coef = pd.Series(M["ridge"].coef_, index=M["FEAT_REG"]).sort_values()
        colors_c = [C_RED if v>0 else C_BLUE for v in coef.values]
        fig = go.Figure(go.Bar(x=coef.values, y=coef.index, orientation="h",
                               marker_color=colors_c))
        fig.add_vline(x=0, line_color="#CBD5E1", line_width=1)
        _layout(fig, 380, xaxis_title="Coefficient (standardised)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        ibox("Red = positive price driver. Blue = reduces price. Bedrooms and Pool are the strongest positive drivers; Distance the strongest negative.")

    with c2:
        slabel("🎯 Actual vs Predicted Price")
        rng  = np.random.default_rng(42)
        idx  = rng.choice(len(DF), min(400,len(DF)), replace=False)
        Xs   = M["sc_reg"].transform(DF[M["FEAT_REG"]].values)
        yp   = M["ridge"].predict(Xs)
        yt   = DF["Price_Per_Night"].values
        fig  = px.scatter(x=yt[idx], y=yp[idx], opacity=0.4,
                          color_discrete_sequence=[C_RED])
        mn,mx = float(yt.min()), float(yt.max())
        fig.add_shape(type="line", x0=mn, x1=mx, y0=mn, y1=mx,
                      line=dict(color=C_NAVY, width=1.5, dash="dash"))
        _layout(fig, 380, xaxis_title="Actual (AED)", yaxis_title="Predicted (AED)")
        st.plotly_chart(fig, use_container_width=True)
        ibox("Points along the diagonal = accurate prediction. Variance is highest in the luxury tier (>AED 1,000).")

    slabel("🔍 Price Driver Analysis — Interactive")
    col_sel = st.selectbox("Feature to vary", M["FEAT_REG"], key="regfeat")
    base_v  = {f: float(DF[f].mean()) for f in M["FEAT_REG"]}
    rng_v   = np.linspace(DF[col_sel].min(), DF[col_sel].max(), 40)
    preds   = []
    for v in rng_v:
        r = base_v.copy(); r[col_sel] = v
        Xs_r = M["sc_reg"].transform(pd.DataFrame([r])[M["FEAT_REG"]].values)
        preds.append(float(M["ridge"].predict(Xs_r)[0]))
    fig = px.line(x=rng_v, y=preds, color_discrete_sequence=[C_RED], markers=False)
    fig.add_hline(y=float(DF["Price_Per_Night"].mean()), line_dash="dot",
                  line_color="#94A3B8", annotation_text="Market avg", annotation_font_size=10)
    _layout(fig, 280, xaxis_title=col_sel, yaxis_title="Predicted Price (AED)")
    st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# ── DEMAND FORECASTING ───────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
elif pname == "Demand Forecasting":
    hero("📅 Demand Forecasting",
         "Monthly booking demand, seasonal patterns & 12-month projections")

    # Controls
    cc1,cc2,cc3 = st.columns([2,1,1])
    with cc2:
        pt_f = st.selectbox("Property Type",["All"]+sorted(DF["Property_Type"].unique().tolist()), key="dpt")
    with cc3:
        nh_f = st.selectbox("Neighborhood",["All"]+sorted(DF["Neighborhood"].unique().tolist()), key="dnh")
    dfm = DF.copy()
    if pt_f != "All": dfm = dfm[dfm["Property_Type"]==pt_f]
    if nh_f != "All": dfm = dfm[dfm["Neighborhood"]==nh_f]

    bm = dfm[dfm["Booking_Status"]=="Booked"].groupby("Month").size().reset_index(name="Bookings")
    bm["Month_Name"] = bm["Month"].map(MONTH_NAMES)

    with cc1:
        slabel(f"📊 Monthly Demand + Forecast  ({len(dfm):,} listings)")

    if len(bm) >= 4:
        Xm   = bm["Month"].values.reshape(-1,1)
        poly = PolynomialFeatures(degree=3)
        Xp   = poly.fit_transform(Xm)
        rd   = Ridge(alpha=1).fit(Xp, bm["Bookings"].values)
        fut  = list(range(1,19))
        yf   = np.clip(rd.predict(poly.transform(np.array(fut).reshape(-1,1))), 0, None)
        mn_lab = [MONTH_NAMES.get(m%12 or 12,"") + ("'" if m>12 else "") for m in fut]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=bm["Month_Name"], y=bm["Bookings"],
                             name="Actual", marker_color=C_RED, opacity=0.85))
        fig.add_trace(go.Scatter(x=mn_lab, y=yf, name="Forecast",
                                 mode="lines+markers",
                                 line=dict(color=C_NAVY, width=2, dash="dot"),
                                 marker=dict(size=5)))
        _layout(fig, 300, yaxis_title="Bookings")
        st.plotly_chart(fig, use_container_width=True)
        ibox("Dubai peaks Nov–Feb (winter tourism). Summer sees 30-40% fewer bookings — lower minimums and offer long-stay discounts.")
    else:
        st.info("Insufficient data for forecast with current filters.")

    c1,c2 = st.columns(2)
    with c1:
        slabel("📆 Bookings by Day of Week")
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dd = (dfm[dfm["Booking_Status"]=="Booked"]
              .groupby("Day_of_Week").size().reindex(day_order).reset_index(name="Bookings"))
        fig = px.bar(dd, x="Day_of_Week", y="Bookings",
                     color="Bookings", color_continuous_scale=[[0,"#FEE2E2"],[1,C_RED]])
        _layout(fig, 280, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        ibox("Thu–Sat accounts for ~45% of bookings. Raise minimums on peak days.")

    with c2:
        slabel("👥 Demand by Traveller Type & Season")
        sd = (dfm[dfm["Booking_Status"]=="Booked"]
              .groupby(["Season","Traveler_Type"]).size().reset_index(name="Bookings"))
        fig = px.bar(sd, x="Season", y="Bookings", color="Traveler_Type",
                     barmode="stack", color_discrete_sequence=PALETTE)
        _layout(fig, 280)
        st.plotly_chart(fig, use_container_width=True)
        ibox("Couples dominate winter. Families peak in spring. Business travellers are consistent year-round.")

    slabel("🗓️ Demand Heatmap: Month × Day of Week")
    bh = dfm[dfm["Booking_Status"]=="Booked"]
    if len(bh) > 20:
        ph = bh.pivot_table(values="Listing_ID", index="Month",
                            columns="Day_of_Week", aggfunc="count")
        ph = ph.reindex(columns=[d for d in day_order if d in ph.columns])
        ph.index = [MONTH_NAMES[m] for m in ph.index]
        fig = go.Figure(go.Heatmap(
            z=ph.values, x=list(ph.columns), y=list(ph.index),
            colorscale=[[0,"#FFF5F5"],[0.5,"#FF8C8F"],[1,C_RED]],
            hoverinfo="skip",
        ))
        fig.update_layout(**PBASE, height=380)
        st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# ── REVENUE OPTIMIZATION ─────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
elif pname == "Revenue Optimization":
    hero("🚀 Revenue Optimization",
         f"Analysing {len(df):,} listings · adjust sidebar filters to focus on your segment")

    df2 = df.copy()
    df2["Monthly_Revenue"] = df2["Price_Per_Night"] * df2["Occupancy_Rate"] * 30
    top_h = df2.groupby("Neighborhood")["Monthly_Revenue"].mean().idxmax()
    top_p = df2.groupby("Property_Type")["Monthly_Revenue"].mean().idxmax()

    c = st.columns(4)
    for col,(v,l) in zip(c,[
        (f"AED {df2['Monthly_Revenue'].mean():,.0f}", "Avg Monthly Revenue"),
        (top_h,                                        "Top Neighborhood"),
        (top_p,                                        "Top Property Type"),
        (f"AED {df2['Monthly_Revenue'].max():,.0f}",   "Peak Monthly Revenue"),
    ]):
        col.markdown(kcard(v,l), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        slabel("📈 Price vs Revenue Curve")
        bins = pd.cut(df2["Price_Per_Night"], bins=25)
        rc = df2.groupby(bins, observed=True)["Monthly_Revenue"].mean().reset_index()
        rc["Price_Mid"] = rc["Price_Per_Night"].apply(lambda x: x.mid)
        rc = rc.dropna(subset=["Price_Mid"])
        opt_x = float(rc.loc[rc["Monthly_Revenue"].idxmax(), "Price_Mid"])
        fig = px.line(rc, x="Price_Mid", y="Monthly_Revenue",
                      color_discrete_sequence=[C_RED], markers=True)
        fig.add_vline(x=opt_x, line_dash="dash", line_color=C_GREEN,
                      annotation_text=f"Optimal ≈ AED {opt_x:.0f}",
                      annotation_font_color=C_GREEN, annotation_font_size=11)
        _layout(fig, 320, xaxis_title="Price/Night (AED)", yaxis_title="Avg Monthly Revenue (AED)")
        st.plotly_chart(fig, use_container_width=True)
        ibox(f"Revenue-maximising price ≈ AED {opt_x:.0f}/night. Above this, occupancy falls faster than rate rises.")

    with c2:
        slabel("🏙️ Revenue by Neighborhood")
        hr = df2.groupby("Neighborhood")["Monthly_Revenue"].mean().sort_values().reset_index()
        fig = px.bar(hr, x="Monthly_Revenue", y="Neighborhood", orientation="h",
                     color="Monthly_Revenue", color_continuous_scale=[[0,"#FEE2E2"],[1,C_RED]])
        _layout(fig, 320, coloraxis_showscale=False, xaxis_title="Avg Monthly Revenue (AED)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        ibox("Palm Jumeirah and Downtown generate 2.5-3× the revenue of outer districts.")

    slabel("🗺️ Revenue Heatmap: Neighborhood × Property Type")
    rp = df2.pivot_table(values="Monthly_Revenue", index="Neighborhood",
                         columns="Property_Type", aggfunc="mean").round(0)
    fig = go.Figure(go.Heatmap(
        z=rp.values, x=list(rp.columns), y=list(rp.index),
        colorscale=[[0,"#FFF5F5"],[0.5,"#FF8C8F"],[1,C_RED]],
        text=rp.fillna(0).values.astype(int),
        texttemplate="AED %{text}", textfont=dict(size=9), hoverinfo="skip",
    ))
    fig.update_layout(**PBASE, height=480)
    st.plotly_chart(fig, use_container_width=True)

    slabel("💡 Amenity Revenue Impact")
    am_d = {}
    for am in ["Wifi","Kitchen","Pool","Parking","Self_Checkin","Air_Conditioning"]:
        am_d[am] = round(df2[df2[am]==1]["Monthly_Revenue"].mean() -
                         df2[df2[am]==0]["Monthly_Revenue"].mean(), 0)
    am_plot = pd.DataFrame.from_dict(am_d, orient="index", columns=["Delta"]).sort_values("Delta")
    fig = px.bar(am_plot, x="Delta", y=am_plot.index, orientation="h",
                 color="Delta", color_continuous_scale=[[0,"#FEE2E2"],[0.5,"#FFFFFF"],[1,C_GREEN]])
    _layout(fig, 260, coloraxis_showscale=False, xaxis_title="Revenue Impact (AED/month)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# ── HOST SIMULATOR ───────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
elif pname == "Host Simulator":
    st.markdown(
        "<div class='hero'><div class='hero-title'>🧮 Host Simulator</div>"
        "<div class='hero-sub'>Enter your listing details and get an instant AI-powered revenue forecast, "
        "pricing recommendation, market segment, and personalised strategy report.</div></div>",
        unsafe_allow_html=True)

    st.markdown("### 📝 Enter Your Listing Details")

    with st.form("sim_form", clear_on_submit=False):
        # ── Property ────────────────────────────────────────────────────────
        st.markdown("#### 🏠 Property")
        r1 = st.columns(4)
        neighborhood  = r1[0].selectbox("Neighborhood", sorted(DF["Neighborhood"].unique()))
        prop_type     = r1[1].selectbox("Property Type", sorted(DF["Property_Type"].unique()))
        room_type     = r1[2].selectbox("Room Type",["Entire home/apt","Private room","Shared room"])
        bedrooms      = r1[3].number_input("Bedrooms", 0, 8, 2)

        r2 = st.columns(4)
        bathrooms     = r2[0].number_input("Bathrooms", 1, 8, 1)
        accommodates  = r2[1].number_input("Max Guests", 1, 16, 4)
        dist_center   = r2[2].slider("Distance to City Centre (km)", 0.5, 25.0, 5.0, 0.5)
        dist_metro    = r2[3].slider("Distance to Metro (km)", 0.2, 15.0, 2.0, 0.2)

        # ── Pricing ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 💰 Pricing")
        r3 = st.columns(3)
        price        = r3[0].number_input("Your Price/Night (AED)", 50, 5000, 350, step=25)
        cleaning_fee = r3[1].number_input("Cleaning Fee (AED)", 0, 1000, 75, step=25)
        min_nights   = r3[2].selectbox("Minimum Nights", [1,2,3,5,7], index=0)

        # ── Amenities ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🛎️ Amenities")
        ac = st.columns(6)
        wifi         = int(ac[0].checkbox("WiFi",           value=True))
        kitchen      = int(ac[1].checkbox("Kitchen",        value=True))
        air_con      = int(ac[2].checkbox("Air Con",        value=True))
        parking      = int(ac[3].checkbox("Parking",        value=False))
        self_checkin = int(ac[4].checkbox("Self Check-in",  value=False))
        pool         = int(ac[5].checkbox("Pool",           value=False))
        amenities_count = st.slider(
            "Total amenity count (including all items)",
            min_value=3, max_value=30,
            value=min(30, 3 + wifi + kitchen + air_con + parking + self_checkin + pool + 4))

        # ── Host & reviews ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### ⭐ Host & Reviews")
        hr = st.columns(3)
        review_score   = hr[0].slider("Your Review Score", 1.0, 5.0, 4.3, 0.1)
        num_reviews    = hr[1].number_input("Number of Reviews", 0, 500, 25)
        host_resp_rate = hr[2].slider("Host Response Rate (%)", 0, 100, 92)
        superhost      = int(st.checkbox("I hold Superhost status", value=False))

        # ── Booking context ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📅 Booking Context")
        bc = st.columns(3)
        season      = bc[0].selectbox("Target Season",["Winter","Spring","Summer","Autumn"])
        lead_time   = bc[1].slider("Typical Lead Time (days)", 0, 180, 14)
        length_stay = bc[2].slider("Typical Stay Length (nights)", 1, 30, 3)

        submitted = st.form_submit_button("🚀 Run My Simulation",
                                           use_container_width=True, type="primary")

    # ─────────────────────────────────────────────────────────────────────────
    # SIMULATION ENGINE
    # ─────────────────────────────────────────────────────────────────────────
    if submitted:
        st.markdown("---")
        st.markdown("## 📊 Simulation Results")

        # ── 1. Estimate occupancy from comparable listings ────────────────────
        comp = DF[(DF["Neighborhood"]==neighborhood) & (DF["Property_Type"]==prop_type)]
        base_occ = comp["Occupancy_Rate"].mean() if len(comp) >= 5 else DF["Occupancy_Rate"].mean()

        rev_adj   = (review_score - DF["Review_Score"].mean()) * 0.08
        sh_adj    = 0.06 if superhost else 0.0
        price_adj = -(max(0.0, float(price) - 500.0) / 2000.0) * 0.15
        am_adj    = (amenities_count - DF["Amenities_Count"].mean()) / DF["Amenities_Count"].std() * 0.02

        est_occ = float(np.clip(base_occ + rev_adj + sh_adj + price_adj + am_adj, 0.05, 0.97))

        # ── 2. Booking probability (Random Forest) ─────────────────────────────
        clf_row = pd.DataFrame([{
            "Price_Per_Night":         float(price),
            "Review_Score":            review_score,
            "Occupancy_Rate":          est_occ,
            "Amenities_Count":         amenities_count,
            "Distance_to_City_Center": dist_center,
            "Superhost_Status":        superhost,
            "Host_Response_Rate":      float(host_resp_rate),
            "Number_of_Reviews":       float(num_reviews),
            "Bedrooms":                float(bedrooms),
            "Wifi":                    float(wifi),
            "Pool":                    float(pool),
            "Parking":                 float(parking),
            "Booking_Lead_Time":       float(lead_time),
            "Length_of_Stay":          float(length_stay),
        }])
        rf_mdl      = M["clf_models"]["Random Forest"]
        booking_prob = float(rf_mdl.predict_proba(clf_row[M["FEAT_CLF"]].values)[0][1])

        # ── 3. Market price prediction (Ridge) ────────────────────────────────
        reg_row = pd.DataFrame([{
            "Bedrooms":                float(bedrooms),
            "Bathrooms":               float(bathrooms),
            "Accommodates":            float(accommodates),
            "Amenities_Count":         float(amenities_count),
            "Review_Score":            review_score,
            "Superhost_Status":        float(superhost),
            "Distance_to_City_Center": dist_center,
            "Wifi":                    float(wifi),
            "Kitchen":                 float(kitchen),
            "Pool":                    float(pool),
            "Parking":                 float(parking),
            "Air_Conditioning":        float(air_con),
            "Host_Response_Rate":      float(host_resp_rate),
            "Occupancy_Rate":          est_occ,
        }])
        reg_input_s  = M["sc_reg"].transform(reg_row[M["FEAT_REG"]].values)
        market_price = float(max(50.0, M["ridge"].predict(reg_input_s)[0]))

        # ── 4. Revenue calculations ────────────────────────────────────────────
        monthly_rev  = float(price) * est_occ * 30.0
        market_rev   = market_price * est_occ * 30.0
        price_gap    = float(price) - market_price
        if   price_gap >  50: price_status = "overpriced"
        elif price_gap < -50: price_status = "underpriced"
        else:                 price_status = "well_priced"

        season_mult = {"Winter":1.20,"Spring":1.05,"Summer":0.85,"Autumn":1.00}[season]
        seasonal_rev = monthly_rev * season_mult

        # ── 5. Market segment ─────────────────────────────────────────────────
        clu_row     = M["sc_clu"].transform([[float(price), float(amenities_count),
                                               review_score, float(bedrooms),
                                               dist_center,  est_occ]])
        seg_id      = int(M["km"].predict(clu_row)[0])
        SEG_NAMES   = {0:"💼 Business Apartment",1:"🏖️ Luxury Retreat",
                       2:"👨‍👩‍👧 Family Home",      3:"💸 Budget Stay"}
        my_segment  = SEG_NAMES[seg_id]

        # ── 6. Seasonal 12-month projection ───────────────────────────────────
        S_MAP  = {1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                  6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",11:"Autumn",12:"Winter"}
        SMULT  = {"Winter":1.20,"Spring":1.05,"Summer":0.85,"Autumn":1.00}
        monthly_proj = [{"Month":MONTH_NAMES[m],"Season":S_MAP[m],
                          "Revenue":round(monthly_rev * SMULT[S_MAP[m]], 0)}
                         for m in range(1,13)]
        proj_df   = pd.DataFrame(monthly_proj)
        annual_rev = proj_df["Revenue"].sum()

        # ── 7. Pool delta (local) ──────────────────────────────────────────────
        df_tmp = DF.copy()
        df_tmp["_R"] = df_tmp["Price_Per_Night"] * df_tmp["Occupancy_Rate"] * 30
        pool_delta   = int(df_tmp[df_tmp["Pool"]==1]["_R"].mean() -
                           df_tmp[df_tmp["Pool"]==0]["_R"].mean())

        # ─────────────────────────────────────────────────────────────────────
        # DISPLAY RESULTS
        # ─────────────────────────────────────────────────────────────────────

        # Top KPI strip
        rc = st.columns(5)
        kpi_items = [
            (f"{booking_prob:.0%}",     "Booking Probability",  booking_prob > 0.5),
            (f"AED {monthly_rev:,.0f}", "Est. Monthly Revenue", monthly_rev > 5000),
            (f"{est_occ:.0%}",          "Est. Occupancy",       est_occ > 0.6),
            (f"AED {market_price:,.0f}","Market Price Suggest.", True),
            (f"AED {annual_rev:,.0f}",  "Est. Annual Revenue",  True),
        ]
        for col,(v,l,good) in zip(rc, kpi_items):
            color = C_GREEN if good else C_RED
            col.markdown(
                f"<div class='kcard' style='border-top-color:{color}'>"
                f"<div class='kcard-val' style='color:{color};font-size:1.45rem'>{v}</div>"
                f"<div class='kcard-lbl'>{l}</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge + revenue comparison
        g1, g2 = st.columns(2)
        with g1:
            slabel("🎯 Booking Probability Gauge")
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=booking_prob * 100,
                title={"text":"Booking Probability (%)","font":{"size":13}},
                number={"suffix":"%","valueformat":".1f",
                        "font":{"size":38,"color": C_GREEN if booking_prob>=0.6 else C_AMBER if booking_prob>=0.4 else C_RED}},
                gauge={
                    "axis":  {"range":[0,100],"tickwidth":1,"tickcolor":"#CBD5E1"},
                    "bar":   {"color": C_GREEN if booking_prob>=0.6 else C_AMBER if booking_prob>=0.4 else C_RED},
                    "steps": [{"range":[0,40],"color":"#FEE2E2"},
                               {"range":[40,65],"color":"#FEF3C7"},
                               {"range":[65,100],"color":"#D1FAE5"}],
                    "threshold":{"line":{"color":C_NAVY,"width":3},"value":60},
                }
            ))
            fig_g.update_layout(**PBASE, height=280)
            st.plotly_chart(fig_g, use_container_width=True)

        with g2:
            slabel("📊 Revenue Scenario Comparison")
            fig_rv = go.Figure(go.Bar(
                x=["Your Price","Market Price",f"Seasonal ({season})"],
                y=[monthly_rev, market_rev, seasonal_rev],
                marker_color=[C_RED, C_BLUE, C_AMBER],
                text=[f"AED {v:,.0f}" for v in [monthly_rev, market_rev, seasonal_rev]],
                textposition="outside",
            ))
            _layout(fig_rv, 280, yaxis_title="Monthly Revenue (AED)")
            st.plotly_chart(fig_rv, use_container_width=True)

        # Pricing verdict
        slabel("💰 Pricing Verdict")
        if price_status == "overpriced":
            pct = (float(price) - market_price) / market_price * 100
            ebox(f"Your price (AED {price}) is {pct:.0f}% above market (AED {market_price:.0f}). "
                 f"Reducing to AED {market_price:.0f} could recover lost occupancy and net more monthly revenue. "
                 f"Try dropping by AED 25 increments and monitor conversion over 2 weeks.")
        elif price_status == "underpriced":
            pct = (market_price - float(price)) / market_price * 100
            gain = (market_price - float(price)) * est_occ * 30
            st.markdown(
                f"<div class='ibox'>✅ You are {pct:.0f}% below market (AED {market_price:.0f}). "
                f"Raising to market rate could add ~AED {gain:,.0f}/month. "
                f"Your reviews ({review_score}★) justify the increase — raise by AED 25 every 2 weeks.</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='background:#EFF6FF;border-left:3px solid {C_BLUE};border-radius:0 6px 6px 0;"
                f"padding:9px 13px;font-size:0.82rem;color:#1E40AF;'>✅ Your price (AED {price}) is well-aligned "
                f"with market (AED {market_price:.0f}). Focus on review quality and Superhost status to grow occupancy.</div>",
                unsafe_allow_html=True)

        # Market segment
        slabel(f"🎯 Your Market Segment: {my_segment}")
        df_c = DF.copy()
        df_c["Cluster"]  = M["km_labels"]
        df_c["Segment"]  = df_c["Cluster"].map(SEG_NAMES)
        peer_df          = df_c[df_c["Segment"] == my_segment]
        peer_avg_p  = peer_df["Price_Per_Night"].mean()
        peer_avg_o  = peer_df["Occupancy_Rate"].mean()
        peer_avg_r  = (peer_df["Price_Per_Night"] * peer_df["Occupancy_Rate"] * 30).mean()

        pc = st.columns(4)
        for col,(v,l) in zip(pc,[
            (my_segment,                   "Segment"),
            (f"AED {peer_avg_p:,.0f}",     "Peer Avg Price"),
            (f"{peer_avg_o:.0%}",           "Peer Avg Occupancy"),
            (f"AED {peer_avg_r:,.0f}",      "Peer Avg Monthly Rev"),
        ]):
            col.markdown(kcard(v,l), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig_seg = px.scatter(
            peer_df.sample(min(350, len(peer_df)), random_state=42),
            x="Price_Per_Night", y="Occupancy_Rate",
            color="Neighborhood", opacity=0.45,
            color_discrete_sequence=PALETTE,
            title=f"Peer listings — {my_segment}",
        )
        fig_seg.add_trace(go.Scatter(
            x=[float(price)], y=[est_occ],
            mode="markers",
            marker=dict(color=C_RED, size=16, symbol="star",
                        line=dict(color="white", width=2)),
            name="YOUR LISTING",
        ))
        _layout(fig_seg, 340, xaxis_title="Price/Night (AED)", yaxis_tickformat=".0%")
        st.plotly_chart(fig_seg, use_container_width=True)

        # 12-month projection
        slabel("📅 12-Month Revenue Projection")
        c_map = {"Winter":C_BLUE,"Spring":C_GREEN,"Summer":C_AMBER,"Autumn":C_RED}
        fig_yr = px.bar(proj_df, x="Month", y="Revenue", color="Season",
                        color_discrete_map=c_map,
                        text=proj_df["Revenue"].map("AED {:,.0f}".format))
        fig_yr.update_traces(textposition="outside", textfont_size=8)
        fig_yr.add_hline(y=monthly_rev, line_dash="dot", line_color="#94A3B8",
                          annotation_text=f"Baseline: AED {monthly_rev:,.0f}",
                          annotation_font_size=10)
        _layout(fig_yr, 320, yaxis_title="Monthly Revenue (AED)")
        st.plotly_chart(fig_yr, use_container_width=True)

        st.markdown(
            f"<div style='background:#F0FDF4;border:1px solid #86EFAC;border-radius:8px;"
            f"padding:12px 18px;text-align:center;margin-top:-8px'>"
            f"<span style='font-size:1.15rem;font-weight:700;color:#065F46'>"
            f"Projected Annual Revenue: AED {annual_rev:,.0f}</span></div>",
            unsafe_allow_html=True)

        # Strategy report
        st.markdown("<br>", unsafe_allow_html=True)
        slabel("📋 Personalised Strategy Report")

        def rec(icon, title, body, c=C_RED):
            st.markdown(
                f"<div class='rcard' style='border-left-color:{c}'>"
                f"<div class='rcard-title'>{icon} {title}</div>"
                f"<div class='rcard-body'>{body}</div></div>",
                unsafe_allow_html=True)

        # Review score
        if review_score < 4.5:
            rec("⭐","Improve Review Score",
                f"Your score ({review_score}) is below the 4.5 threshold that unlocks the occupancy premium. "
                f"Focus on cleanliness, communication and accuracy. Each 0.1 point improvement adds ~3% occupancy.",
                "#EF4444")
        else:
            rec("⭐","Strong Review Score ✓",
                f"Your {review_score}★ puts you in the top tier. Maintain through consistent communication "
                f"and prompt issue resolution.", C_GREEN)

        # Superhost
        if not superhost:
            rec("🏆","Pursue Superhost Status",
                "Superhost listings earn ~22% more monthly revenue. Requirements: >4.8★, >90% response rate, "
                "zero cancellations. Missing this badge is leaving significant revenue on the table.",
                "#EF4444")
        else:
            rec("🏆","Superhost Status Active ✓",
                "Your badge is boosting platform visibility and conversion. Protect it — never cancel and "
                "keep responding fast.", C_GREEN)

        # WiFi + self check-in
        if wifi and self_checkin:
            rec("🛜","Optimal Amenity Bundle: WiFi + Self Check-in ✓",
                "You have the highest-lift amenity combination — 2.8× more likely to achieve 70%+ occupancy "
                "according to association rules.", C_GREEN)
        elif not wifi:
            rec("🛜","Add WiFi — Critical",
                "WiFi is the single highest-return amenity. Its absence significantly reduces booking "
                "probability for virtually all traveller types.", "#EF4444")
        elif not self_checkin:
            rec("🔑","Add Self Check-in",
                "Combined with WiFi this is the top occupancy-boosting bundle. A smart lock or lockbox "
                "costs AED 200–500 and typically pays back within 2 months.", C_AMBER)

        # Pool
        if not pool and dist_center < 5 and prop_type in ["Villa","Penthouse"]:
            rec("🏊","Consider Adding Pool",
                f"Your property type ({prop_type}) and proximity to the city centre ({dist_center:.1f} km) "
                f"place you in the segment where pool commands the highest ROI. "
                f"Market data shows pool-enabled listings generate ~AED {pool_delta:,}/month more.", C_AMBER)

        # Pricing recommendation
        if price_status == "overpriced":
            rec("💰","Reduce Price to Market Rate",
                f"At AED {price}/night you are {((float(price)-market_price)/market_price*100):.0f}% above market. "
                f"Try AED {int(market_price)}–{int(market_price*1.05)} and monitor over 4 weeks. "
                f"Lower price + higher occupancy = more net revenue.", "#EF4444")
        elif price_status == "underpriced":
            rec("💰","Increase Your Price",
                f"You are AED {abs(int(price_gap))} below market. Raise in AED 25 steps every 2 weeks. "
                f"Your reviews ({review_score}★) justify the premium.", C_GREEN)

        # Season
        if season == "Summer":
            rec("📅","Summer Survival Strategy",
                "Summer is Dubai's slowest season (85% baseline demand). Cut minimum nights to 1-2, "
                "offer 7-night discounts of 15-20%, and target business travellers who travel year-round.",
                C_AMBER)
        elif season == "Winter":
            rec("📅","Maximise Winter Peak",
                "Winter is Dubai's peak (120% demand). Raise prices 15-20%, set 3-night weekend minimums, "
                "restrict last-minute discounts, and require advance booking.", C_GREEN)

        # Response rate
        if host_resp_rate < 90:
            rec("📱","Improve Response Rate",
                f"Your response rate ({host_resp_rate}%) is below the 90% threshold that protects your "
                f"Airbnb search ranking. Use auto-messages for enquiries within 24 hours.", C_AMBER)

        # Download
        st.markdown("<br>", unsafe_allow_html=True)
        summary = f"""AIRBNB HOST SIMULATOR — RESULTS SUMMARY
=========================================
Listing:       {prop_type} in {neighborhood} ({room_type})
Bedrooms:      {bedrooms}  |  Guests: {accommodates}
Your Price:    AED {price}/night
Market Price:  AED {market_price:.0f}/night
Status:        {price_status.replace('_',' ').title()}

PROJECTIONS
-----------
Booking Probability : {booking_prob:.0%}
Estimated Occupancy : {est_occ:.0%}
Monthly Revenue     : AED {monthly_rev:,.0f}
Seasonal ({season:<6})    : AED {seasonal_rev:,.0f}
Annual Revenue      : AED {annual_rev:,.0f}

MARKET SEGMENT
--------------
Segment            : {my_segment}
Peer Avg Price     : AED {peer_avg_p:,.0f}
Peer Avg Occupancy : {peer_avg_o:.0%}
Peer Avg Monthly   : AED {peer_avg_r:,.0f}

Generated by Airbnb Revenue Analytics Platform
https://airbnbdiadmia.streamlit.app/
"""
        st.download_button(
            "📥 Download Summary Report (.txt)",
            summary,
            file_name=f"simulation_{neighborhood.replace(' ','_')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    else:
        st.markdown("""
        <div style='background:#FFF8F8;border:2px dashed #FECACA;border-radius:12px;
                    padding:40px;text-align:center;margin-top:24px'>
            <div style='font-size:2.8rem;margin-bottom:10px'>🏠</div>
            <div style='font-size:1.05rem;font-weight:600;color:#1E293B;margin-bottom:6px'>
                Fill in your listing details above and click <strong>Run My Simulation</strong>
            </div>
            <div style='font-size:0.85rem;color:#64748B;max-width:480px;margin:0 auto'>
                You will receive: booking probability gauge · revenue forecast · pricing verdict ·
                market segment analysis · 12-month projection · personalised strategy report · downloadable summary
            </div>
        </div>
        """, unsafe_allow_html=True)
