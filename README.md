# 🏙️ Bangalore House Prices — Advanced Machine Learning & Power BI Master Analytics Suite

An end-to-end residential real estate intelligence system and machine learning valuation pipeline built on **7,269 verified listings across 241 Bangalore micro-markets**, featuring an executive **3-page Power BI Dashboard**, an **interactive HTML web suite**, and **regularized ML valuation models ($R^2 = 82.54\%$)**.

---

## 📑 Project Ecosystem & Deliverables

| Asset | File / Path | Key Capabilities |
|---|---|---|
| **📊 Power BI Master Suite** | [`Bangalore_Advanced_Dashboard.pbix`](file:///c:/Users/abhir/Downloads/git%20ptojects/Python-BANGALORE-HOUSE-PRICES/Bangalore_Advanced_Dashboard.pbix) | 3 Pages, 26 Visuals, Dark Navy Executive Theme (`#0B1929`), Dual-Axis Combo Charts, Floor Area Curves, Micro-market Treemaps |
| **🌐 Interactive Web Dashboard** | [`dashboard.html`](file:///c:/Users/abhir/Downloads/git%20ptojects/Python-BANGALORE-HOUSE-PRICES/dashboard.html) | Chart.js 4.4 Engine, 9 Interactive Charts, 5 Sparkline KPIs, Dynamic Locality Matrix, Real-time Filters |
| **📄 Master Analytics Report (PDF)** | [`Bangalore_Real_Estate_Master_Analytics_Report.pdf`](file:///c:/Users/abhir/Downloads/git%20ptojects/Python-BANGALORE-HOUSE-PRICES/Bangalore_Real_Estate_Master_Analytics_Report.pdf) | Publication-grade executive report with data tables, econometric pricing models, and corridor benchmarks |
| **📋 Exhaustive Technical Report** | [`PROJECT_REPORT.md`](file:///c:/Users/abhir/Downloads/git%20ptojects/Python-BANGALORE-HOUSE-PRICES/PROJECT_REPORT.md) | Full mathematical formulations, OLS/Ridge/Lasso benchmarks, feature breakdowns, and distribution stats |
| **📁 Cleaned Dataset** | [`final_cleaned_data.csv`](file:///c:/Users/abhir/Downloads/git%20ptojects/Python-BANGALORE-HOUSE-PRICES/final_cleaned_data.csv) | 7,269 records, 11 columns, 0 nulls, 4-stage domain-specific outlier filtering |
| **📓 Machine Learning Notebook** | [`Bangalore House Prices.ipynb`](file:///c:/Users/abhir/Downloads/git%20ptojects/Python-BANGALORE-HOUSE-PRICES/Bangalore%20House%20Prices.ipynb) | End-to-end exploratory analysis, feature engineering, and model benchmarking |

---

## 🏛️ Power BI Dashboard Architecture (3 Pages • 26 Visuals)

```
Bangalore_Advanced_Dashboard.pbix
│
├── 📑 Page 1: "1. Market Overview & KPIs" (14 Visuals)
│    ├── 5 Executive KPI Cards (Listings, Avg Price, Unit Rate, Floor Area, Bath Ratio)
│    ├── 2 Interactive Slicers (Locality Filter & BHK Filter)
│    ├── Dual-Axis Combo Chart: Avg Price (Bars) & Rate/Sqft (Line) by BHK
│    ├── Living Floor Area Expansion Curve (Area Chart)
│    ├── Area Type Composition % by BHK (100% Stacked Bar)
│    ├── Top 15 High-Valuation Localities (Ranked Horizontal Bar)
│    ├── Market Inventory Density across Localities (Treemap)
│    └── Area Type Market Share (Donut Chart)
│
├── 📑 Page 2: "2. Micro-Market & Price Analytics" (6 Visuals)
│    ├── Price Elasticity Scatter Plot (Price vs Total Sqft by Locality)
│    ├── Total Capital Volume by Area Type (Treemap)
│    ├── Full Locality Intelligence Data Table (Searchable & Sortable Matrix)
│    ├── BHK Inventory & Demand Funnel (Funnel Chart)
│    └── Area Type × BHK Valuation Multi-Dimensional Stacked Bar
│
└── 📑 Page 3: "3. Feature Valuations & Amenities" (6 Visuals)
     ├── Balcony Count vs Average Price Impact (Column Chart)
     ├── Unit Rate Premium Curve by Bathroom Count (₹/Sqft Area Curve)
     ├── Bathroom Count Market Share Distribution (Donut Chart)
     ├── Clustered Matrix: Avg Price by BHK & Balcony Tier (Clustered Column)
     └── High-Density Localities: Supply Volume (Bars) vs Unit Rate (Line) (Dual-Axis Combo)
```

---

## 📈 Key Market Benchmarks

- **Market Capitalization**: **₹7,039.17 Crores** (₹703,916.93 Lakhs)
- **Mean Property Price**: **₹96.84 Lakhs** (Median: **₹72.54 Lakhs**, IQR: ₹50.00L – ₹110.00L)
- **Mean Unit Rate**: **₹6,101.92 / sqft** (Median: **₹5,666.67 / sqft**)
- **Dominant Configurations**: **2 BHK (49.95%)** + **3 BHK (33.99%)** = **83.94%** total market share
- **Structural Premium**: Gated Plot Villas command **₹8,171.35/sqft** (+39.0% unit rate premium over Super built-up apartments)
- **Top Supply Epicenters**: Whitefield (244 listings), Sarjapur Road (190), Electronic City (162)
- **Top Ultra-Luxury Hubs**: Cunningham Road (Avg ₹744.56L), Giri Nagar (₹402.43L), Benson Town (₹320.00L)

---

## 🤖 Machine Learning Model Benchmarks

$$\hat{Y}_{\text{price}} = \beta_0 + \beta_1(\text{total\_sqft}) + \beta_2(\text{bath}) + \beta_3(\text{balcony}) + \beta_4(\text{bhk}) + \sum_{j=1}^{K} \gamma_j (\text{Locality}_j) + \sum_{m=1}^{M} \delta_m (\text{AreaType}_m)$$

| Algorithm | R² Score (%) | RMSE (₹ Lakhs) | MAE (₹ Lakhs) | Status |
|:---|:---:|:---:|:---:|:---:|
| **Linear Regression (OLS)** | **82.71%** | **₹46.85 L** | **₹18.60 L** | Baseline |
| **Ridge Regression ($L_2$)** | **82.54%** | **₹47.07 L** | **₹18.56 L** | **Champion Model** |
| **Lasso Regression ($L_1$)** | 81.24% | ₹48.80 L | ₹20.34 L | Strong |
| **Random Forest Regressor** | 67.82% | ₹63.91 L | ₹16.78 L | Non-linear tree ensemble |
| **Gradient Boosting Regressor** | 66.48% | ₹65.23 L | ₹19.49 L | Boosted trees |

---

## 🚀 How to Run

### 1. View Power BI Dashboard
Open [`Bangalore_Advanced_Dashboard.pbix`](file:///c:/Users/abhir/Downloads/git%20ptojects/Python-BANGALORE-HOUSE-PRICES/Bangalore_Advanced_Dashboard.pbix) in **Power BI Desktop** and navigate between all 3 tabs at the bottom.

### 2. View Interactive Web Dashboard
```bash
python -m http.server 8080
# Open http://localhost:8080/dashboard.html
```

### 3. Review Master Reports
- Markdown: [`PROJECT_REPORT.md`](file:///c:/Users/abhir/Downloads/git%20ptojects/Python-BANGALORE-HOUSE-PRICES/PROJECT_REPORT.md)
- PDF: [`Bangalore_Real_Estate_Master_Analytics_Report.pdf`](file:///c:/Users/abhir/Downloads/git%20ptojects/Python-BANGALORE-HOUSE-PRICES/Bangalore_Real_Estate_Master_Analytics_Report.pdf)
