# 🏡 Bangalore Real Estate Price Prediction & Market Valuation

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![PowerBI](https://img.shields.io/badge/Power%20BI-Dashboard%20Included-yellow.svg)]()
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()

> **An end-to-end econometric and machine learning valuation pipeline diagnosing spatial real estate pricing patterns across Bangalore, removing domain outliers, benchmarking multiple regression models with 5-Fold Cross-Validation, and providing real-time property valuation.**

---

## 📌 1. Project Background & Business Context

Bangalore (Bengaluru) is one of the most dynamic real estate markets in Asia. Driven by massive IT corridors and rapid urban expansion, residential property prices exhibit extreme variance based on:
- **Location Premiums**: Central heritage hubs (Indiranagar, Rajaji Nagar, Malleshwaram) vs. outer tech corridors (Whitefield, Electronic City, Sarjapur Road).
- **Structural Configurations**: Total square footage, bedroom count (BHK), bathroom-to-room ratio, and construction area type (Super built-up vs. Plot vs. Carpet area).
- **Data Inconsistencies & Speculative Spikes**: Non-standard square footage text entries, speculative listing spikes, and data entry errors.

This project delivers:
1. **Robust Data Preprocessing**: Parsing textual square footage ranges, cleaning missing values, and engineering `bhk`, `total_sqft`, and `price_per_sqft`.
2. **Domain-Driven Outlier Removal**: Statistical filters eliminating unrealistic space-per-room entries ($< 300\text{ sq ft/BHK}$), location price spikes ($\mu \pm 1\sigma$), BHK price inversions, and bathroom count anomalies.
3. **Machine Learning Benchmarking**: Comparing 6 regression algorithms across **5-Fold Cross-Validation** to select the champion valuation model.
4. **Interactive Valuation Function**: Real-time property appraisal function `predict_price(location, sqft, bath, bhk)`.

---

## 🛡️ 2. Domain & Statistical Outlier Removal Pipeline

Raw real estate listings contain severe anomalies that distort linear and ensemble algorithms. We apply a 4-stage cleaning funnel:

```
                            OUTLIER FILTERING FUNNEL
┌───────────────────────────────────────┬─────────────────────────────────────────────────────────────┐
│ Filtering Stage                       │ Rationale & Business Rule                                   │
├───────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 1. Minimum Area per Bedroom           │ Exclude properties with < 300 sq ft per BHK (unrealistic).  │
│ 2. Location-Wise Price Distribution   │ Filter listings outside μ ± 1σ price/sqft per locality.     │
│ 3. BHK Pricing Inversion              │ Eliminate 2 BHKs priced higher than 3 BHKs of similar sqft. │
│ 4. Bathroom-to-Room Ratio             │ Remove listings where Bathrooms > BHK + 2.                  │
└───────────────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 📈 3. Machine Learning Model Benchmarks

All models were evaluated using **5-Fold ShuffleSplit Cross-Validation** on the training split (80%) and tested on an independent holdout set (20% = 1,448 properties).

| Regression Algorithm | 5-Fold CV $R^2$ Score | Holdout Test $R^2$ | RMSE (Lakhs INR) | MAE (Lakhs INR) |
|:---|:---:|:---:|:---:|:---:|
| **Decision Tree Regressor** | 0.7296 | 0.6770 | 54.78 Lakhs | 19.88 Lakhs |
| **Random Forest Regressor** (100 Trees) | 0.7915 | 0.7088 | 52.01 Lakhs | 17.30 Lakhs |
| **Lasso Regression** ($\alpha=0.1$) | 0.8038 | 0.7792 | 45.29 Lakhs | 19.95 Lakhs |
| **Gradient Boosting Regressor** | 0.8258 | 0.7427 | 48.89 Lakhs | **16.16 Lakhs** |
| **Linear Regression** (OLS) | 0.8441 | 0.7984 | 43.28 Lakhs | 17.84 Lakhs |
| **Ridge Regression** ($\alpha=1.0$ - Champion) | **0.8401** | **0.8003** | **43.08 Lakhs** | **17.83 Lakhs** |

### 🏆 Champion Model Highlights (Ridge Regression)
- **Holdout Test $R^2$ Score**: `0.8003` (explains 80% of price variance across 240+ localities)
- **Root Mean Squared Error (RMSE)**: `43.08 Lakhs INR`
- **Mean Absolute Error (MAE)**: `17.83 Lakhs INR`

---

## 💡 4. Real-Time Property Valuation Function

The notebook implements an interactive valuation engine:

```python
def predict_price(location, sqft, bath, bhk):
    # Evaluates one-hot encoded locality + dimensions
    ...
```

### Sample Real-World Valuation Outputs:
* 📍 **1st Phase JP Nagar** | 2 BHK | 1,000 sq ft | 2 Bath $\rightarrow$ **₹84.12 Lakhs**
* 📍 **1st Phase JP Nagar** | 3 BHK | 1,000 sq ft | 3 Bath $\rightarrow$ **₹86.25 Lakhs**
* 📍 **Indira Nagar** | 2 BHK | 1,000 sq ft | 2 Bath $\rightarrow$ **₹181.28 Lakhs**
* 📍 **Indira Nagar** | 3 BHK | 1,000 sq ft | 3 Bath $\rightarrow$ **₹183.40 Lakhs**
* 📍 **Whitefield** | 2 BHK | 1,200 sq ft | 2 Bath $\rightarrow$ **₹68.50 Lakhs**
* 📍 **Electronic City Phase II** | 2 BHK | 1,000 sq ft | 2 Bath $\rightarrow$ **₹51.24 Lakhs**

---

## 💼 5. Real Estate Investment Playbook

| Strategic Dimension | Market Finding | Actionable Recommendation |
|:---|:---|:---|
| **Local Premium Spread** | Prime heritage areas (Indiranagar, Rajaji Nagar) command 3× higher price per sq ft than peripheral tech corridors. | Focus on central zones for long-term capital preservation; target peripheral growth corridors (Sarjapur, Electronic City) for higher rental yield. |
| **Liquidity Configuration** | Standard 2 BHK and 3 BHK units between 1,000–1,500 sq ft drive over 70% of transaction velocity. | Developers should prioritize standard rectangular floorplans over non-standard layouts. |
| **Price per Sqft Variance** | Statistical filtering eliminated over 40% noise from speculative listings and construction variance. | Real estate portals should integrate automated $\mu \pm 1\sigma$ price filters to protect retail buyers from speculative listing spikes. |

---

---

## 📊 6. Power BI Interactive Dashboard Report (`powerbi report.pbix`)

The repository includes an interactive Power BI report connected directly to the cleaned dataset (`pre_processed_data.csv` / `final_cleaned_data.csv`):

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                           POWER BI DASHBOARD VISUAL ARCHITECTURE                            │
├────────────────────┬─────────────────────────────┬──────────────────────────────────────────┤
│ Component          │ Visual Type                 │ Key Metrics & Functionality              │
├────────────────────┼─────────────────────────────┼──────────────────────────────────────────┤
│ 1. Location Slicer │ Interactive Slicer Dropdown │ Filter entire dashboard by 241 Localities│
│ 2. BHK Slicer      │ Multi-Select Slicer Tile    │ Segment properties by 1, 2, 3, 4, 5+ BHK │
│ 3. Spatial Volume  │ Clustered Column Chart      │ Total Property Availability by Location  │
│ 4. Configuration $ │ Column Chart                │ Average & Maximum Price by BHK           │
│ 5. Location Price  │ Clustered Column Chart      │ Average Listing Price across Localities  │
│ 6. Unit Absorption │ Horizontal Bar Chart        │ Availability Volume by Bedroom Count     │
│ 7. Unit Economics  │ Column Chart                │ Average Price per Sqft by BHK            │
└────────────────────┴─────────────────────────────┴──────────────────────────────────────────┘
```

---

## 📁 7. Repository Layout

```
Python-BANGALORE-HOUSE-PRICES/
├── Bangalore House Prices.ipynb              # Complete, executed, and documented Jupyter Notebook
├── Bangalore_House_Prices_Valuation_Report.pdf # High-depth PDF technical valuation report
├── PROJECT_REPORT.md                         # Detailed markdown project report
├── bengaluru_house_prices.csv                # Raw dataset (13,320 properties × 9 columns)
├── pre_processed_data.csv                    # Cleaned pre-processed dataset (0 nulls, 7,269 rows)
├── final_cleaned_data.csv                    # Cleaned dataset for Power BI data refresh
├── powerbi report.pbix                       # Interactive Power BI dashboard report
├── requirements.txt                          # Environment dependencies
└── README.md                                 # Human-written technical project documentation
```

---

## 🚀 7. Quickstart & How to Run

### Prerequisites
- Python 3.9+ installed
- Jupyter Notebook or VS Code with Jupyter extension

### Setup
```bash
# 1. Clone the repository
git clone https://github.com/Abhiram1213/Python-BANGALORE-HOUSE-PRICES.git
cd Python-BANGALORE-HOUSE-PRICES

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt
```

### Run the Notebook
Launch Jupyter Notebook to inspect plots and run the live prediction engine:
```bash
jupyter notebook "Bangalore House Prices.ipynb"
```
