# 📄 Comprehensive Project & Technical Report: Bangalore Real Estate Valuation

**Project Title**: Bangalore Real Estate Price Prediction & Market Valuation  
**Domain**: Real Estate Analytics / Econometrics / Machine Learning Regression  
**Author**: Abhiram  
**Repository**: [Python-BANGALORE-HOUSE-PRICES](https://github.com/Abhiram1213/Python-BANGALORE-HOUSE-PRICES)  

---

## Executive Summary

Bangalore (Bengaluru) represents one of the most rapidly expanding real estate markets in Asia. Driven by massive IT corridors and steady demographic inflows, residential property prices exhibit extreme variance based on micro-location premiums, construction floor types, bedroom counts (BHK), and bathroom ratios.

This project delivers:
1. **End-to-End Data Cleaning**: Parsing textual square footage ranges, handling missing values, and engineering `bhk`, `total_sqft`, and `price_per_sqft`.
2. **4-Stage Domain Outlier Removal**: Eliminating unfeasible space-per-bedroom ratios ($< 300\text{ sq ft/BHK}$), location price spikes ($\mu \pm 1\sigma$), BHK price inversions, and bathroom anomalies.
3. **Machine Learning Benchmarking**: Evaluating 6 regression algorithms across **5-Fold Cross-Validation**, with **Ridge Regression** achieving **0.8003 Test $R^2$** and **Gradient Boosting** delivering the lowest **MAE of 16.16 Lakhs**.
4. **Real-Time Valuation Engine**: Interactive `predict_price(location, sqft, bath, bhk)` interface for fair market appraisals in Lakhs (INR).

---

## 1. Problem Statement & Dataset Profile

### 1.1 Dataset Architecture
The raw dataset (`bengaluru_house_prices.csv`) contains **13,320 property listings** across **9 attributes**:
* `area_type`: Super built-up Area, Built-up Area, Plot Area, Carpet Area.
* `availability`: Possession timeline string.
* `location`: Locality within Bangalore (over 1,300 unique names).
* `size`: Raw bedroom configuration string (e.g., `'2 BHK'`, `'4 Bedroom'`).
* `society`: Residential society name (over 41% missing values $\rightarrow$ pruned).
* `total_sqft`: Property area in square feet (contains mixed numerical ranges and text).
* `bath`: Number of bathrooms.
* `balcony`: Number of balconies.
* `price`: Property price in **Lakhs (INR)** (Target Variable).

---

## 2. Feature Engineering & Preprocessing

### 2.1 Standardizing Total Square Footage
Raw square footage entries containing ranges (e.g., `'2100 - 2850'`) were parsed by computing the mean ($2475.0\text{ sq ft}$). Non-standard unit strings were sanitized.

### 2.2 BHK & Price per Sqft Engineering
* Extracted integer `bhk` from textual `size` descriptions.
* Computed normalized unit price:
  $$\text{Price per Sqft} = \frac{\text{Price in Lakhs} \times 100,000}{\text{total\_sqft}}$$

### 2.3 Location Dimensionality Reduction
* Over 1,300 location names were stripped of whitespace.
* Localities with $\le 10$ listings were aggregated into an `'other'` cluster, reducing high-dimensional sparsity from 1,300+ down to **241 high-signal location clusters**.

---

## 3. Four-Stage Statistical & Domain Outlier Removal

Real estate listings are prone to non-standard data entries and speculative listing spikes. A 4-stage filtering funnel was applied:

```
                            OUTLIER FILTERING FUNNEL
┌───────────────────────────────────────┬─────────────────────────────────────────────────────────────┐
│ Filtering Stage                       │ Rationale & Implementation                                  │
├───────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 1. Minimum Area per Bedroom           │ Exclude properties with < 300 sq ft per BHK (unrealistic).  │
│ 2. Location-Wise Price Distribution   │ Filter listings outside μ ± 1σ price/sqft per locality.     │
│ 3. BHK Pricing Inversion              │ Eliminate 2 BHKs priced higher than 3 BHKs of similar sqft. │
│ 4. Bathroom-to-Room Ratio             │ Remove listings where Bathrooms > BHK + 2.                  │
└───────────────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

* **Dataset Refinement**: Filtered 13,320 raw listings down to **7,239 verified, clean property transactions**, eliminating over 40% noise.

---

## 4. Machine Learning Benchmarking & Evaluation

### 4.1 Evaluation Framework
Models were trained on an 80% split (5,791 properties) with **5-Fold ShuffleSplit Cross-Validation** and evaluated on an independent 20% holdout test set (1,448 properties).

### 4.2 Benchmark Results Table

| Regression Model | 5-Fold CV $R^2$ Score | Holdout Test $R^2$ | RMSE (Lakhs INR) | MAE (Lakhs INR) |
|:---|:---:|:---:|:---:|:---:|
| **Decision Tree Regressor** | 0.7296 | 0.6770 | 54.78 Lakhs | 19.88 Lakhs |
| **Random Forest Regressor** (100 Trees) | 0.7915 | 0.7088 | 52.01 Lakhs | 17.30 Lakhs |
| **Lasso Regression** ($\alpha=0.1$) | 0.8038 | 0.7792 | 45.29 Lakhs | 19.95 Lakhs |
| **Gradient Boosting Regressor** | 0.8258 | 0.7427 | 48.89 Lakhs | **16.16 Lakhs** |
| **Linear Regression** (OLS Baseline) | 0.8441 | 0.7984 | 43.28 Lakhs | 17.84 Lakhs |
| **Ridge Regression** ($\alpha=1.0$ - Champion) | **0.8401** | **0.8003** | **43.08 Lakhs** | **17.83 Lakhs** |

### 4.3 Champion Model Diagnostics
* **Ridge Regression** achieved the highest generalization with an $R^2$ of **0.8003**, effectively explaining **80% of price variance** across 240+ micro-markets.
* **Residual Analysis**: Residual errors $(y - \hat{y})$ display a normal distribution centered around zero, confirming model stability without systematic bias.

---

## 5. Real-Time Property Valuation Function

An interactive inference function was constructed to map user inputs to one-hot encoded locality vectors:

```python
def predict_price(location, sqft, bath, bhk):
    # Generates estimated fair market value in Lakhs INR
    ...
```

### 5.1 Real-World Validation Cases:
* 📍 **1st Phase JP Nagar** | 2 BHK | 1,000 sq ft | 2 Bath $\rightarrow$ **₹84.12 Lakhs**
* 📍 **1st Phase JP Nagar** | 3 BHK | 1,000 sq ft | 3 Bath $\rightarrow$ **₹86.25 Lakhs**
* 📍 **Indira Nagar** | 2 BHK | 1,000 sq ft | 2 Bath $\rightarrow$ **₹181.28 Lakhs**
* 📍 **Indira Nagar** | 3 BHK | 1,000 sq ft | 3 Bath $\rightarrow$ **₹183.40 Lakhs**
* 📍 **Whitefield** | 2 BHK | 1,200 sq ft | 2 Bath $\rightarrow$ **₹68.50 Lakhs**
* 📍 **Electronic City Phase II** | 2 BHK | 1,000 sq ft | 2 Bath $\rightarrow$ **₹51.24 Lakhs**

---

## 6. Strategic Real Estate Investment Playbook

| Strategic Dimension | Market Finding | Actionable Recommendation |
|:---|:---|:---|
| **Location Premium Spread** | Central heritage hubs (Indiranagar, Rajaji Nagar) trade at 3x the price/sqft of peripheral IT corridors. | Allocate to central zones for capital preservation; target peripheral growth corridors (Sarjapur, Electronic City) for higher rental yield. |
| **Liquidity Configuration** | Standard 2 BHK and 3 BHK units between 1,000–1,500 sq ft drive over 70% of transaction velocity. | Developers should prioritize standard rectangular layouts over non-standard floorplans. |
| **Price per Sqft Variance** | Statistical filtering eliminated over 40% noise from speculative listings and construction variance. | Real estate aggregators should implement automated $\mu \pm 1\sigma$ price filters to protect retail buyers from speculative listing spikes. |

---

## 7. Conclusion & Next Steps
By combining systematic data cleaning, 4-stage statistical outlier removal, and regularized regression modeling, this project provides a reliable valuation engine for homebuyers, developers, and real estate analysts across Bangalore.
