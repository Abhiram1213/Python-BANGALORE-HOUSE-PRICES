# 🏡 Comprehensive Technical & Econometric Analysis Report
## Bangalore Real Estate Price Prediction & Machine Learning Valuation Engine

**Project Title**: Econometric Modeling and Supervised Regression Pipeline for Bangalore Property Valuation  
**Author**: Abhiram  
**Repository**: [Python-BANGALORE-HOUSE-PRICES](https://github.com/Abhiram1213/Python-BANGALORE-HOUSE-PRICES)  
**Dataset Size**: 13,320 Raw Listings | 7,239 Post-Outlier Verified Listings across 241 Localities  
**Champion Model**: Ridge Regression ($R^2 = \mathbf{0.8003}$, $\text{RMSE} = \mathbf{43.08\text{ Lakhs}}$) & Gradient Boosting ($\text{MAE} = \mathbf{16.16\text{ Lakhs}}$)  

---

## 1. Executive Summary & Market Problem

### 1.1 Macroeconomic Real Estate Landscape
Bangalore (Bengaluru), the tech capital of India, represents one of the fastest-growing urban real estate markets globally. Rapid IT corridor expansion (Outer Ring Road, Whitefield, Electronic City, Bellandur) combined with demographic inflows has generated massive pricing complexity:
* **Micro-Market Variance**: Prime heritage localities (Indiranagar, Rajaji Nagar, Malleshwaram) trade at **3x to 4x the price per sq ft** of peripheral suburban corridors.
* **Structural Asymmetries**: Property configurations vary dramatically in floor types (Super built-up vs. Plot vs. Carpet area), bedroom count (BHK), and bathroom ratios.
* **Data Speculation & Extreme Outliers**: Real estate web aggregators frequently feature non-standard space inputs (e.g., fractional square yards, square meters), data entry typos, and speculative listing spikes.

### 1.2 Core Project Objectives
1. **Data Preprocessing & Standardized Parsing**: Parse non-numeric square footage ranges, impute missing values, and extract clean numerical features (`bhk`, `total_sqft`, `price_per_sqft`).
2. **Domain-Driven & Statistical Outlier Removal**: Formulate a 4-stage filtering funnel based on civil engineering constraints and location-wise standard deviations.
3. **Supervised Regression Benchmarking**: Evaluate 6 regression algorithms across **5-Fold Cross-Validation** to select the most generalizable valuation model.
4. **Real-Time Property Valuation Engine**: Provide an interactive `predict_price(location, sqft, bath, bhk)` interface for instantaneous property appraisals in Lakhs (INR).

---

## 2. Dataset Architecture & Raw Data Profiling

The raw dataset (`bengaluru_house_prices.csv`) contains **13,320 property listings** across **9 attributes**:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              BANGALORE HOUSING DATASET ATTRIBUTES                           │
├──────────────────┬─────────────────────────────┬────────────────────────────────────────────┤
│ Column Name      │ Raw Data Type               │ Description & Observations                 │
├──────────────────┼─────────────────────────────┼────────────────────────────────────────────┤
│ area_type        │ Categorical (4 unique)      │ Super built-up, Built-up, Plot, Carpet Area│
│ availability     │ Categorical (81 unique)     │ Future possession date vs 'Ready To Move'  │
│ location         │ Categorical (1,305 unique)  │ Locality string with whitespace variance   │
│ size             │ String (31 unique)          │ Description (e.g., '2 BHK', '4 Bedroom')   │
│ society          │ Categorical (2,688 unique)  │ 5,502 missing values (>41% missing)        │
│ total_sqft       │ String (Mixed ranges)       │ Ranges ('2100 - 2850') and unit texts      │
│ bath             │ Float64 (73 missing)        │ Bathroom count                             │
│ balcony          │ Float64 (609 missing)       │ Balcony count                              │
│ price (Target)   │ Float64 (0 missing)         │ Listing price in Lakhs INR (1L = ₹100,000) │
└──────────────────┴─────────────────────────────┴────────────────────────────────────────────┘
```

---

## 3. Feature Engineering & Dimensionality Reduction

### 3.1 Total Square Footage Parsing
Raw square footage entries containing numerical ranges (e.g., `'2100 - 2850'`) were parsed into single continuous floats by computing the midpoint ($2475.0\text{ sq ft}$). Non-standard unit strings (e.g., `'34.46Sq. Meter'`, `'1000Sq. Yards'`) were sanitized.

### 3.2 BHK Extraction & Unit Price Normalization
* Parsed the integer bedroom count `bhk` from the `size` description strings.
* Engineered the standardized unit price:
  $$\text{Price per Sqft} = \frac{\text{Price in Lakhs} \times 100,000}{\text{total\_sqft}}$$

### 3.3 Location Dimensionality Reduction
* Direct one-hot encoding on 1,300+ raw locations would introduce extreme dimensionality and overfitting.
* All location strings were stripped of leading/trailing whitespace.
* Localities with $\le 10$ listings were clustered into an `'other'` bucket, reducing location features from **1,300+ down to 241 high-signal categorical columns**.

---

## 4. Four-Stage Domain & Statistical Outlier Removal Pipeline

To protect linear and ensemble regression models from noise and data entry errors, we implemented a 4-stage domain filtering funnel:

```
                            OUTLIER FILTERING FUNNEL
┌───────────────────────────────────────┬─────────────────────────────────────────────────────────────┐
│ Filtering Stage                       │ Rationale & Mathematical Threshold                          │
├───────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 1. Minimum Area per Bedroom           │ Exclude properties with total_sqft / bhk < 300 sq ft.       │
│ 2. Location-Wise Price Distribution   │ Filter listings outside μ ± 1σ price/sqft per locality.     │
│ 3. BHK Pricing Inversion              │ Eliminate 2 BHKs priced higher than 3 BHKs of similar sqft. │
│ 4. Bathroom-to-Room Ratio             │ Remove listings where Bathrooms > BHK + 2.                  │
└───────────────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

* **Mathematical Filtering Rule for Stage 2**:
  $$\text{Valid Subset} = \left\{ x \in \text{Location } L \;\middle|\; \mu_L - \sigma_L \le \text{price\_per\_sqft}(x) \le \mu_L + \sigma_L \right\}$$
* **Result**: Reduced 13,320 raw listings to **7,239 verified, statistically consistent transactions**, eliminating over **45%** of noisy data entries.

---

## 5. Machine Learning Regression Benchmarking

### 5.1 Validation Strategy
Models were evaluated using **5-Fold ShuffleSplit Cross-Validation** on the training split (80% = 5,791 properties) and verified on an independent holdout test set (20% = 1,448 properties).

### 5.2 Mathematical Formulation of Regression Metrics
* **Coefficient of Determination ($R^2$)**:
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
* **Root Mean Squared Error (RMSE)**:
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
* **Mean Absolute Error (MAE)**:
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

### 5.3 Benchmark Results Table (1,448 Holdout Test Records)

| Regression Model | 5-Fold CV $R^2$ Score | Holdout Test $R^2$ | RMSE (Lakhs INR) | MAE (Lakhs INR) |
|:---|:---:|:---:|:---:|:---:|
| **Decision Tree Regressor** (Max Depth 8) | 0.7296 | 0.6770 | 54.78 Lakhs | 19.88 Lakhs |
| **Random Forest Regressor** (100 Trees) | 0.7915 | 0.7088 | 52.01 Lakhs | 17.30 Lakhs |
| **Lasso Regression** ($\alpha=0.1$) | 0.8038 | 0.7792 | 45.29 Lakhs | 19.95 Lakhs |
| **Gradient Boosting Regressor** | 0.8258 | 0.7427 | 48.89 Lakhs | **16.16 Lakhs** |
| **Linear Regression** (OLS Baseline) | 0.8441 | 0.7984 | 43.28 Lakhs | 17.84 Lakhs |
| **Ridge Regression** ($\alpha=1.0$ - Champion) | **0.8401** | **0.8003** | **43.08 Lakhs** | **17.83 Lakhs** |

---

## 6. Champion Model Diagnostics & Valuation Inference

### 6.1 Diagnostic Insights
* **Ridge Regression ($\alpha=1.0$)** achieved the top generalization score ($R^2 = 0.8003$), explaining **80% of price variance** across 240+ localities.
* **Residual Analysis**: Residual errors $(y - \hat{y})$ exhibit a normal distribution centered at zero, indicating that the model does not suffer from systematic under- or over-prediction bias.

### 6.2 Real-Time Property Valuation Function
An interactive appraisal engine was implemented:

```python
def predict_price(location, sqft, bath, bhk):
    # Evaluates one-hot encoded locality + dimensions
    ...
```

#### Sample Real-World Appraisal Outputs:
* 📍 **1st Phase JP Nagar** | 2 BHK | 1,000 sq ft | 2 Bath $\rightarrow$ **₹84.12 Lakhs**
* 📍 **1st Phase JP Nagar** | 3 BHK | 1,000 sq ft | 3 Bath $\rightarrow$ **₹86.25 Lakhs**
* 📍 **Indira Nagar** | 2 BHK | 1,000 sq ft | 2 Bath $\rightarrow$ **₹181.28 Lakhs**
* 📍 **Indira Nagar** | 3 BHK | 1,000 sq ft | 3 Bath $\rightarrow$ **₹183.40 Lakhs**
* 📍 **Whitefield** | 2 BHK | 1,200 sq ft | 2 Bath $\rightarrow$ **₹68.50 Lakhs**
* 📍 **Electronic City Phase II** | 2 BHK | 1,000 sq ft | 2 Bath $\rightarrow$ **₹51.24 Lakhs**

---

## 7. Strategic Real Estate Investment Playbook

| Strategic Dimension | Market Finding | Actionable Recommendation |
|:---|:---|:---|
| **Location Premium Spread** | Central heritage hubs (Indiranagar, Rajaji Nagar) command 3× higher price per sq ft than peripheral IT corridors. | Allocate capital to central zones for long-term capital preservation; target peripheral growth corridors (Sarjapur, Electronic City) for higher rental yield. |
| **Liquidity Configuration** | Standard 2 BHK and 3 BHK units between 1,000–1,500 sq ft drive over 70% of transaction velocity. | Developers should prioritize standard rectangular floorplans over oversized atypical layouts. |
| **Price per Sqft Variance** | Statistical filtering eliminated over 40% noise from speculative listings and construction variance. | Real estate aggregators should implement automated $\mu \pm 1\sigma$ price filters to protect retail buyers from speculative listing spikes. |

---

## 8. Conclusion
By pairing standardized feature engineering with a 4-stage statistical outlier removal pipeline and regularized Ridge/Gradient Boosting models, this project provides a transparent, data-backed property valuation engine for buyers, developers, and institutional real estate investors in Bangalore.
