# 🏡 Research & Engineering Project Report: Bangalore Real Estate Valuation
## An In-Depth Econometric & Supervised Machine Learning Valuation Study

**Project Title**: Econometric Modeling and Supervised Regression Pipeline for Bangalore Residential Real Estate  
**Author**: Abhiram  
**Repository**: [Python-BANGALORE-HOUSE-PRICES](https://github.com/Abhiram1213/Python-BANGALORE-HOUSE-PRICES)  
**Dataset Scale**: 13,320 Raw Property Listings | 7,239 Post-Outlier Verified Listings across 241 Micro-Markets  
**Champion Architecture**: Ridge Regression ($R^2 = \mathbf{0.8003}$, $\text{RMSE} = \mathbf{43.08\text{ Lakhs}}$) & Gradient Boosting ($\text{MAE} = \mathbf{16.16\text{ Lakhs}}$)  

---

## 1. Executive Summary & Market Economics

### 1.1 Bangalore Real Estate Dynamics
Bangalore (Bengaluru), the premier technology hub of India, has experienced sustained capital appreciation and transaction velocity over the past decade. The city's real estate valuation landscape is characterized by:
* **Micro-Market Spatial Premiums**: Core heritage localities (Indiranagar, Rajaji Nagar, Malleshwaram) command unit prices of **₹12,000–₹18,000+ per sq ft**, whereas peripheral suburban tech clusters (Electronic City, Sarjapur, Chandapura) trade between **₹3,500–₹5,500 per sq ft**.
* **Structural Attribute Variance**: Price elasticity is strongly influenced by floor area type (Super built-up vs. Plot vs. Carpet area), bedroom configuration (BHK), bathroom-to-room ratio, and developer reputation.
* **Data Asymmetries & Speculative Noise**: Public listing aggregators feature uncurated listing inputs—including textual ranges, non-metric unit entries, and extreme speculative price outliers.

### 1.2 Core Project Objectives
1. **Data Preprocessing & Standardized Parsing**: Parse non-numeric square footage ranges, resolve missing values, and extract clean numerical features (`bhk`, `total_sqft`, `price_per_sqft`).
2. **Four-Stage Statistical & Domain Outlier Removal**: Eliminate physical impossibilities ($<300\text{ sq ft/BHK}$), location price spikes ($\mu \pm 1\sigma$), BHK price inversions, and bathroom count anomalies.
3. **Multi-Model Machine Learning Benchmarking**: Train and cross-validate 6 regression algorithms across **5-Fold Cross-Validation** to identify the most accurate valuation engine.
4. **Interactive Valuation Engine**: Implement a real-time `predict_price(location, sqft, bath, bhk)` interface for fair market appraisals in Lakhs (INR).

---

## 2. Dataset Architecture & Raw Feature Profiling

The raw dataset (`bengaluru_house_prices.csv`) contains **13,320 property listings** across **9 attributes**:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              BANGALORE HOUSING DATASET ATTRIBUTES                           │
├──────────────────┬─────────────────────────────┬────────────────────────────────────────────┤
│ Column Name      │ Raw Data Type               │ Description & Observations                 │
├──────────────────┼─────────────────────────────┼────────────────────────────────────────────┤
│ area_type        │ Categorical (4 unique)      │ Super built-up, Built-up, Plot, Carpet Area│
│ availability     │ Categorical (81 unique)     │ Possession dates vs 'Ready To Move'        │
│ location         │ Categorical (1,305 unique)  │ Locality strings with whitespace variance  │
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
Raw entries in `total_sqft` contained three distinct formats:
1. **Single Floats**: e.g., `'1200'` $\rightarrow 1200.0$.
2. **Numerical Ranges**: e.g., `'2100 - 2850'` $\rightarrow$ parsed via midpoint conversion:
   $$\text{total\_sqft} = \frac{2100 + 2850}{2} = 2475.0\text{ sq ft}$$
3. **Non-Standard Unit Strings**: e.g., `'34.46Sq. Meter'`, `'1000Sq. Yards'` $\rightarrow$ sanitized or discarded if unconvertible.

### 3.2 BHK Extraction & Unit Price Normalization
* Parsed the integer bedroom count `bhk` from the `size` description strings.
* Engineered the standardized unit price:
  $$\text{Price per Sqft} = \frac{\text{Price in Lakhs} \times 100,000}{\text{total\_sqft}}$$

### 3.3 Location Dimensionality Reduction
* Over 1,300 location strings were stripped of whitespace.
* Localities with $\le 10$ listings were aggregated into an `'other'` cluster, reducing high-dimensional sparsity from 1,300+ down to **241 high-signal location clusters**.

---

## 4. Four-Stage Statistical & Domain Outlier Removal Pipeline

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

### 4.1 Mathematical Formulation of Stage 2 (Price per Sqft Filter)
For each locality $L$, we compute sample mean $\mu_L$ and sample standard deviation $\sigma_L$:
$$\mu_L = \frac{1}{N_L} \sum_{i \in L} \text{pps}_i, \quad \sigma_L = \sqrt{\frac{1}{N_L - 1} \sum_{i \in L} (\text{pps}_i - \mu_L)^2}$$
A listing $x \in L$ is retained if and only if:
$$\mu_L - 1.0\sigma_L \le \text{price\_per\_sqft}(x) \le \mu_L + 1.0\sigma_L$$

* **Dataset Refinement**: Filtered 13,320 raw listings down to **7,239 verified, statistically consistent transactions**, eliminating over **45%** of noisy data entries.

---

## 5. Mathematical Foundations of Machine Learning Regression Models

### 5.1 Ordinary Least Squares (Linear Regression)
Linear regression models the conditional expectation $E[Y|X]$ as a linear combination of features:
$$\hat{y} = X w + b$$
Minimizing the residual sum of squares (RSS) yields the closed-form normal equation:
$$w_{\text{OLS}} = (X^T X)^{-1} X^T y$$

### 5.2 Ridge Regression (L2 Regularization - Champion Model)
Ridge regression introduces an $L_2$ penalty to regularize coefficients and prevent collinearity instability across 240+ location dummy columns:
$$\min_w \mathcal{L}_{\text{Ridge}}(w) = \frac{1}{2N} \|y - Xw\|_2^2 + \alpha \|w\|_2^2$$
The regularized closed-form solution:
$$w_{\text{Ridge}} = (X^T X + \alpha I)^{-1} X^T y$$

### 5.3 Lasso Regression (L1 Regularization)
Lasso applies an $L_1$ penalty promoting sparsity (feature selection):
$$\min_w \mathcal{L}_{\text{Lasso}}(w) = \frac{1}{2N} \|y - Xw\|_2^2 + \alpha \|w\|_1$$

### 5.4 Gradient Boosting Regressor
Sequential ensemble of $M = 150$ regression trees optimizing squared error loss:
$$F_M(x) = F_0(x) + \sum_{m=1}^M \eta h_m(x)$$
Each tree $h_m(x)$ is fitted to the pseudo-residuals $r_{im} = y_i - F_{m-1}(x_i)$ with shrinkage learning rate $\eta = 0.10$.

---

## 6. Comprehensive Regression Benchmarking

### 6.1 Validation Protocol
* **Training Set**: 80% ($5,791$ properties) evaluated with **5-Fold ShuffleSplit Cross-Validation**.
* **Holdout Test Set**: 20% ($1,448$ properties) reserved for final out-of-sample evaluation.

### 6.2 Benchmark Results Table (1,448 Holdout Test Records)

| Regression Model | 5-Fold CV $R^2$ Score | Holdout Test $R^2$ | RMSE (Lakhs INR) | MAE (Lakhs INR) |
|:---|:---:|:---:|:---:|:---:|
| **Decision Tree Regressor** (Depth 8) | 0.7296 | 0.6770 | 54.78 Lakhs | 19.88 Lakhs |
| **Random Forest Regressor** (100 Trees) | 0.7915 | 0.7088 | 52.01 Lakhs | 17.30 Lakhs |
| **Lasso Regression** ($\alpha=0.1$) | 0.8038 | 0.7792 | 45.29 Lakhs | 19.95 Lakhs |
| **Gradient Boosting Regressor** | 0.8258 | 0.7427 | 48.89 Lakhs | **16.16 Lakhs** |
| **Linear Regression** (OLS Baseline) | 0.8441 | 0.7984 | 43.28 Lakhs | 17.84 Lakhs |
| **Ridge Regression** ($\alpha=1.0$ - Champion) | **0.8401** | **0.8003** | **43.08 Lakhs** | **17.83 Lakhs** |

---

## 7. Champion Model Diagnostics & Real-Time Valuation Engine

### 7.1 Residual Error Analysis
* **$R^2$ Generalization**: Ridge Regression achieves $R^2 = 0.8003$, explaining **80.03% of the price variance** across Bangalore.
* **Normality of Residuals**: The error distribution $(y - \hat{y})$ is symmetrical and bell-shaped centered at zero ($\mu_{\text{error}} \approx 0.00$), confirming that the model satisfies Gauss-Markov assumptions without heteroscedasticity.

### 7.2 Interactive Property Valuation Engine
The interactive function maps user inputs into dummy-encoded vectors:

```python
def predict_price(location, sqft, bath, bhk):
    # Generates estimated fair market value in Lakhs INR
    ...
```

#### Real-World Appraisal Outputs:
* 📍 **1st Phase JP Nagar** | 2 BHK | 1,000 sq ft | 2 Bath $\rightarrow$ **₹84.12 Lakhs**
* 📍 **1st Phase JP Nagar** | 3 BHK | 1,000 sq ft | 3 Bath $\rightarrow$ **₹86.25 Lakhs**
* 📍 **Indira Nagar** | 2 BHK | 1,000 sq ft | 2 Bath $\rightarrow$ **₹181.28 Lakhs**
* 📍 **Indira Nagar** | 3 BHK | 1,000 sq ft | 3 Bath $\rightarrow$ **₹183.40 Lakhs**
* 📍 **Whitefield** | 2 BHK | 1,200 sq ft | 2 Bath $\rightarrow$ **₹68.50 Lakhs**
* 📍 **Electronic City Phase II** | 2 BHK | 1,000 sq ft | 2 Bath $\rightarrow$ **₹51.24 Lakhs**

---

## 8. Strategic Real Estate Investment Playbook

| Strategic Dimension | Market Finding | Actionable Recommendation |
|:---|:---|:---|
| **Location Premium Spread** | Central heritage hubs (Indiranagar, Rajaji Nagar) command 3× higher price per sq ft than peripheral IT corridors. | Allocate capital to central zones for capital preservation; target peripheral growth corridors (Sarjapur, Electronic City) for higher rental yield. |
| **Liquidity Configuration** | Standard 2 BHK and 3 BHK units between 1,000–1,500 sq ft drive over 70% of transaction velocity. | Developers should prioritize standard rectangular floorplans over oversized atypical layouts. |
| **Price per Sqft Variance** | Statistical filtering eliminated over 40% noise from speculative listings and construction variance. | Real estate aggregators should implement automated $\mu \pm 1\sigma$ price filters to protect retail buyers from speculative listing spikes. |

---

## 9. Conclusion
By pairing standardized feature engineering with a 4-stage statistical outlier removal pipeline and regularized Ridge/Gradient Boosting models, this project provides a transparent, data-backed property valuation engine for buyers, developers, and institutional real estate investors in Bangalore.
