# Subscription Burnout Segmentation: Time-Series Analysis of Video Engagement Patterns

## 📋 Overview

This project applies **machine learning clustering algorithms** to YouTube trending video data to identify and segment videos based on **burnout patterns**. By analyzing engagement metrics over time, we classify videos into distinct segments that represent different engagement trajectories, enabling data-driven strategies for content creators and platform managers.

The project uses **K-Means clustering with Particle Swarm Optimization (PSO)** to identify the optimal number of segments and provides actionable insights into video engagement dynamics.

---

## 🎯 Problem Statement

YouTube content creators and platform managers need to understand **why and when videos lose momentum**. Understanding engagement burnout patterns helps:

- **Identify at-risk content** before decline accelerates
- **Segment audiences** by viewing behavior patterns
- **Optimize intervention strategies** for different content segments
- **Predict sustainability** of video engagement over time

This project addresses these challenges by segmenting videos into burnout patterns using time-series feature engineering.

---

## 📊 Dataset

**Source:** YouTube US Trending Videos Dataset
- **Kaggle Dataset:** [YouTube New - datasnaek](https://www.kaggle.com/datasets/datasnaek/youtube-new/data)
- **Total Records:** 40,949 trending events
- **Unique Videos:** 5,644 distinct videos
- **Time Window:** 8-point observation periods for feature extraction
- **Key Metrics:** Video ID, Trending Date, Views (engagement proxy)

**Data Location:** `Data/USvideos.csv`

**Attribution:** Dataset sourced from Kaggle's YouTube New dataset. Original data compiled from YouTube's trending videos API across multiple countries and time periods.

---

## 🔍 Methodology

### Feature Engineering

Three burnout indicators are extracted from the 8-point time-series window:

1. **Slope** - Measures gradual decline/growth in engagement
   - Formula: Linear regression coefficient of views over time points
   - Interpretation: Negative slope = declining engagement

2. **Peak Drop** - Measures sudden engagement collapse
   - Formula: Max views - Final views in window
   - Interpretation: Large values indicate sharp declines

3. **Initial Engagement** - Baseline popularity at start of window
   - Formula: Mean of first 3 observations
   - Interpretation: Higher values = strong initial traction

### Clustering Methodology

**Algorithm:** K-Means Clustering (Optimal: k=4)

**Optimization:** Particle Swarm Optimization (PSO)
- Particles: 10
- Iterations: 50
- Objective: Maximize silhouette score
- Search space: k ∈ [2, 10]

**Evaluation Metrics:**
- **Silhouette Score:** Measures cluster cohesion and separation
- **Calinski-Harabasz Index:** Ratio of between-cluster to within-cluster variance
- **Davies-Bouldin Index:** Average similarity ratio between clusters

---

## 📁 Project Structure

```
Subscription-Burnout-Segmentation/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── batch_processor.py                 # Daily inference pipeline
│
├── Data/
│   └── USvideos.csv                   # YouTube trending dataset
│
├── notebooks/
│   └── subscription_segmentation.ipynb # Complete analysis & visualization
│
├── src/
│   └── inference.py                   # Feature extraction & prediction
│
├── models/
│   ├── k-means_best_model.pkl        # Trained K-Means model
│   └── feature_scaler.pkl            # StandardScaler for normalization
│
└── reports/
    ├── 1_segment_distribution_pie.png
    ├── 2_slope_vs_peak_drop_scatter.png
    ├── 3_feature_boxplots.png
    ├── 4_segment_means_bar.png
    ├── 5_time_series_examples.png
    ├── 6_correlation_heatmap.png
    └── 7_silhouette_scores.png
```

---

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/shahaanabbas-13/Subscription-Burnout-Segmentation.git
cd Subscription-Burnout-Segmentation
```

### 2. Create Virtual Environment

```bash
python3.12 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Analysis Notebook

```bash
jupyter notebook notebooks/subscription_segmentation.ipynb
```

### 5. Run Batch Inference

```bash
python batch_processor.py
```

This will:
- Load the latest data
- Extract burnout features
- Run segmentation
- Generate CSV reports with high-risk videos
- Save timestamped results to `reports/`

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | Latest | Data manipulation & analysis |
| numpy | Latest | Numerical computing |
| scikit-learn | Latest | Machine learning algorithms |
| matplotlib | Latest | Static visualizations |
| seaborn | Latest | Statistical data visualization |
| pyswarms | Latest | Particle Swarm Optimization |
| jupyter | Latest | Interactive notebooks |
| joblib | Latest | Model serialization |

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## 📈 Key Results

### Segment Distribution
- **4 distinct burnout segments** identified through PSO optimization
- **Optimal Silhouette Score:** Validated through K value evaluation
- **169 high-risk videos** identified in primary burnout segment

### Segment Profiles
Each segment represents a unique engagement trajectory:
- **Segment 0:** Rapid burners (high peak drop)
- **Segment 1:** Steady performers (low slope decline)
- **Segment 2:** Slow starters (low initial engagement)
- **Segment 3:** Stable content (minimal volatility)

### Feature Correlations
Burnout features show meaningful patterns:
- Slope and Peak Drop are moderately correlated (r ≈ 0.45)
- Initial Engagement weak predictor of burnout severity
- Segment assignment strongly influenced by feature combination

---

## 📊 Visualizations

The analysis generates **7 comprehensive visualizations**:

1. **Donut Pie Chart** - Segment distribution overview
2. **Scatter Plot** - Slope vs Peak Drop relationships
3. **Box Plots** - Feature variability within segments
4. **Bar Chart** - Average feature values per segment
5. **Time Series** - Real engagement trends by segment
6. **Correlation Heatmap** - Feature interdependencies
7. **Silhouette Analysis** - Model quality validation

All visualizations saved to `reports/` at 300 DPI for publication quality.

---

## 💻 Core Components

**Feature Extraction** - `src/inference.py` extracts Slope, Peak_Drop, and Initial_Engagement from time-series data.

**Prediction** - `src/inference.py` loads the trained model and scaler to segment new videos.

**Batch Processing** - `batch_processor.py` automates daily inference and generates high-risk video reports.

---

## 🔧 Configuration

**Model Parameters** - Adjust `time_window_points`, `MODEL_PATH`, and `SCALER_PATH` in `src/inference.py`.

**PSO Parameters** - Modify `c1`, `c2`, `w`, `n_particles`, and bounds in the notebook to fine-tune optimization.

**Default Settings** - Feature window: 8 points | PSO particles: 10 | Cluster range: k ∈ [2, 10]

---

## 📖 Quick Usage

**Segment Videos:**
```python
from src.inference import load_and_predict_new_data
results = load_and_predict_new_data(df)
```

**Filter High-Risk Content:**
```python
high_risk = results[results['Segment'] == 0]
```

**Run Batch Processing:**
```bash
python batch_processor.py
```


## 📋 Reproducibility

1. Install Python 3.12+
2. Run `pip install -r requirements.txt`
3. Execute notebook: `jupyter notebook notebooks/subscription_segmentation.ipynb`
4. Verify models in `models/` directory
5. Run `python batch_processor.py` for inference

-------