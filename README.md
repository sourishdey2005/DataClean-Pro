# 📊 DataClean Pro | Advanced Data Preprocessing & Insight Dashboard

**DataClean Pro** is a high-performance, professional-grade Streamlit application designed to transform messy, real-world datasets into clean, analysis-ready intelligence. Whether you're dealing with missing values, outliers, inconsistent formatting, or high-dimensional complexity, DataClean Pro provides the tools to audit, clean, and visualize your data with precision.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sourishdey2005-dataclean-pro.streamlit.app/)

## 🚀 Key Features

### 1. 🧹 Smart Data Preprocessing
- **Automatic Schema Detection**: Instant recognition of numeric, categorical, and datetime formats.
- **Bulk Operations**: Auto-pruning of empty columns and intelligent duplicate management.
- **Inconsistent Text Fixer**: Standardize categories, trim whitespace, and normalize casing with one click.

### 2. 🔍 Missing Value Intelligence
- **Interactive Heatmaps**: Visualize exactly where your data is thin.
- **Advanced Imputation**: Multi-strategy support (Mean, Median, Mode, Forward/Backward Fill, or Constant Value).
- **Comparison Engine**: See "Before vs After" impact on your data health.

### 3. 🛡️ Outlier Detection & Treatment
- **Statistical Auditing**: Detect anomalies using IQR, Z-Score, or Percentile-based methods.
- **Dynamic Treatment**: Cap (Winsorize), remove, or replace outliers with medians directly from the UI.
- **Visual Validation**: Box and Strip plots with real-time outlier highlighting.

### 4. 📊 Ultimate Visual Intelligence (40+ Chart Types)
- **Univariate**: Histograms, KDE, Violin, and Rug plots.
- **Bivariate**: Regression Scatter plots, 2D Density Heatmaps, and Hex-Contour charts.
- **Multivariate**: Parallel Coordinates, Radar Charts, and **2D/3D PCA Dimensionality Reduction**.
- **Temporal**: Area flows, rolling averages, and stacked trend analysis.
- **Geospatial**: Automatic Mapbox rendering for latitude/longitude data.

### 5. 💡 Automated Insight Engine
- **KPI Feature Pulse**: Instant statistics (Mean, Median, Std Dev) for any selected segment.
- **Segment Benchmarker**: Automatically compare filtered sub-groups against the overall population.
- **Pattern Recognition**: Auto-detection of strong correlations and distribution shifts.

## 🛠️ Tech Stack
- **Backend**: Python 3.9+
- **Framework**: [Streamlit](https://streamlit.io/)
- **Analysis**: Pandas, NumPy, Scipy, Scikit-Learn
- **Visualization**: Plotly Express, Plotly Graph Objects
- **Models**: Statsmodels (OLS Regression), IsolationForest (Anomalies)

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sourishdey2005/-DataClean-Pro.git
   cd -DataClean-Pro
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the dashboard**:
   ```bash
   streamlit run app.py
   ```

## 🤝 Project Contribution
Made with ❤️ by [Sourish Dey](https://sourishdeyportfolio.vercel.app/)

---
*Transforming raw data into strategic intelligence.*
