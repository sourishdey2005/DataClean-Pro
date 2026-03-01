import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from scipy import stats
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="DataClean Pro | Advanced Insight Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    /* Force white background */
    .stApp {
        background-color: #ffffff;
        color: #1e1e1e;
    }
    .main {
        background-color: #ffffff;
    }
    /* Metric Card Styling */
    [data-testid="stMetricValue"] {
        color: #1e1e1e !important;
    }
    .stMetric {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    /* Header and Text Styling */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #1e1e1e !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        color: #495057;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        border-bottom: 2px solid #2e7bcf !important;
        color: #2e7bcf !important;
        font-weight: 700;
    }
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f1f3f5 !important;
        border-right: 1px solid #dee2e6;
    }
    </style>
    """, unsafe_allow_html=True)

# Set global Plotly template
px.defaults.template = "plotly_white"

# --- Session State Initialization ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'imputed_mask' not in st.session_state:
    st.session_state.imputed_mask = None

# --- Helper Functions ---
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, parse_dates=True)
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                except (ValueError, TypeError):
                    pass
        else:
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def auto_clean_nan_columns(df):
    initial_cols = df.columns.tolist()
    df = df.dropna(axis=1, how='all')
    removed_cols = list(set(initial_cols) - set(df.columns.tolist()))
    return df, removed_cols

def get_column_metadata(df):
    meta = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = df[col].isna().sum()
        missing_pct = (missing / len(df)) * 100
        unique = df[col].nunique()
        stat_min = stat_max = stat_mean = "N/A"
        if pd.api.types.is_numeric_dtype(df[col]):
            stat_min = f"{df[col].min():.2f}"
            stat_max = f"{df[col].max():.2f}"
            stat_mean = f"{df[col].mean():.2f}"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            stat_min = df[col].min().strftime('%Y-%m-%d')
            stat_max = df[col].max().strftime('%Y-%m-%d')
        meta.append({
            "Column": col, "Type": dtype, "Missing": missing,
            "Missing %": f"{missing_pct:.1f}%", "Unique": unique,
            "Min": stat_min, "Max": stat_max, "Mean": stat_mean
        })
    return pd.DataFrame(meta)

def plot_missing_heatmap(df):
    if df.isna().sum().sum() == 0: return None
    mask = df.isna()
    fig = px.imshow(mask.T, labels=dict(x="Row Index", y="Column Name", color="Is Missing"),
                    color_continuous_scale=[[0, 'white'], [1, '#ef4444']], title="Missing Value Heatmap")
    fig.update_layout(coloraxis_showscale=False)
    return fig

def apply_filters(df):
    filtered_df = df.copy()
    
    with st.sidebar:
        with st.expander("🔍 Advanced Filter Panel", expanded=False):
            st.write("Drill down into specific segments.")
            
            # 1. Global Search
            search = st.text_input("📝 Search rows by value", "")
            if search:
                mask = np.column_stack([filtered_df[col].astype(str).str.contains(search, case=False, na=False) for col in filtered_df.columns])
                filtered_df = filtered_df.loc[mask.any(axis=1)]
            
            # 2. Numeric Filters
            num_cols = filtered_df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                min_v, max_v = float(filtered_df[col].min()), float(filtered_df[col].max())
                if min_v != max_v:
                    rng = st.slider(f"Range: {col}", min_v, max_v, (min_v, max_v))
                    filtered_df = filtered_df[(filtered_df[col] >= rng[0]) & (filtered_df[col] <= rng[1])]
            
            # 3. Categorical Filters
            cat_cols = filtered_df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                opts = filtered_df[col].unique().tolist()
                sel = st.multiselect(f"Filter: {col}", opts)
                if sel:
                    filtered_df = filtered_df[filtered_df[col].isin(sel)]
            
            # 4. Date Filters
            date_cols = filtered_df.select_dtypes(include=['datetime64']).columns
            for col in date_cols:
                d_min, d_max = filtered_df[col].min(), filtered_df[col].max()
                if pd.notnull(d_min) and pd.notnull(d_max):
                    d_rng = st.date_input(f"Dates: {col}", [d_min.date(), d_max.date()])
                    if len(d_rng) == 2:
                        filtered_df = filtered_df[(filtered_df[col].dt.date >= d_rng[0]) & (filtered_df[col].dt.date <= d_rng[1])]

            if st.button("♻️ Reset All Filters"):
                st.rerun()
                
    return filtered_df

def extract_insights(df, df_original):
    insights = []
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    # 1. Extremes & Averages
    if len(num_cols) > 0:
        for col in num_cols:
            avg = df[col].mean()
            med = df[col].median()
            insights.append(f"**{col}**: Average is {avg:.2f}, Median is {med:.2f}. Range: {df[col].min():.2f} to {df[col].max():.2f}.")
    
    # 2. Top Categories
    if len(cat_cols) > 0:
        for col in cat_cols:
            top = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
            insights.append(f"**{col}**: Most frequent value is '{top}'.")
            
    # 3. Correlations
    if len(num_cols) > 1:
        corr_matrix = df[num_cols].corr()
        for i in range(len(num_cols)):
            for j in range(i+1, len(num_cols)):
                c = corr_matrix.iloc[i, j]
                if abs(c) > 0.7:
                    strength = "strong" if abs(c) > 0.8 else "moderate"
                    direction = "positive" if c > 0 else "negative"
                    insights.append(f"📈 **{num_cols[i]}** & **{num_cols[j]}** show a {strength} {direction} correlation ({c:.2f}).")
    
    return insights

# --- Sidebar Controls ---
with st.sidebar:
    st.title("⚙️ Global Controls")
    uploaded_file = st.file_uploader("Upload dataset (CSV/XLSX)", type=["csv", "xlsx"])
    if uploaded_file:
        if st.button("🔄 Reset to Original"):
            st.session_state.df_cleaned = st.session_state.df.copy()
            st.session_state.imputed_mask = None
            st.rerun()

    st.markdown("---")
    if st.session_state.df is not None:
        st.subheader("🧹 Bulk Operations")
        if st.button("Auto-Prune Empty Cols"):
            st.session_state.df_cleaned, removed = auto_clean_nan_columns(st.session_state.df_cleaned)
            st.success(f"Removed {len(removed)} empty columns.")
        
        with st.expander("🗑️ Duplicate Management"):
            df_temp = st.session_state.df_cleaned.copy()
            cols_for_dup = st.multiselect("Select columns for uniqueness", df_temp.columns.tolist(), default=df_temp.columns.tolist())
            if cols_for_dup:
                dups = df_temp[df_temp.duplicated(subset=cols_for_dup, keep=False)]
                st.info(f"Detected {len(dups)} duplicate rows.")
                if not dups.empty and st.button("✅ Remove Duplicates"):
                    st.session_state.df_cleaned = df_temp.drop_duplicates(subset=cols_for_dup, keep="first")
                    st.rerun()

# --- Main Dashboard ---
c1, c2 = st.columns([1, 6])
with c1:
    st.image("https://imgs.search.brave.com/osQStUYtm_ZhDc8hAUB9lEEqjoA8WCXO-k4h70PrOXA/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9zdGF0/aWMuc3RhcnR1cHRh/bGt5LmNvbS8yMDIx/LzAzL2NvZGluZy1u/aW5qYXMtc3RhcnR1/cHRhbGt5LTEuanBn", width=100)
with c2:
    st.title("🚀 DataClean Pro")
    st.markdown("Advanced Preprocessing & Insight Engine")

if uploaded_file:
    if st.session_state.df is None or (uploaded_file.name != st.session_state.get('last_file')):
        raw_df = load_data(uploaded_file)
        st.session_state.df, _ = auto_clean_nan_columns(raw_df)
        st.session_state.df_cleaned = st.session_state.df.copy()
        st.session_state.last_file = uploaded_file.name

    df_raw, df_clean = st.session_state.df, st.session_state.df_cleaned
    
    # --- Global Filter Engine ---
    df_filtered = apply_filters(df_clean)
    df = df_filtered # Critical: All downstream tools now use the filtered view

    # --- Metrics Panel ---
    st.subheader("📊 Dataset Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", len(df), f"{len(df) - len(df_raw)} vs Initial")
    m2.metric("Columns", df.shape[1], f"{df.shape[1] - df_raw.shape[1]} Change")
    m3.metric("Missing Cells", df.isna().sum().sum())
    m4.metric("Duplicates", df.duplicated().sum())

    tabs = st.tabs(["📋 Understanding", "🔍 Missing Values", "🧹 Cleaning", "🛡️ Outliers", "📊 Intelligence", "💡 Insights", "📥 Export"])

    with tabs[0]: # Understanding
        st.subheader("📊 Column Metadata & Health")
        meta = get_column_metadata(df_clean)
        for idx, row in meta.iterrows():
            with st.container():
                h = 100 - float(row['Missing %'].strip('%'))
                st.caption(f"**{row['Column']}** Health")
                st.progress(h / 100)
        st.dataframe(meta, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            with st.expander("📝 Raw Data Snapshot (Original)"):
                st.dataframe(df_raw.head(10), use_container_width=True)
        with c2:
            with st.expander("✨ Cleaned Data Snapshot (Current)"):
                st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        st.subheader("Statistical Deep Dive")
        st.write(df.describe(include='all').T)

    with tabs[1]: # Missing Values
        heatmap = plot_missing_heatmap(df_clean)
        if heatmap: st.plotly_chart(heatmap, use_container_width=True)
        else: st.success("No missing values!")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Drop Strategy")
            axis = st.radio("By:", ["Rows", "Cols"])
            thresh = st.slider("Min Non-Null %", 0, 100, 70)
            if st.button("Drop Sparsity"):
                t = int((thresh/100.0) * (len(df_clean) if axis=="Cols" else len(df_clean.columns)))
                st.session_state.df_cleaned = df_clean.dropna(axis=1 if axis=="Cols" else 0, thresh=t)
                st.rerun()
        with c2:
            st.subheader("Imputation")
            met = st.selectbox("Strategy", ["Mean", "Median", "Mode", "F-Fill", "B-Fill"])
            if st.button("Apply Imputation"):
                st.session_state.imputed_mask = df_clean.isna()
                temp = df_clean.copy()
                if met == "Mean":
                    num = temp.select_dtypes(include=[np.number]).columns
                    temp[num] = temp[num].fillna(temp[num].mean())
                elif met == "Median":
                    num = temp.select_dtypes(include=[np.number]).columns
                    temp[num] = temp[num].fillna(temp[num].median())
                elif met == "Mode": temp = temp.fillna(temp.mode().iloc[0])
                elif met == "F-Fill": temp = temp.ffill()
                elif met == "B-Fill": temp = temp.bfill()
                st.session_state.df_cleaned = temp
                st.rerun()

    with tabs[2]: # Cleaning
        st.subheader("Advanced Text & Schema Cleaning")
        c1, c2 = st.columns(2)
        with c1:
            with st.expander("🔡 Text Normalization", expanded=True):
                obj = df_clean.select_dtypes(include=['object']).columns.tolist()
                if obj:
                    sel = st.multiselect("Columns", obj)
                    act = st.radio("Action", ["Trim", "Lower", "Upper", "Title"])
                    if st.button("Standardize Text"):
                        for c in sel:
                            if act == "Trim": st.session_state.df_cleaned[c] = df_clean[c].str.strip()
                            elif act == "Lower": st.session_state.df_cleaned[c] = df_clean[c].str.lower()
                            elif act == "Upper": st.session_state.df_cleaned[c] = df_clean[c].str.upper()
                            elif act == "Title": st.session_state.df_cleaned[c] = df_clean[c].str.title()
                        st.rerun()
            with st.expander("🏷️ Rename & Type"):
                col = st.selectbox("Choose Column", df_clean.columns)
                opt = st.radio("Change", ["Rename", "Convert Type"])
                if opt == "Rename":
                    new = st.text_input("New Name")
                    if st.button("Rename Col"): 
                        st.session_state.df_cleaned = df_clean.rename(columns={col: new})
                        st.rerun()
                else:
                    t = st.selectbox("To Type", ["Float", "Int", "String", "Date"])
                    if st.button("Convert"):
                        try:
                            if t == "Float": st.session_state.df_cleaned[col] = pd.to_numeric(df_clean[col])
                            elif t == "Int": st.session_state.df_cleaned[col] = pd.to_numeric(df_clean[col]).astype(int)
                            elif t == "String": st.session_state.df_cleaned[col] = df_clean[col].astype(str)
                            elif t == "Date": st.session_state.df_cleaned[col] = pd.to_datetime(df_clean[col])
                            st.rerun()
                        except: st.error("Incompatible data.")
        with c2:
            with st.expander("🛑 Domain Logic Fixes", expanded=True):
                num = df_clean.select_dtypes(include=[np.number]).columns.tolist()
                if num:
                    target = st.selectbox("Audit Column", num)
                    logic = st.radio("Issue", ["Negative Values", "Future Dates (if Date)"])
                    if logic == "Negative Values":
                        fix = st.selectbox("Fix", ["Set to 0", "Remove Row"])
                        if st.button("Fix Negatives"):
                            mask = df_clean[target] < 0
                            if fix == "Set to 0": st.session_state.df_cleaned.loc[mask, target] = 0
                            else: st.session_state.df_cleaned = df_clean[~mask]
                            st.rerun()
                dates = df_clean.select_dtypes(include=['datetime64']).columns.tolist()
                if dates:
                    d_target = st.selectbox("Audit Date", dates)
                    if st.button("Cap to Today"):
                        mask = df_clean[d_target] > pd.Timestamp.now()
                        st.session_state.df_cleaned.loc[mask, d_target] = pd.Timestamp.now()
                        st.rerun()

    with tabs[3]: # Outliers
        num = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if num:
            s_col = st.selectbox("Audit Feature", num)
            mode = st.radio("Technique", ["IQR", "Z-Score", "Percentile"])
            if mode == "IQR":
                k = st.slider("IQR Factor", 1.0, 3.0, 1.5)
                q1, q3 = df_clean[s_col].quantile(0.25), df_clean[s_col].quantile(0.75)
                iqr = q3 - q1
                lb, ub = q1 - k*iqr, q3 + k*iqr
            elif mode == "Z-Score":
                z = st.slider("Threshold", 2.0, 5.0, 3.0)
                lb, ub = df_clean[s_col].mean()-z*df_clean[s_col].std(), df_clean[s_col].mean()+z*df_clean[s_col].std()
            else:
                p = st.slider("Percentile", 0.0, 100.0, (1.0, 99.0))
                lb, ub = df_clean[s_col].quantile(p[0]/100), df_clean[s_col].quantile(p[1]/100)
            
            fig = px.box(df_clean, y=s_col, points="all", title=f"Outlier Audit: {s_col}")
            fig.add_hline(y=lb, line_dash="dash", line_color="red")
            fig.add_hline(y=ub, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
            
            if st.button("Winsorize (Cap)"):
                st.session_state.df_cleaned[s_col] = df_clean[s_col].clip(lb, ub)
                st.rerun()
            if st.button("Remove Selected Outliers"):
                st.session_state.df_cleaned = df_clean[(df_clean[s_col] >= lb) & (df_clean[s_col] <= ub)]
                st.rerun()

    with tabs[4]: # Intelligence
        st.subheader("📊 Ultimate Visual Intelligence")
        
        mode = st.selectbox("Select Visual Category", [
            "📈 Univariate Distributions",
            "🔗 Relationship & Correlation",
            "🕰️ Time-Series & Trends",
            "🛡️ Data Quality & Anomaly Audit",
            "🧠 Multivariate & Expert Projections",
            "🌍 Geospatial Intelligence"
        ])

        num_c = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_c = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_c = df.select_dtypes(include=['datetime64']).columns.tolist()

        if mode == "📈 Univariate Distributions":
            c1, c2 = st.columns([1, 3])
            with c1:
                chart = st.selectbox("Chart Type", [
                    "Histogram", "Density Plot (KDE)", "Box Plot", "Violin Plot", 
                    "Rug Plot", "Strip Plot", "Bar (Cat)", "Pie/Donut"
                ])
                feat = st.selectbox("Feature", num_c if chart not in ["Bar (Cat)", "Pie/Donut"] else cat_c)
            
            with c2:
                if chart == "Histogram":
                    st.plotly_chart(px.histogram(df, x=feat, marginal="box", title=f"Histogram of {feat}"), use_container_width=True)
                elif chart == "Density Plot (KDE)":
                    import plotly.figure_factory as ff
                    try:
                        valid_data = df[feat].dropna()
                        if not valid_data.empty:
                            fig = ff.create_distplot([valid_data], [feat], show_hist=False)
                            st.plotly_chart(fig, use_container_width=True)
                        else: st.warning("No valid data for KDE.")
                    except Exception as e: st.error(f"KDE Error: {e}")
                elif chart == "Box Plot":
                    st.plotly_chart(px.box(df, y=feat, points="all", title=f"Box Plot of {feat}"), use_container_width=True)
                elif chart == "Violin Plot":
                    st.plotly_chart(px.violin(df, y=feat, box=True, points="all", title=f"Violin Plot of {feat}"), use_container_width=True)
                elif chart == "Rug Plot":
                    st.plotly_chart(px.histogram(df, x=feat, marginal="rug", title=f"Rug & Hist of {feat}"), use_container_width=True)
                elif chart == "Strip Plot":
                    st.plotly_chart(px.strip(df, x=feat, title=f"Strip Plot of {feat}"), use_container_width=True)
                elif chart == "Bar (Cat)":
                    horiz = st.checkbox("Horizontal Orientation")
                    counts = df[feat].value_counts().reset_index()
                    counts.columns = [feat, "Count"]
                    fig = px.bar(counts, x="Count" if horiz else feat, y=feat if horiz else "Count", orientation='h' if horiz else 'v', title=f"Frequency: {feat}")
                    st.plotly_chart(fig, use_container_width=True)
                elif chart == "Pie/Donut":
                    hole = st.slider("Donut Hole", 0.0, 0.8, 0.4)
                    st.plotly_chart(px.pie(df, names=feat, hole=hole, title=f"Proportions: {feat}"), use_container_width=True)

        elif mode == "🔗 Relationship & Correlation":
            sub = st.selectbox("Sub-type", ["Numeric vs Numeric", "Numeric vs Categorical", "Categorical vs Categorical"])
            if sub == "Numeric vs Numeric":
                c1, c2, c3 = st.columns(3)
                x, y = c1.selectbox("X Axis", num_c), c2.selectbox("Y Axis", num_c)
                type = c3.selectbox("Visual Type", ["Scatter", "Bubble", "Regression", "Density 2D", "Hexbin"])
                if type == "Scatter":
                    st.plotly_chart(px.scatter(df, x=x, y=y, title=f"{x} vs {y}"), use_container_width=True)
                elif type == "Bubble":
                    z = st.selectbox("Size/Color By", num_c)
                    st.plotly_chart(px.scatter(df, x=x, y=y, size=z, color=z, title=f"Bubble Chart: {x} vs {y}"), use_container_width=True)
                elif type == "Regression":
                    st.plotly_chart(px.scatter(df, x=x, y=y, trendline="ols", title=f"Regression: {x} vs {y}"), use_container_width=True)
                elif type == "Density 2D":
                    st.plotly_chart(px.density_heatmap(df, x=x, y=y, marginal_x="histogram", marginal_y="histogram", title=f"Density: {x} vs {y}"), use_container_width=True)
                elif type == "Hexbin":
                    st.plotly_chart(px.density_contour(df, x=x, y=y, title=f"Hex-Contour: {x} vs {y}"), use_container_width=True)

            elif sub == "Numeric vs Categorical":
                x, y = st.selectbox("Category Group", cat_c), st.selectbox("Numeric Metric", num_c)
                v = st.radio("Style", ["Box Plot", "Violin Plot", "Grouped Bar", "Strip Plot"], horizontal=True)
                if v == "Box Plot": st.plotly_chart(px.box(df, x=x, y=y, color=x, title=f"{y} by {x}"), use_container_width=True)
                elif v == "Violin Plot": st.plotly_chart(px.violin(df, x=x, y=y, color=x, box=True, title=f"{y} split by {x}"), use_container_width=True)
                elif v == "Strip Plot": st.plotly_chart(px.strip(df, x=x, y=y, color=x, title=f"Strip: {y} by {x}"), use_container_width=True)
                else: st.plotly_chart(px.bar(df, x=x, y=y, color=x, barmode="group", title=f"Grouped Bar: {y} vs {x}"), use_container_width=True)
            
            elif sub == "Categorical vs Categorical":
                x, y = st.selectbox("Category A", cat_c), st.selectbox("Category B", cat_c)
                st.plotly_chart(px.density_heatmap(df, x=x, y=y, text_auto=True, title=f"Frequency Heatmap: {x} vs {y}"), use_container_width=True)

        elif mode == "🕰️ Time-Series & Trends":
            if date_c:
                t_col = st.selectbox("Timeline Axis", date_c)
                val = st.selectbox("Metric", num_c)
                chart = st.radio("Trend Style", ["Line Trend", "Area Flow", "Rolling Average", "Stacked Area"], horizontal=True)
                
                ts_base = df.set_index(t_col)[val].sort_index()
                if chart == "Rolling Average":
                    win = st.slider("Window Size", 2, 60, 7)
                    ts_plot = ts_base.rolling(window=win).mean().reset_index()
                else: ts_plot = ts_base.reset_index()
                
                if chart == "Area Flow": fig = px.area(ts_plot, x=t_col, y=val, title=f"Area Trend: {val}")
                elif chart == "Stacked Area" and cat_c:
                    group = st.selectbox("Split By", cat_c)
                    ts_stacked = df.groupby([t_col, group])[val].mean().reset_index()
                    fig = px.area(ts_stacked, x=t_col, y=val, color=group, title=f"Stacked Trend: {val}")
                else: fig = px.line(ts_plot, x=t_col, y=val, title=f"Line Trend: {val}", markers=True)
                st.plotly_chart(fig, use_container_width=True)
            else: st.warning("No time data.")

        elif mode == "🛡️ Data Quality & Anomaly Audit":
            audit = st.radio("Drill-down", ["Null Pattern Matrix", "Missing Bar (%)", "Outlier Highlight", "Z-Score Dist"])
            if audit == "Null Pattern Matrix":
                st.plotly_chart(plot_missing_heatmap(df), use_container_width=True)
            elif audit == "Missing Bar (%)":
                null_pct = (df.isna().sum() / len(df) * 100).reset_index()
                null_pct.columns = ["Column", "Null %"]
                st.plotly_chart(px.bar(null_pct, x="Column", y="Null %", title="Missingness per Feature"), use_container_width=True)
            elif audit == "Outlier Highlight":
                col = st.selectbox("Audit Feature", num_c)
                st.plotly_chart(px.box(df, y=col, points="outliers", title=f"Anomaly Audit: {col}"), use_container_width=True)
            else:
                col = st.selectbox("Feature for Z-Score", num_c)
                z_vals = np.abs(stats.zscore(df[col].dropna()))
                st.plotly_chart(px.histogram(z_vals, nbins=50, title=f"Z-Score Spread: {col}"), use_container_width=True)

        elif mode == "🧠 Multivariate & Expert Projections":
            adv = st.selectbox("Select Technique", ["Parallel Coordinates", "Scatter Matrix", "PCA Projection (2D)", "PCA Projection (3D)", "Correlation Matrix", "Radar Chart", "Sankey Diagram"])
            if adv == "Parallel Coordinates" and len(num_c) > 1:
                st.plotly_chart(px.parallel_coordinates(df, dimensions=num_c[:6]), use_container_width=True)
            elif adv == "Scatter Matrix" and num_c:
                st.plotly_chart(px.scatter_matrix(df, dimensions=num_c[:4]), use_container_width=True)
            elif "PCA" in adv and len(num_c) >= 3:
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                try:
                    X = df[num_c].dropna()
                    if len(X) > 5:
                        X_scaled = StandardScaler().fit_transform(X)
                        pca_obj = PCA(n_components=3 if "3D" in adv else 2)
                        res = pca_obj.fit_transform(X_scaled)
                        if "3D" in adv: fig = px.scatter_3d(x=res[:,0], y=res[:,1], z=res[:,2], title="3D PCA Dimensional Reduction")
                        else: fig = px.scatter(x=res[:,0], y=res[:,1], title="2D PCA Latent Space")
                        st.plotly_chart(fig, use_container_width=True)
                    else: st.warning("Not enough rows for PCA.")
                except Exception as e: st.error(f"PCA Error: {e}")
            elif adv == "Correlation Matrix" and len(num_c) > 1:
                st.plotly_chart(px.imshow(df[num_c].corr(), text_auto=True, color_continuous_scale="RdBu_r"), use_container_width=True)
            elif adv == "Radar Chart" and num_c:
                import plotly.graph_objects as go
                categories = num_c[:6]
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=df[categories].mean().values, theta=categories, fill='toself', name='Dataset Average'))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, title="Feature Radar")
                st.plotly_chart(fig, use_container_width=True)
            elif adv == "Sankey Diagram" and len(cat_c) >= 2:
                s_cols = st.multiselect("Dimensions", cat_c, default=cat_c[:2])
                if len(s_cols) >= 2:
                    df_s = df.groupby(s_cols).size().reset_index(name='count')
                    nodes = pd.concat([df_s[c] for c in s_cols]).unique()
                    node_map = {n: i for i, n in enumerate(nodes)}
                    sources, targets, values = [], [], []
                    for i in range(len(s_cols)-1):
                        for _, row in df_s.iterrows():
                            sources.append(node_map[row[s_cols[i]]])
                            targets.append(node_map[row[s_cols[i+1]]])
                            values.append(row['count'])
                    fig = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=20, label=nodes),
                                                 link=dict(source=sources, target=targets, value=values))])
                    st.plotly_chart(fig, use_container_width=True)

        elif mode == "🌍 Geospatial Intelligence":
            geo_cols = [c for c in df.columns if any(k in c.lower() for k in ['lat', 'lon', 'latitude', 'longitude'])]
            if len(geo_cols) >= 2:
                lat_col = next(c for c in geo_cols if 'lat' in c.lower())
                lon_col = next(c for c in geo_cols if 'lon' in c.lower())
                st.plotly_chart(px.scatter_mapbox(df, lat=lat_col, lon=lon_col, zoom=3, mapbox_style="carto-positron", title="Geospatial Hub"), use_container_width=True)
            else: st.info("No spatial data (Lat/Lon) found.")

    with tabs[5]: # Insights
        st.subheader("💡 Multi-Dimensional Automated Insights")
        
        # KPI Cards for filtered segment
        k1, k2, k3, k4 = st.columns(4)
        diff = len(df)-len(df_clean)
        k1.metric("Segment Rows", len(df), delta=str(diff))
        if num_c:
            target_kpi = st.selectbox("Select KPI Feature", num_c, key="kpi_feat")
            k2.metric(f"Avg {target_kpi}", f"{df[target_kpi].mean():.2f}")
            k3.metric(f"Median {target_kpi}", f"{df[target_kpi].median():.2f}")
            k4.metric(f"Std Dev", f"{df[target_kpi].std():.2f}")
        
        st.markdown("---")
        insights = extract_insights(df, df_original=df_clean)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.write("### 🧠 Automated Findings")
            for i in insights: st.markdown(f"> {i}")
        
        with c2:
            st.write("### 🎯 Segment Analysis")
            if num_c:
                comp_data = [{"Feature": c, "Segment Mean": df[c].mean(), "Overall Mean": df_clean[c].mean()} for c in num_c]
                st.plotly_chart(px.bar(pd.DataFrame(comp_data), x="Feature", y=["Segment Mean", "Overall Mean"], barmode="group"), use_container_width=True)

    with tabs[6]: # Export
        st.subheader("📥 Finalize and Download")
        st.dataframe(df.head(100), use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Cleaned Data (CSV)", data=csv, file_name="cleaned_dataset.csv", mime="text/csv")
else:
    st.info("👋 Welcome! Please upload a CSV or XLSX file to get started.")
    st.image("https://imgs.search.brave.com/osQStUYtm_ZhDc8hAUB9lEEqjoA8WCXO-k4h70PrOXA/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9zdGF0/aWMuc3RhcnR1cHRh/bGt5LmNvbS8yMDIx/LzAzL2NvZGluZy1u/aW5qYXMtc3RhcnR1/cHRhbGt5LTEuanBn", caption="DataClean Pro - Powered by Coding Ninjas")

st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 20px;">
        <p>Made with ❤️ by <a href="https://sourishdeyportfolio.vercel.app/" target="_blank" style="color: #2e7bcf; text-decoration: none; font-weight: bold;">Sourish Dey</a></p>
    </div>
""", unsafe_allow_html=True)
