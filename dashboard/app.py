"""
Telco Customer Churn Analytics Dashboard
=========================================
Streamlit dashboard for visualizing churn analysis results.
Run with:  streamlit run dashboard/app.py  (from project root)
"""

import pickle
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROC = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "reports" / "figures"

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        color: #718096;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-card-churn {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    }
    .metric-card-churn .metric-value,
    .metric-card-churn .metric-label {
        color: white !important;
    }

    .cluster-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }

    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }

    .insight-box {
        background: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    div[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_raw_data():
    return pd.read_csv(DATA_RAW / "WA_Fn-UseC_-Telco-Customer-Churn.csv")


@st.cache_data
def load_processed():
    data = {}
    files = {
        "X_train": "X_train.pkl",
        "X_test": "X_test.pkl",
        "y_train": "y_train.pkl",
        "y_test": "y_test.pkl",
        "df_cleaned": "df_cleaned.pkl",
        "feature_names": "feature_names.pkl",
    }
    for key, fname in files.items():
        path = DATA_PROC / fname
        if path.exists():
            if fname == "feature_names.pkl":
                with open(path, "rb") as f:
                    data[key] = pickle.load(f)
            else:
                data[key] = pd.read_pickle(path)
    # Optional files
    for key, fname in [("df_with_clusters", "df_with_clusters.pkl"),
                       ("cluster_labels", "cluster_labels.pkl"),
                       ("model_metrics", "model_metrics.pkl"),
                       ("feature_importance", "feature_importance.pkl")]:
        path = DATA_PROC / fname
        if path.exists():
            if fname == "cluster_labels.pkl":
                with open(path, "rb") as f:
                    data[key] = pickle.load(f)
            else:
                data[key] = pd.read_pickle(path)
    return data


@st.cache_resource
def load_models():
    models = {}
    for key, fname in [("scaler", "scaler.pkl"),
                       ("kmeans", "kmeans.pkl"),
                       ("logreg_best", "logreg_best.pkl"),
                       ("naive_bayes_best", "naive_bayes_best.pkl"),
                       ("all_models", "all_models.pkl")]:
        path = MODEL_DIR / fname
        if path.exists():
            models[key] = joblib.load(path)
    return models


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📊 Telco Churn Analytics")
    st.markdown("---")
    st.markdown("""
    **Tugas Besar Penambangan Data**  
    Prediksi & Segmentasi Customer Churn  
    menggunakan K-Means, Logistic Regression, dan Naïve Bayes.
    """)
    st.markdown("---")
    page = st.radio(
        "📌 Navigasi",
        [
            "📊 Overview",
            "🔍 Data Exploration",
            "🎯 Segmentasi (K-Means)",
            "🤖 Prediksi Churn",
            "📈 Feature Importance",
            "🔮 Prediksi Individual",
        ],
    )
    st.markdown("---")
    st.markdown("*CRISP-DM Methodology*")


# ══════════════════════════════════════════════════════════════════════════════
#  Load Data
# ══════════════════════════════════════════════════════════════════════════════

try:
    df_raw = load_raw_data()
except Exception as e:
    st.error(f"Gagal memuat data raw: {e}")
    st.stop()

proc = load_processed()
mdl = load_models()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1: Overview
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Overview":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Telco Customer Churn Analytics</h1>
        <p>Dashboard analisis prediksi dan segmentasi churn pelanggan telekomunikasi</p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    total = len(df_raw)
    churn_count = (df_raw["Churn"] == "Yes").sum()
    churn_rate = churn_count / total * 100
    avg_tenure = df_raw["tenure"].mean()
    avg_monthly = df_raw["MonthlyCharges"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total:,}</div>
            <div class="metric-label">Total Pelanggan</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card metric-card-churn">
            <div class="metric-value">{churn_rate:.1f}%</div>
            <div class="metric-label">Churn Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_tenure:.1f}</div>
            <div class="metric-label">Rata-rata Tenure (bulan)</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${avg_monthly:.2f}</div>
            <div class="metric-label">Rata-rata Biaya Bulanan</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        churn_dist = df_raw["Churn"].value_counts()
        fig_pie = px.pie(
            values=churn_dist.values,
            names=churn_dist.index,
            title="Distribusi Churn",
            color_discrete_sequence=["#667eea", "#ff6b6b"],
            hole=0.4,
        )
        fig_pie.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        contract_churn = df_raw.groupby("Contract")["Churn"].apply(
            lambda x: (x == "Yes").mean() * 100
        ).reset_index()
        contract_churn.columns = ["Contract", "Churn Rate (%)"]
        fig_bar = px.bar(
            contract_churn,
            x="Contract",
            y="Churn Rate (%)",
            title="Churn Rate per Tipe Kontrak",
            color="Contract",
            color_discrete_sequence=["#ff6b6b", "#feca57", "#48dbfb"],
        )
        fig_bar.update_layout(template="plotly_white", height=400, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Quick insights
    st.markdown("""
    <div class="insight-box">
        <strong>💡 Key Insight:</strong> Pelanggan dengan kontrak Month-to-month memiliki churn rate 
        tertinggi. Kontrak jangka panjang (One year / Two year) secara signifikan mengurangi risiko churn.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2: Data Exploration
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Data Exploration":
    st.markdown('<div class="section-header">🔍 Data Exploration</div>', unsafe_allow_html=True)

    # Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        contract_filter = st.multiselect(
            "Filter by Contract", df_raw["Contract"].unique(), default=df_raw["Contract"].unique()
        )
    with col_f2:
        internet_filter = st.multiselect(
            "Filter by Internet Service", df_raw["InternetService"].unique(),
            default=df_raw["InternetService"].unique()
        )

    filtered = df_raw[
        (df_raw["Contract"].isin(contract_filter)) &
        (df_raw["InternetService"].isin(internet_filter))
    ]
    st.markdown(f"**{len(filtered):,}** pelanggan sesuai filter")

    col1, col2 = st.columns(2)

    with col1:
        fig_tenure = px.histogram(
            filtered, x="tenure", color="Churn",
            title="Distribusi Tenure",
            color_discrete_map={"Yes": "#ff6b6b", "No": "#667eea"},
            barmode="overlay", opacity=0.7, nbins=30,
        )
        fig_tenure.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_tenure, use_container_width=True)

    with col2:
        fig_monthly = px.histogram(
            filtered, x="MonthlyCharges", color="Churn",
            title="Distribusi Monthly Charges",
            color_discrete_map={"Yes": "#ff6b6b", "No": "#667eea"},
            barmode="overlay", opacity=0.7, nbins=30,
        )
        fig_monthly.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_monthly, use_container_width=True)

    # Churn rate by categories
    st.markdown("### Churn Rate per Kategori")
    cat_feature = st.selectbox(
        "Pilih fitur:", ["InternetService", "Contract", "PaymentMethod",
                         "OnlineSecurity", "TechSupport", "StreamingTV", "gender",
                         "SeniorCitizen", "Partner", "Dependents"]
    )
    cat_churn = filtered.groupby(cat_feature)["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).reset_index()
    cat_churn.columns = [cat_feature, "Churn Rate (%)"]
    fig_cat = px.bar(
        cat_churn, x=cat_feature, y="Churn Rate (%)",
        title=f"Churn Rate by {cat_feature}",
        color="Churn Rate (%)", color_continuous_scale="RdYlGn_r",
    )
    fig_cat.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig_cat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3: Segmentasi (K-Means)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🎯 Segmentasi (K-Means)":
    st.markdown('<div class="section-header">🎯 Segmentasi Pelanggan (K-Means Clustering)</div>',
                unsafe_allow_html=True)

    if "df_with_clusters" not in proc or "kmeans" not in mdl:
        st.warning("⚠️ Data clustering belum tersedia. Jalankan `src/03_clustering_kmeans.py` terlebih dahulu.")
        st.stop()

    df_clust = proc["df_with_clusters"]
    km_model = mdl["kmeans"]
    n_clusters = len(df_clust["Cluster"].unique())

    # Cluster sizes
    st.markdown("### Distribusi Cluster")
    col1, col2 = st.columns(2)

    with col1:
        cluster_sizes = df_clust["Cluster"].value_counts().sort_index().reset_index()
        cluster_sizes.columns = ["Cluster", "Count"]
        cluster_sizes["Cluster"] = cluster_sizes["Cluster"].astype(str)
        fig_sizes = px.pie(
            cluster_sizes, values="Count", names="Cluster",
            title="Ukuran Cluster",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.35,
        )
        fig_sizes.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_sizes, use_container_width=True)

    with col2:
        churn_per_cluster = df_clust.groupby("Cluster")["Churn"].mean().reset_index()
        churn_per_cluster.columns = ["Cluster", "Churn Rate"]
        churn_per_cluster["Churn Rate (%)"] = churn_per_cluster["Churn Rate"] * 100
        churn_per_cluster["Cluster"] = churn_per_cluster["Cluster"].astype(str)
        fig_churn_cl = px.bar(
            churn_per_cluster, x="Cluster", y="Churn Rate (%)",
            title="Churn Rate per Cluster",
            color="Churn Rate (%)", color_continuous_scale="RdYlGn_r",
            text_auto=".1f",
        )
        fig_churn_cl.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_churn_cl, use_container_width=True)

    # PCA Visualization
    st.markdown("### Visualisasi PCA 2D")
    if "X_clustering" in proc:
        X_cl = proc.get("X_clustering", proc.get("df_cleaned"))
        if "Churn" in X_cl.columns:
            X_pca_data = X_cl.drop(columns=["Churn"])
        else:
            X_pca_data = X_cl
        if "Cluster" in X_pca_data.columns:
            X_pca_data = X_pca_data.drop(columns=["Cluster"])

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_pca_data)
        pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
        pca_df["Cluster"] = df_clust["Cluster"].astype(str).values

        fig_pca = px.scatter(
            pca_df, x="PC1", y="PC2", color="Cluster",
            title=f"PCA 2D — {n_clusters} Clusters",
            color_discrete_sequence=px.colors.qualitative.Set2,
            opacity=0.6,
        )
        fig_pca.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_pca, use_container_width=True)

    # Cluster profiles
    st.markdown("### Profil Cluster")
    profile_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    available_cols = [c for c in profile_cols if c in df_clust.columns]
    if available_cols:
        profile = df_clust.groupby("Cluster")[available_cols].mean().round(3)
        st.dataframe(profile.style.background_gradient(cmap="YlOrRd"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4: Prediksi Churn
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Prediksi Churn":
    st.markdown('<div class="section-header">🤖 Hasil Prediksi Churn</div>', unsafe_allow_html=True)

    if "model_metrics" not in proc or "all_models" not in mdl:
        st.warning("⚠️ Model belum tersedia. Jalankan `src/04_classification.py` terlebih dahulu.")
        st.stop()

    metrics_df = proc["model_metrics"]
    all_models = mdl["all_models"]
    X_test = proc.get("X_test")
    y_test = proc.get("y_test")

    # Model comparison table
    st.markdown("### 📋 Perbandingan Model")
    st.dataframe(
        metrics_df.style.highlight_max(
            subset=[c for c in metrics_df.columns if c != "Model"],
            color="#a8e6cf"
        ).format({c: "{:.4f}" for c in metrics_df.columns if c != "Model"}),
        use_container_width=True,
    )

    # Metrics bar chart
    metric_cols = [c for c in metrics_df.columns if c != "Model"]
    fig_metrics = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, (_, row) in enumerate(metrics_df.iterrows()):
        fig_metrics.add_trace(go.Bar(
            name=row["Model"],
            x=metric_cols,
            y=[row[c] for c in metric_cols],
            marker_color=colors[i % len(colors)],
        ))
    fig_metrics.update_layout(
        barmode="group", template="plotly_white",
        title="Perbandingan Metrik Model", height=450,
        yaxis_range=[0, 1.05],
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    if X_test is not None and y_test is not None:
        col1, col2 = st.columns(2)

        # Confusion Matrix
        with col1:
            st.markdown("### Confusion Matrix")
            selected_model = st.selectbox("Pilih model:", list(all_models.keys()))
            model = all_models[selected_model]
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm, text_auto=True,
                x=["No Churn", "Churn"], y=["No Churn", "Churn"],
                color_continuous_scale="Blues",
                title=f"Confusion Matrix — {selected_model}",
                labels=dict(x="Predicted", y="Actual"),
            )
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm, use_container_width=True)

        # ROC Curves
        with col2:
            st.markdown("### ROC Curve")
            fig_roc = go.Figure()
            for i, (name, model) in enumerate(all_models.items()):
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, "decision_function"):
                    y_score = model.decision_function(X_test)
                else:
                    continue
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc_val = auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{name} (AUC={roc_auc_val:.3f})",
                    line=dict(color=colors[i % len(colors)], width=2),
                ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Random", line=dict(dash="dash", color="gray"),
            ))
            fig_roc.update_layout(
                template="plotly_white", height=400,
                title="ROC Curve Comparison",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
            )
            st.plotly_chart(fig_roc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5: Feature Importance
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Feature Importance":
    st.markdown('<div class="section-header">📈 Feature Importance</div>', unsafe_allow_html=True)

    if "feature_importance" not in proc:
        st.warning("⚠️ Data feature importance belum tersedia. Jalankan classification terlebih dahulu.")
        st.stop()

    fi_df = proc["feature_importance"]
    top15 = fi_df.sort_values("importance", ascending=False).head(15)

    fig_fi = px.bar(
        top15.sort_values("importance", ascending=True),
        x="importance", y="feature", orientation="h",
        title="Top 15 Feature Importance (Logistic Regression Coefficients)",
        color="importance", color_continuous_scale="Viridis",
    )
    fig_fi.update_layout(template="plotly_white", height=550, yaxis_title="", xaxis_title="Importance")
    st.plotly_chart(fig_fi, use_container_width=True)

    # Interpretation
    top3 = top15.head(3)["feature"].tolist()
    st.markdown(f"""
    <div class="insight-box">
        <strong>💡 Interpretasi:</strong><br>
        Fitur dengan pengaruh terbesar terhadap churn adalah: <strong>{', '.join(top3)}</strong>.<br><br>
        <strong>Rekomendasi Bisnis:</strong><br>
        • Pelanggan dengan kontrak month-to-month memiliki risiko churn tertinggi — tawarkan insentif upgrade kontrak<br>
        • Layanan keamanan online dan dukungan teknis dapat mengurangi churn — promosikan bundling layanan<br>
        • Pelanggan baru (tenure rendah) perlu program onboarding yang kuat dalam 12 bulan pertama
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 6: Prediksi Individual
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Prediksi Individual":
    st.markdown('<div class="section-header">🔮 Prediksi Churn Individual</div>', unsafe_allow_html=True)

    if "logreg_best" not in mdl or "scaler" not in mdl:
        st.warning("⚠️ Model belum tersedia. Jalankan scripts terlebih dahulu.")
        st.stop()

    if "feature_names" not in proc:
        st.warning("⚠️ Feature names belum tersedia.")
        st.stop()

    st.markdown("Masukkan data pelanggan untuk memprediksi kemungkinan churn:")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])

        with col2:
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_sec = st.selectbox("Online Security", ["No", "Yes"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes"])
            device_prot = st.selectbox("Device Protection", ["No", "Yes"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])

        with col3:
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            tenure = st.slider("Tenure (bulan)", 0, 72, 12)
            monthly = st.slider("Monthly Charges ($)", 0.0, 120.0, 50.0, step=0.5)

        total_charges = st.slider("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly), step=10.0)

        submitted = st.form_submit_button("🔮 Prediksi Churn", use_container_width=True)

    if submitted:
        # Build feature vector matching preprocessing
        yes_no = {"No": 0, "Yes": 1}
        input_data = {
            "gender": 0 if gender == "Female" else 1,
            "SeniorCitizen": yes_no[senior],
            "Partner": yes_no[partner],
            "Dependents": yes_no[dependents],
            "PhoneService": yes_no[phone_service],
            "MultipleLines": yes_no[multiple_lines],
            "OnlineSecurity": yes_no[online_sec],
            "OnlineBackup": yes_no[online_backup],
            "DeviceProtection": yes_no[device_prot],
            "TechSupport": yes_no[tech_support],
            "StreamingTV": yes_no[streaming_tv],
            "StreamingMovies": yes_no[streaming_movies],
            "PaperlessBilling": yes_no[paperless],
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total_charges,
        }
        # One-hot encode (drop_first=True)
        # InternetService: base=DSL → Fiber optic, No
        input_data["InternetService_Fiber optic"] = 1 if internet == "Fiber optic" else 0
        input_data["InternetService_No"] = 1 if internet == "No" else 0
        # Contract: base=Month-to-month → One year, Two year
        input_data["Contract_One year"] = 1 if contract == "One year" else 0
        input_data["Contract_Two year"] = 1 if contract == "Two year" else 0
        # PaymentMethod: base=Bank transfer → Credit card, Electronic check, Mailed check
        input_data["PaymentMethod_Credit card (automatic)"] = 1 if payment == "Credit card (automatic)" else 0
        input_data["PaymentMethod_Electronic check"] = 1 if payment == "Electronic check" else 0
        input_data["PaymentMethod_Mailed check"] = 1 if payment == "Mailed check" else 0

        # Create DataFrame with correct feature order
        feature_names = proc["feature_names"]
        input_df = pd.DataFrame([input_data])

        # Ensure all features present
        for feat in feature_names:
            if feat not in input_df.columns:
                input_df[feat] = 0
        input_df = input_df[feature_names]

        # Scale numeric columns
        scaler = mdl["scaler"]
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Predict
        model = mdl["logreg_best"]
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        st.markdown("<br>", unsafe_allow_html=True)

        if prediction == 1:
            st.error(f"⚠️ **Pelanggan diprediksi CHURN** dengan probabilitas **{proba[1]*100:.1f}%**")
        else:
            st.success(f"✅ **Pelanggan diprediksi TIDAK CHURN** dengan probabilitas **{proba[0]*100:.1f}%**")

        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=proba[1] * 100,
            title={"text": "Risiko Churn (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ff6b6b" if prediction == 1 else "#667eea"},
                "steps": [
                    {"range": [0, 30], "color": "#c8e6c9"},
                    {"range": [30, 60], "color": "#fff9c4"},
                    {"range": [60, 100], "color": "#ffcdd2"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color: #718096; font-size: 0.85rem;'>"
    "Tugas Besar Penambangan Data — Telkom University — 2025/2026"
    "</center>",
    unsafe_allow_html=True,
)
