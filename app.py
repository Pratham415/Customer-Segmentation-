import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Customer Segmentation", page_icon="📊", layout="wide")

st.title("📊 Scalable Customer Segmentation Using K-Means and Big Data Analytics")
st.write(
    "Upload a customer dataset (CSV). The system cleans the data, scales it, "
    "applies K-Means clustering, visualizes the segments and generates business insights."
)

# -----------------------------
# 1. LOAD DATA
# -----------------------------
st.sidebar.header("⚙️ Controls")
uploaded = st.sidebar.file_uploader("Upload customer CSV", type=["csv"])

if uploaded is None:
    st.info("👈 Upload your Kaggle customer CSV from the sidebar to begin.")
    st.stop()

# sep=None lets pandas auto-detect comma / tab / semicolon
raw = pd.read_csv(uploaded, sep=None, engine="python")

# -----------------------------
# 2. DATA CLEANING
# -----------------------------
rows_before = len(raw)
df = raw.drop_duplicates().copy()
duplicates_removed = rows_before - len(df)
missing_total = int(df.isnull().sum().sum())

numeric_cols = [
    c for c in df.select_dtypes(include=np.number).columns
    if "id" not in c.lower()
]

if len(numeric_cols) < 2:
    st.error("The dataset needs at least 2 numeric columns for clustering.")
    st.stop()

# -----------------------------
# 3. SIDEBAR: FEATURES + K
# -----------------------------
default_feats = [c for c in numeric_cols if any(k in c.lower() for k in ["income", "spending"])]
if len(default_feats) < 2:
    default_feats = numeric_cols[:3]

features = st.sidebar.multiselect("Select features for clustering", numeric_cols, default=default_feats)
k = st.sidebar.slider("Number of clusters (K)", 2, 10, 5)
run = st.sidebar.button("🚀 Run Clustering", type="primary")

# -----------------------------
# 4. DATASET OVERVIEW
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(df):,}")
c2.metric("Numeric Features", len(numeric_cols))
c3.metric("Missing Values Fixed", missing_total)
c4.metric("Duplicates Removed", duplicates_removed)

with st.expander("👀 Preview dataset", expanded=True):
    st.dataframe(df.head(10), use_container_width=True)
    st.write("**Summary statistics**")
    st.dataframe(df[numeric_cols].describe().T.round(2), use_container_width=True)

if len(features) < 2:
    st.warning("Select at least 2 features in the sidebar.")
    st.stop()

if not run:
    st.info("Choose features and K, then click **Run Clustering** in the sidebar.")
    st.stop()

# -----------------------------
# 5. HANDLE MISSING + SCALE
# -----------------------------
X = df[features].copy()
X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 6. K-MEANS
# -----------------------------
km = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = km.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, df["Cluster"], sample_size=min(10000, len(df)), random_state=42)

st.subheader("📈 Results")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers", f"{len(df):,}")
m2.metric("Features Used", len(features))
m3.metric("Clusters (K)", k)
m4.metric("Silhouette Score", f"{sil:.3f}")
st.caption("Silhouette: closer to 1 = well-separated clusters, near 0 = overlapping, negative = poor.")

# -----------------------------
# 7. VISUALIZATION
# -----------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader("🎯 Customer Cluster Visualization")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("tab10")

    if len(features) == 2:
        xs, ys = X[features[0]].values, X[features[1]].values
        centers = scaler.inverse_transform(km.cluster_centers_)
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
    else:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        xs, ys = coords[:, 0], coords[:, 1]
        centers = pca.transform(km.cluster_centers_)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")

    for cl in range(k):
        mask = (df["Cluster"] == cl).values
        ax.scatter(xs[mask], ys[mask], s=25, alpha=0.7, color=cmap(cl), label=f"Cluster {cl}")
    ax.scatter(centers[:, 0], centers[:, 1], s=250, c="black", marker="X", label="Centroids")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with right:
    st.subheader("👥 Cluster Sizes")
    sizes = df["Cluster"].value_counts().sort_index()
    fig2, ax2 = plt.subplots(figsize=(5, 5.5))
    ax2.bar([f"C{i}" for i in sizes.index], sizes.values, color=[cmap(i) for i in sizes.index])
    ax2.set_ylabel("Customers")
    for i, v in enumerate(sizes.values):
        ax2.text(i, v, str(v), ha="center", va="bottom")
    st.pyplot(fig2)

# -----------------------------
# 8. ELBOW + SILHOUETTE (why this K?)
# -----------------------------
with st.expander("📉 Elbow Method & Silhouette vs K (why this K?)"):
    ks = list(range(2, 11))
    inertias, sils = [], []
    for kk in ks:
        m = KMeans(n_clusters=kk, random_state=42, n_init=10).fit(X_scaled)
        inertias.append(m.inertia_)
        sils.append(silhouette_score(X_scaled, m.labels_, sample_size=min(5000, len(df)), random_state=42))
    e1, e2 = st.columns(2)
    with e1:
        f, a = plt.subplots(figsize=(5, 3.5))
        a.plot(ks, inertias, marker="o")
        a.axvline(k, color="red", linestyle="--", label=f"Chosen K = {k}")
        a.set_xlabel("K")
        a.set_ylabel("Inertia")
        a.set_title("Elbow Method")
        a.legend()
        st.pyplot(f)
    with e2:
        f, a = plt.subplots(figsize=(5, 3.5))
        a.plot(ks, sils, marker="o", color="green")
        a.axvline(k, color="red", linestyle="--")
        a.set_xlabel("K")
        a.set_ylabel("Silhouette Score")
        a.set_title("Silhouette vs K")
        st.pyplot(f)
    best_k = ks[int(np.argmax(sils))]
    st.write(f"Highest silhouette score is at **K = {best_k}**.")

# -----------------------------
# 9. CLUSTER PROFILE
# -----------------------------
st.subheader("📋 Cluster Profile (average values)")
profile = df.groupby("Cluster")[features].mean().round(1)
profile.insert(0, "Customers", df["Cluster"].value_counts().sort_index())
profile.insert(1, "Share %", (profile["Customers"] / len(df) * 100).round(1))
st.dataframe(profile, use_container_width=True)

# -----------------------------
# 10. BUSINESS INSIGHTS
# -----------------------------
st.subheader("💡 Business Insights & Recommendations")

income_col = next((c for c in features if "income" in c.lower()), None)
spend_col = next((c for c in features if "spend" in c.lower()), None)


def level(z):
    if z > 0.5:
        return "High"
    if z < -0.5:
        return "Low"
    return "Medium"


overall_mean = df[features].mean()
overall_std = df[features].std().replace(0, 1)

for cl in range(k):
    cl_mean = df.loc[df["Cluster"] == cl, features].mean()
    z = (cl_mean - overall_mean) / overall_std
    count = int((df["Cluster"] == cl).sum())

    if income_col and spend_col:
        inc_l, sp_l = level(z[income_col]), level(z[spend_col])
        if inc_l == "Medium" and sp_l == "Medium":
            name, rec = "Average Customers", "General campaigns, seasonal offers and upselling to move them up."
        elif z[income_col] > 0 and z[spend_col] > 0:
            name, rec = "Premium Customers", "Premium products, loyalty program, early access to new launches."
        elif z[income_col] > 0 and z[spend_col] <= 0:
            name, rec = "High-Potential Customers", "Personalized recommendations and targeted promotions to increase spending."
        elif z[income_col] <= 0 and z[spend_col] > 0:
            name, rec = "Value Seekers", "Discounts, bundles and easy-payment offers; they spend a lot relative to income."
        else:
            name, rec = "Budget Customers", "Low-cost offers and affordable bundles; keep marketing spend minimal."
    else:
        top = z.abs().sort_values(ascending=False).index[0]
        name = f"{level(z[top])} {top} Group"
        rec = "Design targeted campaigns around the feature that defines this group."

    stats = " | ".join(f"{f}: {cl_mean[f]:.1f}" for f in features)
    with st.container(border=True):
        st.markdown(f"**Cluster {cl} → {name}**  ({count} customers, {count / len(df) * 100:.1f}%)")
        st.write(stats)
        st.write(f"✅ **Recommendation:** {rec}")

# -----------------------------
# 11. DOWNLOAD
# -----------------------------
st.download_button(
    "⬇️ Download segmented customers (CSV)",
    df.to_csv(index=False).encode("utf-8"),
    file_name="customers_segmented.csv",
    mime="text/csv",
)

st.caption(
    "Scalability note: this pipeline runs locally on a CSV. For millions of records, "
    "the same steps can be implemented with Apache Spark MLlib (distributed K-Means)."
)