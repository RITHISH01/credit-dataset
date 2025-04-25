import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Show the page title and description.
st.set_page_config(page_title="Credit Risk Prediction", page_icon="💳")
st.title("💳 Credit Risk Prediction Dashboard")
st.write(
    """
    This app visualizes the **Credit Risk Prediction** for loan applicants using clustering 
    and classification techniques. It helps to explore different risk categories based on loan
    applicants' personal and financial information.
    """
)

# Load the data from a CSV. Caching this to prevent reloading each time.
@st.cache_data
def load_data():
    df = pd.read_csv("data/german_credit_data.csv")  # Replace with the correct path to your dataset
    return df

df = load_data()

# Show a multiselect widget for selecting features to explore
features = st.multiselect(
    "Select Features to Explore",
    df.columns.tolist(),
    ["Age", "Credit amount", "Duration"]  # Default selection
)

# Show a slider widget to filter applicants by age
age_range = st.slider("Select Age Range", 18, 100, (20, 60))

# Filter the data based on user input
df_filtered = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]
df_filtered = df_filtered[features]

# Display the filtered data as a table
st.dataframe(df_filtered, use_container_width=True)

# Preprocessing: Handle missing values, scaling, and clustering
df_filtered = df_filtered.fillna(df_filtered.mean())  # Simple fill for missing values (adjust as needed)

# Apply scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_filtered)

# KMeans clustering to predict risk groups (clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
df_filtered['Cluster'] = kmeans.fit_predict(df_scaled)

# PCA for 2D visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_scaled)
df_filtered['PCA1'] = pca_components[:, 0]
df_filtered['PCA2'] = pca_components[:, 1]

# Display KMeans Clustering Results as an Altair chart
st.subheader("PCA Clustering View")
df_chart = pd.DataFrame(df_filtered[['PCA1', 'PCA2', 'Cluster']])
chart = alt.Chart(df_chart).mark_circle(size=60).encode(
    x='PCA1', y='PCA2', color='Cluster:N', tooltip=['PCA1', 'PCA2', 'Cluster']
).properties(height=400)
st.altair_chart(chart, use_container_width=True)

# Display a bar chart showing the distribution of risk clusters
st.subheader("Cluster Distribution")
cluster_counts = df_filtered['Cluster'].value_counts().reset_index()
cluster_counts.columns = ['Cluster', 'Count']
bar_chart = alt.Chart(cluster_counts).mark_bar().encode(
    x='Cluster:N', y='Count:Q', color='Cluster:N'
).properties(height=300)
st.altair_chart(bar_chart, use_container_width=True)

# Ensure 'Credit amount' and 'Duration' are included in df_filtered for boxplots
required_columns = ['Credit amount', 'Duration']
for col in required_columns:
    if col not in df_filtered.columns:
        df_filtered[col] = df[col]

# Boxplot for financial features (e.g., Credit amount, Duration)
st.subheader("Credit Amount and Duration by Risk Category")
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Boxplot for Credit Amount
sns.boxplot(x=df_filtered['Cluster'], y=df_filtered['Credit amount'], ax=ax[0])
ax[0].set_title("Credit Amount by Risk Category")

# Boxplot for Loan Duration
sns.boxplot(x=df_filtered['Cluster'], y=df_filtered['Duration'], ax=ax[1])
ax[1].set_title("Loan Duration by Risk Category")

st.pyplot(fig)

# Insights and conclusions (optional)
st.write(
    """
    ### Insights:
    - Clusters represent distinct risk categories based on personal and financial information.
    - You can analyze the features in the boxplots to understand which variables contribute to credit risk.
    """
)

