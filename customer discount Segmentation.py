import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
def load_data():
    df = pd.read_csv(r"D:\Resume Projects\discount\Customers_Segmented.csv")
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.rename(columns={'Spending Score (1-100)': 'Spending Score'}, inplace=True)
    
    # Debugging: Print column names
    st.write("Columns in dataset:", df.columns.tolist())
    
    return df

df = load_data()

# Sidebar options
st.sidebar.header("Marketing Analytics Dashboard")
visual_type = st.sidebar.selectbox(
    "Select Visualization", 
    ["Bar Chart", "Scatter Plot", "Line Chart", "Pie Chart", "Histogram", "Box Plot", "Heatmap"]
)

x_axis = st.sidebar.selectbox("Select X-axis", df.select_dtypes(include=[np.number]).columns)
y_axis = st.sidebar.selectbox("Select Y-axis", df.select_dtypes(include=[np.number]).columns)

# Display selected visualization
st.subheader(f"{visual_type} of {y_axis} vs {x_axis}")

# Define color theme
color_map = px.colors.qualitative.Vivid  

if visual_type == "Bar Chart":
    fig = px.bar(df, x=x_axis, y=y_axis, color=y_axis, title=f"{y_axis} vs {x_axis}",
                 color_discrete_sequence=color_map)
elif visual_type == "Scatter Plot":
    fig = px.scatter(df, x=x_axis, y=y_axis, color=y_axis, title=f"{y_axis} vs {x_axis}",
                     color_discrete_sequence=color_map, size=y_axis, hover_data=df.columns)
elif visual_type == "Line Chart":
    fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}",
                  color_discrete_sequence=color_map, markers=True, line_shape="spline")
elif visual_type == "Pie Chart":
    fig = px.pie(df, names=x_axis, values=y_axis, title=f"Distribution of {y_axis} by {x_axis}",
                 color_discrete_sequence=color_map, hole=0.3)
elif visual_type == "Histogram":
    fig = px.histogram(df, x=x_axis, nbins=20, title=f"Histogram of {x_axis}",
                       color_discrete_sequence=color_map)
elif visual_type == "Box Plot":
    fig = px.box(df, x=x_axis, y=y_axis, color=x_axis, title=f"Box Plot of {y_axis} by {x_axis}",
                 color_discrete_sequence=color_map)
elif visual_type == "Heatmap":
    fig = px.imshow(df.corr(), color_continuous_scale="Viridis", title="Correlation Heatmap")

# Update layout
fig.update_layout(
    template="plotly_dark",
    font=dict(family="Arial", size=14),
    title_font=dict(size=18, family="Arial", color="white"),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
    margin=dict(l=40, r=40, t=40, b=40)
)

st.plotly_chart(fig)


# K-Means Clustering
st.sidebar.subheader("Customer Segmentation")
k = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)

# Get correct column names dynamically
available_cols = df.columns.tolist()
features = [col for col in available_cols if "Age" in col or "Annual" in col or "Spending" in col]

if len(features) < 3:
    st.error("Error: Required columns not found in dataset!")
else:
    df_scaled = StandardScaler().fit_transform(df[features])
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df_scaled)

    # Cluster Visualization
    st.subheader("Customer Segmentation Visualization")
    fig_cluster = px.scatter(df, x=features[1], y=features[2], color=df['Cluster'].astype(str), 
                            title="Customer Segments", labels={'Cluster': 'Segment'})
    st.plotly_chart(fig_cluster)

    # Insights
    st.subheader("Insights on Marketing Investment")
    st.write("### Suggested Discount Strategies")

    def recommend_discount(cluster):
        if cluster == 0:
            return "10% Discount - Low Spenders"
        elif cluster == 1:
            return "20% Discount - Medium Spenders"
        else:
            return "30% Discount - High Spenders"

    df['Discount_Offer'] = df['Cluster'].apply(recommend_discount)
    st.dataframe(df[['CustomerID', features[1], features[2], 'Cluster', 'Discount_Offer']])





# Visualization for Discount Strategies
st.subheader("Discount Strategy Analysis")

fig_discount = px.bar(df, x=features[1], y=features[2], color="Discount_Offer",
                      title="Discount Distribution by Annual Income and Spending Score",
                      labels={'Discount_Offer': 'Discount Category'},
                      barmode='group', color_discrete_sequence=px.colors.qualitative.Pastel)

st.plotly_chart(fig_discount)

