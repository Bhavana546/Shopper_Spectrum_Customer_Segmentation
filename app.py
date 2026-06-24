# ============================================================
# SHOPPER SPECTRUM
# Customer Segmentation and Product Recommendation System
# ============================================================

# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

# Configure Streamlit page title, icon, and layout
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

# ============================================================
# LOAD SAVED MODELS AND FILES
# ============================================================

@st.cache_resource
def load_models():
    """
    Load all saved machine learning models and files.

    cache_resource keeps models in memory so Streamlit
    does not reload them every time the user clicks a button.
    """

    # Load trained KMeans customer segmentation model
    kmeans_model = joblib.load(
        "kmeans_customer_segmentation_model.pkl"
    )

    # Load StandardScaler used during model training
    scaler = joblib.load(
        "rfm_scaler.pkl"
    )

    # Load product-to-product cosine similarity matrix
    product_similarity_df = joblib.load(
        "product_similarity_matrix.pkl"
    )

    # Load available product names
    available_products = joblib.load(
        "available_products.pkl"
    )

    # Load cluster number to customer segment mapping
    cluster_labels = joblib.load(
        "cluster_labels.pkl"
    )

    return (
        kmeans_model,
        scaler,
        product_similarity_df,
        available_products,
        cluster_labels
    )


# Load all models and data files
(
    kmeans_model,
    scaler,
    product_similarity_df,
    available_products,
    cluster_labels
) = load_models()

# ============================================================
# HELPER FUNCTION: PRODUCT RECOMMENDATIONS
# ============================================================

def get_product_recommendations(product_name, top_n=5):
    """
    Return top similar products using cosine similarity.

    Parameters:
    product_name : Selected product name
    top_n        : Number of recommended products

    Returns:
    DataFrame containing recommended products and similarity scores
    """

    # Get similarity scores for the selected product
    similarity_scores = product_similarity_df[product_name]

    # Sort similarity scores in descending order.
    # [1:] removes the selected product itself because
    # every product has similarity score 1 with itself.
    similar_products = similarity_scores.sort_values(
        ascending=False
    )[1:top_n + 1]

    # Create a clean recommendation table
    recommendations = pd.DataFrame({
        "Recommended Product": similar_products.index,
        "Similarity Score": similar_products.values.round(3)
    })

    return recommendations


# ============================================================
# HELPER FUNCTION: CUSTOMER SEGMENT PREDICTION
# ============================================================

def predict_customer_segment(recency, frequency, monetary):
    """
    Predict customer segment using RFM values.

    Parameters:
    recency   : Days since customer's last purchase
    frequency : Number of customer orders
    monetary  : Total amount spent by customer

    Returns:
    Predicted cluster number and segment label
    """

    # Apply log transformation.
    # This must match the transformation used during model training.
    rfm_log_values = np.log1p(
        [[recency, frequency, monetary]]
    )

    # Standardize values using the saved scaler.
    rfm_scaled_values = scaler.transform(
        rfm_log_values
    )

    # Predict KMeans cluster.
    predicted_cluster = kmeans_model.predict(
        rfm_scaled_values
    )[0]

    # Convert cluster number into meaningful segment name.
    predicted_segment = cluster_labels[predicted_cluster]

    return predicted_cluster, predicted_segment


# ============================================================
# APP HEADER
# ============================================================

st.title("🛒 Shopper Spectrum")
st.subheader("Customer Segmentation and Product Recommendations in E-Commerce")

st.write(
    "Use this application to discover similar products and "
    "predict customer segments using Recency, Frequency, and Monetary values."
)

st.divider()

# ============================================================
# CREATE TWO TABS
# ============================================================

product_tab, customer_tab = st.tabs([
    "🎁 Product Recommendations",
    "👥 Customer Segmentation"
])

# ============================================================
# TAB 1: PRODUCT RECOMMENDATION MODULE
# ============================================================

with product_tab:

    st.header("🎁 Product Recommendation Module")

    st.write(
        "Select a product to receive 5 similar product recommendations "
        "based on customer purchase behavior."
    )

    # Create dropdown containing all available products.
    selected_product = st.selectbox(
        "Select a Product",
        options=sorted(available_products)
    )

    # Create recommendation button.
    if st.button("Get Recommendations", key="recommend_button"):

        # Get top 5 similar products.
        recommendations = get_product_recommendations(
            selected_product
        )

        # Display selected product.
        st.success(f"Recommendations for: {selected_product}")

        # Display recommendations as a table.
        st.dataframe(
            recommendations,
            use_container_width=True
        )

# ============================================================
# TAB 2: CUSTOMER SEGMENTATION MODULE
# ============================================================

with customer_tab:

    st.header("👥 Customer Segmentation Module")

    st.write(
        "Enter customer RFM values to predict the customer segment."
    )

    # Create 3 columns for RFM inputs.
    col1, col2, col3 = st.columns(3)

    # Input for Recency.
    with col1:
        recency_input = st.number_input(
            "Recency (Days Since Last Purchase)",
            min_value=0,
            value=30,
            step=1
        )

    # Input for Frequency.
    with col2:
        frequency_input = st.number_input(
            "Frequency (Number of Orders)",
            min_value=1,
            value=5,
            step=1
        )

    # Input for Monetary.
    with col3:
        monetary_input = st.number_input(
            "Monetary (Total Amount Spent)",
            min_value=0.0,
            value=500.0,
            step=10.0
        )

    # Create prediction button.
    if st.button("Predict Customer Segment", key="predict_button"):

        # Predict cluster and segment.
        cluster, segment = predict_customer_segment(
            recency_input,
            frequency_input,
            monetary_input
        )

        # Display prediction result.
        st.success(f"Predicted Customer Segment: {segment}")

        # Display extra details.
        st.info(f"Predicted Cluster Number: {cluster}")

        # Show segment-specific business interpretation.
        if segment == "High-Value":
            st.write(
                "⭐ This customer is highly valuable. "
                "Offer loyalty rewards, premium recommendations, "
                "and early access to new products."
            )

        elif segment == "Regular":
            st.write(
                "🔁 This customer purchases regularly. "
                "Use personalized offers and cross-selling campaigns."
            )

        elif segment == "Occasional":
            st.write(
                "🛍️ This customer purchases occasionally. "
                "Use discounts and product reminders to increase engagement."
            )

        elif segment == "At-Risk":
            st.write(
                "⚠️ This customer has not purchased recently. "
                "Use retention campaigns, special offers, and re-engagement emails."
            )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built using RFM Analysis, KMeans Clustering, "
    "Cosine Similarity, and Streamlit."
)