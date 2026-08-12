# ==========================================
# IMPORT LIBRARIES
# ==========================================

import streamlit as st
import pandas as pd
import pickle
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Kalimpong Homestay Recommender",
    page_icon="🏡",
    layout="wide"
)


# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "data/final/homestays_prepared.csv"
    )


@st.cache_resource
def load_models():

    with open(
        "models/tfidf_vectorizer.pkl",
        "rb"
    ) as f:

        tfidf = pickle.load(f)

    with open(
        "models/tfidf_matrix.pkl",
        "rb"
    ) as f:

        tfidf_matrix = pickle.load(f)

    return tfidf, tfidf_matrix


df = load_data()

tfidf, tfidf_matrix = load_models()


# ==========================================
# TITLE
# ==========================================

st.title(
    "🏡 Kalimpong Homestay Recommendation System"
)

st.markdown(
    """
    Find personalized homestay recommendations
    based on your preferences.
    """
)


# ==========================================
# SIDEBAR FILTERS
# ==========================================

st.sidebar.header(
    "Traveler Preferences"
)

selected_block = st.sidebar.selectbox(
    "Preferred Location",
    sorted(
        df["block"].dropna().unique()
    )
)

budget = st.sidebar.slider(
    "Maximum Budget (₹)",
    min_value=int(df["price"].min()),
    max_value=int(df["price"].max()),
    value=3000,
    step=100
)

min_rating = st.sidebar.slider(
    "Minimum Rating",
    0.0,
    5.0,
    4.0,
    0.1
)

selected_amenities = st.sidebar.multiselect(
    "Required Amenities",
    [
        "wifi",
        "parking",
        "breakfast",
        "mountain_view",
        "room_service",
        "bonfire_barbeque",
        "pickup_dropoff_service"
    ]
)

top_n = st.sidebar.slider(
    "Number of Recommendations",
    1,
    10,
    5
)


# ==========================================
# USER QUERY CREATION
# ==========================================

user_text = (
    selected_block
    + " "
    + " ".join(selected_amenities)
    + f" budget {budget}"
)


# ==========================================
# RECOMMENDATION FUNCTION
# ==========================================

def recommend_homestays():

    user_vector = tfidf.transform(
        [user_text]
    )

    similarity_scores = cosine_similarity(
        user_vector,
        tfidf_matrix
    ).flatten()

    filtered_df = df.copy()

    filtered_df = filtered_df[
        filtered_df["price"] <= budget
    ]

    filtered_df = filtered_df[
        filtered_df["rating"] >= min_rating
    ]

    if len(filtered_df) == 0:
        return pd.DataFrame()

    filtered_indices = filtered_df.index

    filtered_df["similarity"] = (
        similarity_scores[
            filtered_indices
        ]
    )

    recommendations = (
        filtered_df
        .sort_values(
            "similarity",
            ascending=False
        )
        .head(top_n)
    )

    return recommendations


# ==========================================
# EXPLANATION FUNCTION
# ==========================================

def generate_explanation(row):

    reasons = []

    if row["block"] == selected_block:
        reasons.append(
            f"Located in {selected_block}"
        )

    if row["price"] <= budget:
        reasons.append(
            "Matches your budget"
        )

    if row["rating"] >= min_rating:
        reasons.append(
            f"Highly rated ({row['rating']:.1f}/5)"
        )

    shared = []

    for amenity in selected_amenities:

        if row[amenity] == 1:

            shared.append(
                amenity.replace(
                    "_",
                    " "
                ).title()
            )

    if shared:

        reasons.append(
            "Provides "
            + ", ".join(shared)
        )

    return reasons


# ==========================================
# GENERATE RESULTS
# ==========================================

if st.button(
    "Get Recommendations"
):

    recommendations = recommend_homestays()

    if len(recommendations) == 0:

        st.warning(
            "No matching homestays found."
        )

    else:

        st.success(
            f"Found {len(recommendations)} recommendations"
        )

        for _, row in recommendations.iterrows():

            with st.container():

                st.subheader(
                    row["homestay_name"]
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Rating",
                    round(
                        row["rating"],
                        1
                    )
                )

                col2.metric(
                    "Price",
                    f"₹{int(row['price'])}"
                )

                col3.metric(
                    "Reviews",
                    int(
                        row["review_count"]
                    )
                )

                st.write(
                    row["description"]
                )

                st.markdown(
                    "**Why Recommended?**"
                )

                for reason in generate_explanation(
                    row
                ):
                    st.write(
                        f"✓ {reason}"
                    )

                st.divider()