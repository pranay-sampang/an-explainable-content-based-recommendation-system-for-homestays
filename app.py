# ==========================================
# IMPORT LIBRARIES
# ==========================================

import re
import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Kalimpong Homestay Recommender",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# DESIGN SYSTEM
# ==========================================
# Palette and type grounded in Kalimpong itself: misty pine-forest hills,
# monastery gold, and a thin prayer-flag stripe as the recurring
# signature -- rather than a generic dashboard look.

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #EEF1E9;
        --card: #FFFFFF;
        --pine: #1E3A2C;
        --pine-light: #3D6350;
        --saffron: #D9A441;
        --slate: #5C6B63;
        --charcoal: #24291F;
        --border: #DBE1D5;
        --flag-blue: #6D8CAE;
        --flag-cream: #EDE7D3;
        --flag-red: #B2555A;
        --flag-green: #4F7A5C;
        --flag-gold: #D9A441;
    }

    .stApp {
        background: var(--bg);
    }

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
        color: var(--charcoal);
    }

    h1, h2, h3, .hero-title, .rec-name {
        font-family: 'Fraunces', serif;
        color: var(--pine) !important;
        letter-spacing: -0.01em;
    }

    /* Streamlit auto-applies dark-mode text colors based on browser/OS
       preference, which can wash out custom text against a light custom
       background -- .streamlit/config.toml forces light base theme, and
       these !important rules are a second layer of insurance so the
       actual palette always wins regardless of viewer settings. */
    .stApp p, .stMarkdown p, .rec-description, .rec-location {
        color: var(--charcoal) !important;
    }

    /* -------- Hero header -------- */
    .hero {
        padding: 2.25rem 0 1.25rem 0;
    }
    .hero-eyebrow {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--saffron);
        margin-bottom: 0.4rem;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
        line-height: 1.1;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--slate);
        max-width: 40rem;
        line-height: 1.5;
    }
    .flag-stripe {
        height: 5px;
        width: 100%;
        margin-top: 1.5rem;
        border-radius: 3px;
        background: linear-gradient(
            90deg,
            var(--flag-blue) 0%, var(--flag-blue) 20%,
            var(--flag-cream) 20%, var(--flag-cream) 40%,
            var(--flag-red) 40%, var(--flag-red) 60%,
            var(--flag-green) 60%, var(--flag-green) 80%,
            var(--flag-gold) 80%, var(--flag-gold) 100%
        );
    }

    /* -------- Sidebar -------- */
    section[data-testid="stSidebar"] {
        background: var(--pine);
    }
    section[data-testid="stSidebar"] * {
        color: #F2F5EF !important;
    }
    section[data-testid="stSidebar"] h2 {
        color: #FFFFFF !important;
        font-size: 1.3rem;
    }
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label {
        font-weight: 600;
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* -------- Buttons -------- */
    .stButton > button {
        background: var(--saffron);
        color: var(--charcoal);
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-size: 1rem;
        width: 100%;
        transition: transform 0.1s ease, box-shadow 0.1s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(30, 58, 44, 0.2);
        color: var(--charcoal);
    }

    /* -------- Recommendation cards -------- */
    .rec-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .rec-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(
            90deg,
            var(--flag-blue) 0%, var(--flag-blue) 20%,
            var(--flag-cream) 20%, var(--flag-cream) 40%,
            var(--flag-red) 40%, var(--flag-red) 60%,
            var(--flag-green) 60%, var(--flag-green) 80%,
            var(--flag-gold) 80%, var(--flag-gold) 100%
        );
    }
    .rec-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.3rem;
    }
    .rec-name {
        font-family: 'Fraunces', serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--pine);
        margin: 0;
    }
    .rec-location {
        color: var(--slate);
        font-size: 0.9rem;
        margin-bottom: 0.9rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .badge-gold {
        background: #FBEED0;
        color: #92680F;
    }
    .badge-silver {
        background: #E7EAE5;
        color: var(--slate);
    }
    .stat-row {
        display: flex;
        gap: 1.75rem;
        margin: 1rem 0 1.1rem 0;
        padding: 0.85rem 1rem;
        background: var(--bg);
        border-radius: 10px;
    }
    .stat {
        display: flex;
        flex-direction: column;
    }
    .stat-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--slate);
        margin-bottom: 0.15rem;
    }
    .stat-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--charcoal);
    }
    .rec-description {
        color: var(--charcoal);
        line-height: 1.55;
        font-size: 0.96rem;
        margin-bottom: 1.1rem;
    }
    .why-box {
        background: #F3F6F0;
        border-left: 3px solid var(--pine-light);
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
    }
    .why-title {
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--pine);
        margin-bottom: 0.5rem;
    }
    .why-reason {
        font-size: 0.92rem;
        color: var(--charcoal);
        margin-bottom: 0.3rem;
    }
    .match-score {
        font-size: 0.78rem;
        color: var(--slate);
        text-align: right;
    }

    /* -------- Empty / error states -------- */
    .empty-state {
        background: var(--card);
        border: 1px dashed var(--border);
        border-radius: 14px;
        padding: 2.5rem 2rem;
        text-align: center;
        color: var(--slate);
    }

    /* -------- Multiselect dropdown popover -------- */
    /* Streamlit/BaseWeb portals this popup to the end of <body>, not
       inside section[data-testid="stSidebar"] -- so the sidebar-scoped
       "* { color: ... }" rule above never reaches it, and it was
       falling back to its unstyled dark default (most visible as a
       plain black box when a multiselect has no options left to show,
       e.g. every amenity already picked). These selectors target the
       popover directly by its component attributes instead of by DOM
       ancestry, so they apply wherever it's actually rendered.
       Not visually verified against a live run -- check it renders as
       expected. */
    div[data-baseweb="popover"] {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="popover"] ul[data-baseweb="menu"] {
        background: var(--card) !important;
    }
    div[data-baseweb="popover"] li {
        color: var(--charcoal) !important;
    }
    div[data-baseweb="popover"] li:hover {
        background: var(--bg) !important;
    }
    /* The "no options left" message itself -- shown as plain text with
       no distinguishing wrapper in most Streamlit versions, so this
       catches any leftover unstyled text inside the popover. */
    div[data-baseweb="popover"] * {
        color: var(--slate);
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ==========================================
# TEXT PREPROCESSOR (must exactly match notebook 05)
# ==========================================
# pickle doesn't save a vectorizer's custom preprocessor function's
# actual code -- only a reference to where it expects to find it
# (__main__.normalize_text, since Jupyter runs notebook code as
# __main__). When Streamlit runs this file, THIS becomes __main__, so
# the function needs to be defined here too, identically, or unpickling
# fails with "module '__main__' has no attribute 'normalize_text'".

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"wi[\s-]*fi", "wifi", text)
    text = re.sub(r"drop[\s-]*off", "dropoff", text)
    text = re.sub(r"pick[\s-]*up", "pickup", text)
    return text


# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        "data/final/homestays_prepared.csv"
    )

    # Guarantees data.iloc[i].name == i -- the similarity-score lookup
    # below depends on this holding, rather than assuming it by luck.
    return data.reset_index(drop=True)


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


try:
    df = load_data()
    tfidf, tfidf_matrix = load_models()

except FileNotFoundError:
    st.error(
        "Recommendation model files weren't found. Run notebook 05 "
        "(`05_recommendation_system.ipynb`) first to generate "
        "`models/tfidf_vectorizer.pkl` and `models/tfidf_matrix.pkl`."
    )
    st.stop()


# ==========================================
# TOURIST LOCATIONS (for proximity preference)
# ==========================================

TOURIST_LOCATIONS = {
    "Deolo Hill": "deolo",
    "Durpin Monastery": "durpin",
    "Kalimpong Town": "town",
    "Lava": "lava",
    "Pedong": "pedong",
    "Gorubathan": "gorubathan",
    "Rishop": "rishop",
    "Lolegaon": "lolegaon",
}

AMENITY_COLUMNS = [
    "wifi",
    "parking",
    "breakfast",
    "mountain_view",
    "room_service",
    "bonfire_barbeque",
    "pickup_dropoff_service",
]

FEATURE_NAMES = tfidf.get_feature_names_out()


# ==========================================
# ITEM-TO-ITEM SIMILARITY (Section 3.7.3: "user selects a homestay")
# ==========================================
# This mirrors the mechanism actually evaluated in Chapter 4 -- each
# homestay's own row in tfidf_matrix is used as the query vector,
# rather than a synthetic preference profile. Kept alongside the
# preference-search mode below rather than replacing it, since they
# answer different questions ("what's like X" vs "what fits Y").

def recommend_similar_to(homestay_name, top_n):

    matches = df.index[df["homestay_name"] == homestay_name]

    if len(matches) == 0:
        return pd.DataFrame(), None

    seed_position = matches[0]
    seed_vector = tfidf_matrix[seed_position]

    similarity_scores = cosine_similarity(
        seed_vector,
        tfidf_matrix
    ).flatten()

    result_df = df.copy()
    result_df["similarity"] = similarity_scores

    # Exclude the seed homestay itself from its own recommendations.
    result_df = result_df.drop(index=seed_position)

    recommendations = (
        result_df
        .sort_values("similarity", ascending=False)
        .head(top_n)
    )

    return recommendations, df.loc[seed_position]


def get_shared_keywords(seed_position, candidate_position, top_k=5):
    """Non-zero TF-IDF terms both records share, ranked by combined
    weight -- the 'common descriptive keywords' referenced in Section
    3.8.1. Note these come straight from the tokenised vocabulary
    (Section 3.7.1), so results can include fragments like place-name
    pieces or proximity tokens rather than natural-language phrases --
    the same caveat already discussed in Section 4.2."""

    seed_vec = tfidf_matrix[seed_position].toarray().flatten()
    candidate_vec = tfidf_matrix[candidate_position].toarray().flatten()

    shared_mask = (seed_vec > 0) & (candidate_vec > 0)

    if not shared_mask.any():
        return []

    shared_terms = FEATURE_NAMES[shared_mask]
    shared_weights = seed_vec[shared_mask] * candidate_vec[shared_mask]

    order = np.argsort(shared_weights)[::-1][:top_k]

    return list(shared_terms[order])


def generate_similarity_explanation(source_row, candidate_row):
    """Explanation for item-to-item mode: attributes the SOURCE homestay
    and the CANDIDATE recommendation have in common, per Section 3.8.1 --
    as opposed to generate_explanation() below, which compares a
    candidate against the user's stated preferences."""

    reasons = []

    if candidate_row["block"] == source_row["block"]:
        reasons.append(
            f"Also located in {candidate_row['block']}"
        )

    if candidate_row["category"] == source_row["category"]:
        reasons.append(
            f"Same {candidate_row['category']} category"
        )

    shared_amenities = [
        amenity.replace("_", " ").title()
        for amenity in AMENITY_COLUMNS
        if source_row.get(amenity, 0) == 1 and candidate_row.get(amenity, 0) == 1
    ]

    if shared_amenities:
        reasons.append(
            "Both offer " + ", ".join(shared_amenities)
        )

    if abs(candidate_row["price"] - source_row["price"]) <= 300:
        reasons.append(
            f"Comparable price (₹{int(candidate_row['price']):,}/night)"
        )

    matched_landmarks = [
        display_name
        for display_name, key in TOURIST_LOCATIONS.items()
        if source_row.get(f"{key}_proximity") == "Very Close"
        and candidate_row.get(f"{key}_proximity") == "Very Close"
    ]

    if matched_landmarks:
        reasons.append(
            "Both very close to " + ", ".join(matched_landmarks)
        )

    # Rating is excluded from the vectorised profile (Section 3.8.3), and
    # is only ever surfaced here if NEITHER record's location/rating data
    # was affected by the geocoding name-collision failure mode (Section
    # 3.3) -- same gating as generate_explanation() below.
    source_reliable = not source_row.get("location_corrected", False)
    candidate_reliable = not candidate_row.get("location_corrected", False)

    if (
        source_reliable
        and candidate_reliable
        and abs(candidate_row["rating"] - source_row["rating"]) <= 0.5
    ):
        reasons.append(
            f"Similar rating ({candidate_row['rating']:.1f}/5)"
        )

    shared_keywords = get_shared_keywords(
        source_row.name,
        candidate_row.name,
    )

    if shared_keywords:
        reasons.append(
            "Shares the descriptors: " + ", ".join(shared_keywords)
        )

    return reasons


# ==========================================
# HERO
# ==========================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">Kalimpong, West Bengal</div>
        <h1 class="hero-title">Find your homestay in the hills</h1>
        <p class="hero-subtitle">
            Set what matters to you, and we'll match you with homestays
            across Kalimpong's villages, tea gardens, and viewpoints --
            with a clear explanation for every recommendation.
        </p>
        <div class="flag-stripe"></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# BUDGET → PRICE BAND
# ==========================================
# The old version injected the raw budget number as a literal token
# ("budget 3000") into the TF-IDF query text. That token can never
# match anything -- no homestay's feature_text contains a specific
# price figure, only the categorical price_band ("budget"/"mid_range"/
# "premium"). This derives the correct band directly from the live
# data's own price_band boundaries, rather than hardcoding thresholds
# that could drift out of sync with however notebook 04 actually built them.
# Defined here, before the sidebar, since the preference-search branch
# below calls it while building user_text.

def budget_to_price_band(budget_value, data):

    band_ranges = (
        data.groupby("price_band")["price"]
        .agg(["min", "max"])
    )

    for band, row in band_ranges.iterrows():
        if row["min"] <= budget_value <= row["max"]:
            return band

    band_ranges["mid"] = (
        band_ranges["min"] + band_ranges["max"]
    ) / 2

    return (
        (band_ranges["mid"] - budget_value)
        .abs()
        .idxmin()
    )


# ==========================================
# SIDEBAR: SEARCH MODE
# ==========================================

st.sidebar.markdown("## Find Recommendations")

search_mode = st.sidebar.radio(
    "How would you like to search?",
    ["Describe what I want", "I already like a homestay"],
    help=(
        "'Describe what I want' searches by the preferences you set below. "
        "'I already like a homestay' finds homestays most similar to one "
        "you pick -- the same content-based comparison used throughout "
        "this project's evaluation (Section 3.7.3)."
    ),
)

top_n = st.sidebar.slider(
    "Number of recommendations",
    1,
    10,
    5,
)

st.sidebar.markdown("---")

if search_mode == "Describe what I want":

    st.sidebar.markdown("### Your Preferences")

    selected_block = st.sidebar.selectbox(
        "Preferred area",
        sorted(
            df["block"].dropna().unique()
        )
    )

    budget = st.sidebar.slider(
        "Maximum budget (₹ / night)",
        min_value=int(df["price"].min()),
        max_value=int(df["price"].max()),
        value=3000,
        step=100,
    )

    min_rating = st.sidebar.slider(
        "Minimum rating",
        0.0,
        5.0,
        4.0,
        0.1,
    )

    selected_amenities = st.sidebar.multiselect(
        "Amenities you want",
        AMENITY_COLUMNS,
        format_func=lambda a: a.replace("_", " ").title(),
    )

    selected_landmarks = st.sidebar.multiselect(
        "Want to be close to...",
        list(TOURIST_LOCATIONS.keys()),
        help="Recommendations near these spots will be prioritized -- this isn't a hard filter.",
    )

    # Query construction lives here, inside the branch, because it
    # references selected_block/selected_amenities/selected_landmarks --
    # names that only exist in this mode. Computing it unconditionally at
    # module level would raise a NameError as soon as someone switched to
    # "I already like a homestay", since budget_to_price_band(budget, df)
    # would run before budget was ever defined.
    price_band_token = budget_to_price_band(budget, df)

    landmark_tokens = [
        f"{TOURIST_LOCATIONS[name]}_very_close"
        for name in selected_landmarks
    ]

    user_text = " ".join([
        selected_block,
        " ".join(selected_amenities),
        price_band_token,
        " ".join(landmark_tokens),
    ])

else:

    st.sidebar.markdown("### Pick a Homestay You Like")

    selected_homestay_name = st.sidebar.selectbox(
        "Homestay",
        sorted(df["homestay_name"].dropna().unique()),
        help=(
            "We'll rank every other homestay by similarity to this one's "
            "full profile (category, village, price band, amenities, "
            "proximity, and description) -- not just the few attributes "
            "used in the preference search."
        ),
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

    # Positional lookup via get_indexer rather than assuming index
    # labels equal array positions -- correct regardless of whether
    # df's index happens to be a plain range.
    positions = df.index.get_indexer(filtered_df.index)

    filtered_df = filtered_df.copy()
    filtered_df["similarity"] = similarity_scores[positions]

    recommendations = (
        filtered_df
        .sort_values("similarity", ascending=False)
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
            f"Within your ₹{budget:,} budget"
        )

    # Rating claim skipped if this homestay's location (and therefore
    # its rating, sourced from the same Google match) was flagged as
    # manually corrected -- same gating used in notebook 05, since
    # rating/review_count for those records were never independently
    # re-verified.
    rating_is_reliable = not row.get("location_corrected", False)

    if rating_is_reliable and row["rating"] >= min_rating:
        reasons.append(
            f"Highly rated ({row['rating']:.1f}/5)"
        )

    shared_amenities = [
        amenity.replace("_", " ").title()
        for amenity in selected_amenities
        if row.get(amenity, 0) == 1
    ]

    if shared_amenities:
        reasons.append(
            "Offers " + ", ".join(shared_amenities)
        )

    matched_landmarks = [
        name
        for name in selected_landmarks
        if row.get(f"{TOURIST_LOCATIONS[name]}_proximity") == "Very Close"
    ]

    if matched_landmarks:
        reasons.append(
            "Very close to " + ", ".join(matched_landmarks)
        )

    return reasons


# ==========================================
# SHARED CARD RENDERER
# ==========================================
# Both search modes end up with a DataFrame of recommendations plus a
# per-row list of explanation strings -- only how those two things are
# produced differs (recommend_homestays/generate_explanation vs.
# recommend_similar_to/generate_similarity_explanation above). Rendering
# is identical either way, so it's kept in one place rather than
# duplicated per mode, which is what let the two card layouts drift
# apart if only one copy ever got edited.

def render_recommendation_card(row, reasons):

    badge_class = (
        "badge-gold"
        if str(row["category"]).lower() == "gold"
        else "badge-silver"
    )

    reasons_html = "".join(
        f'<div class="why-reason">✓ {r}</div>'
        for r in reasons
    )

    st.markdown(
        f"""
        <div class="rec-card">
            <div class="rec-header">
                <h3 class="rec-name">{row['homestay_name']}</h3>
                <span class="badge {badge_class}">{row['category']}</span>
            </div>
            <div class="rec-location">📍 {row['village']}, {row['block']}</div>
            <div class="stat-row">
                <div class="stat">
                    <span class="stat-label">Rating</span>
                    <span class="stat-value">★ {row['rating']:.1f}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Price / night</span>
                    <span class="stat-value">₹{int(row['price']):,}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Reviews</span>
                    <span class="stat-value">{int(row['review_count'])}</span>
                </div>
            </div>
            <div class="rec-description">{row['description']}</div>
            <div class="why-box">
                <div class="why-title">Why recommended</div>
                {reasons_html if reasons_html else '<div class="why-reason">Matches your general preferences</div>'}
                <div class="match-score">{row['similarity']*100:.0f}% match</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# GENERATE RESULTS
# ==========================================

button_label = (
    "Find My Homestay"
    if search_mode == "Describe what I want"
    else "Show Similar Homestays"
)

if st.button(button_label):

    if search_mode == "Describe what I want":

        recommendations = recommend_homestays()

        if len(recommendations) == 0:

            st.markdown(
                """
                <div class="empty-state">
                    <strong>No homestays match every filter.</strong><br>
                    Try raising your budget or lowering the minimum rating --
                    Kalimpong's homestays are mostly small, family-run places,
                    so narrow combinations can rule out the whole list.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f"**{len(recommendations)} homestays match your preferences**"
            )
            st.write("")

            for _, row in recommendations.iterrows():
                reasons = generate_explanation(row)
                render_recommendation_card(row, reasons)

    else:

        recommendations, seed_row = recommend_similar_to(
            selected_homestay_name,
            top_n,
        )

        if seed_row is None or len(recommendations) == 0:

            st.markdown(
                """
                <div class="empty-state">
                    <strong>Couldn't find that homestay.</strong><br>
                    Try picking another one from the list.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f"**Homestays most similar to {selected_homestay_name}**"
            )
            st.write("")

            for _, row in recommendations.iterrows():
                reasons = generate_similarity_explanation(seed_row, row)
                render_recommendation_card(row, reasons)