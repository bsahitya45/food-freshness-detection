import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"D:\archive\food_freshness_model.h5"
DB_PATH = r"D:\archive\food_spoilage.db"

IMG_SIZE = (224, 224)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Food Freshness Detection",
    layout="wide"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("Food Freshness Detection Dashboard")

st.write(
    "Upload a food image to determine whether the food is Fresh or Spoiled."
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


try:

    model = load_model()

except Exception as e:

    st.error(
        f"Could not load the model: {e}"
    )

    st.stop()


# ============================================================
# DATABASE
# ============================================================

def create_database():

    conn = sqlite3.connect(
        DB_PATH
    )

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS food_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            predicted_class TEXT,
            confidence REAL,
            status TEXT
        )
    """)

    conn.commit()

    conn.close()


create_database()


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("Upload Food Image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    # Open image
    img = Image.open(
        uploaded_file
    ).convert("RGB")

    # Display image
    st.image(
        img,
        caption="Uploaded Image",
        width="stretch"
    )

    # Resize image
    img_resized = img.resize(
        IMG_SIZE,
        Image.Resampling.LANCZOS
    )

    # Convert image to NumPy array
    img_array = np.asarray(
        img_resized,
        dtype=np.float32
    )

    # MobileNetV2 preprocessing
    img_array = preprocess_input(
        img_array
    )

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Make prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )

    probability = float(
        prediction[0][0]
    )


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if probability >= 0.5:

        predicted_class = "Spoiled"
        status = "Spoiled"

        confidence = probability * 100

        st.error(
            "Spoiled Food"
        )

    else:

        predicted_class = "Fresh"
        status = "Fresh"

        confidence = (
            1 - probability
        ) * 100

        st.success(
            "Fresh Food"
        )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )


    if confidence < 70:

        st.warning(
            "Low confidence prediction. "
            "The image may be different from the training dataset."
        )


    # ========================================================
    # SAVE RESULT TO DATABASE
    # ========================================================

    conn = sqlite3.connect(
        DB_PATH
    )

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO food_results
        (
            image_name,
            predicted_class,
            confidence,
            status
        )
        VALUES (?, ?, ?, ?)
    """, (
        uploaded_file.name,
        predicted_class,
        confidence,
        status
    ))

    conn.commit()

    conn.close()


# ============================================================
# PREDICTION HISTORY
# ============================================================

st.divider()

st.header("Prediction History")


conn = sqlite3.connect(
    DB_PATH
)

try:

    df = pd.read_sql_query(
        "SELECT * FROM food_results",
        conn
    )

finally:

    conn.close()


# ============================================================
# KPI CARDS
# ============================================================

if len(df) > 0:

    total_predictions = len(df)

    fresh_count = len(
        df[
            df["status"] == "Fresh"
        ]
    )

    spoiled_count = len(
        df[
            df["status"] == "Spoiled"
        ]
    )

    avg_confidence = df[
        "confidence"
    ].mean()


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Predictions",
            total_predictions
        )


    with col2:

        st.metric(
            "Fresh Items",
            fresh_count
        )


    with col3:

        st.metric(
            "Spoiled Items",
            spoiled_count
        )


    with col4:

        st.metric(
            "Average Confidence",
            f"{avg_confidence:.2f}%"
        )


    # ========================================================
    # ALL PREDICTIONS
    # ========================================================

    st.subheader(
        "All Predictions"
    )

    st.dataframe(
        df,
        width="stretch"
    )


    # ========================================================
    # LAST 5 PREDICTIONS
    # ========================================================

    st.subheader(
        "Last 5 Predictions"
    )

    st.dataframe(
        df.tail(5),
        width="stretch"
    )


    # ========================================================
    # SEARCH IMAGE
    # ========================================================

    st.subheader(
        "Search Image"
    )

    search = st.text_input(
        "Enter image name"
    )


    if search:

        filtered = df[
            df["image_name"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

        st.dataframe(
            filtered,
            width="stretch"
        )


    # ========================================================
    # DOWNLOAD CSV
    # ========================================================

    st.subheader(
        "Download Results"
    )

    csv = df.to_csv(
        index=False
    )

    st.download_button(
        label="Download Results CSV",
        data=csv,
        file_name="food_results.csv",
        mime="text/csv"
    )


    # ========================================================
    # CONFIDENCE TREND
    # ========================================================

    st.subheader(
        "Confidence Trend"
    )

    st.line_chart(
        df["confidence"]
    )


    # ========================================================
    # FRESH VS SPOILED DISTRIBUTION
    # ========================================================

    counts = df[
        "status"
    ].value_counts()


    st.subheader(
        "Fresh vs Spoiled Distribution"
    )


    fig, ax = plt.subplots()

    ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%"
    )

    st.pyplot(
        fig
    )


    # ========================================================
    # FRESH VS SPOILED COUNT
    # ========================================================

    st.subheader(
        "Fresh vs Spoiled Count"
    )

    st.bar_chart(
        counts
    )


else:

    st.info(
        "No predictions yet. Upload a food image above."
    )