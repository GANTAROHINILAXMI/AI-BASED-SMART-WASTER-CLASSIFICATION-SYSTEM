import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import sqlite3
from datetime import datetime
import pandas as pd

CLASS_NAMES = ["Biodegradable", "Non-Recyclable", "Recyclable"]

DISPOSAL_GUIDANCE = {
    "Biodegradable": {
        "bin": "Green Bin (Compost/Organic)",
        "tip": "This item can be composted. Avoid mixing with plastics or metals.",
    },
    "Recyclable": {
        "bin": "Blue Bin (Recyclables)",
        "tip": "Rinse the item if possible before placing it in the recycling bin.",
    },
    "Non-Recyclable": {
        "bin": "Black Bin (General Waste)",
        "tip": "This item is not recyclable. Reduce usage of such items where possible.",
    },
}

CONFIDENCE_THRESHOLD = 70.0

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("waste_classifier_model.h5")

def init_db():
    conn = sqlite3.connect("waste_history.db", check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_category TEXT,
            confidence_score REAL,
            created_at TEXT
        )
    """)
    conn.commit()
    return conn

def save_entry(conn, category, confidence):
    conn.execute(
        "INSERT INTO entries (predicted_category, confidence_score, created_at) VALUES (?, ?, ?)",
        (category, confidence, datetime.now().isoformat()),
    )
    conn.commit()

def get_history(conn):
    return pd.read_sql_query("SELECT * FROM entries ORDER BY created_at DESC", conn)

def classify_image(model, image: Image.Image):
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)[0]
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], float(preds[idx]) * 100

st.set_page_config(page_title="AI Waste Classifier", page_icon=":recycle:", layout="centered")
conn = init_db()

st.title("AI-Based Smart Waste Classifier")
st.write("Upload or capture a photo of a waste item to get instant disposal guidance.")

tab1, tab2 = st.tabs(["Classify Waste", "Dashboard & History"])

with tab1:
    source = st.radio("Choose image source:", ["Upload a file", "Use camera"])
    image = None

    if source == "Upload a file":
        uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded)
    else:
        captured = st.camera_input("Take a photo")
        if captured:
            image = Image.open(captured)

    if image:
        st.image(image, caption="Input image", width=300)

        with st.spinner("Classifying..."):
            model = load_model()
            category, confidence = classify_image(model, image)

        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(f"Low confidence ({confidence:.1f}%). Please retake or upload a clearer image.")
        else:
            st.success(f"Predicted Category: {category}")
            st.metric("Confidence", f"{confidence:.1f}%")

            guidance = DISPOSAL_GUIDANCE[category]
            st.info(f"Disposal Guidance: {guidance['bin']}\n\n{guidance['tip']}")

            save_entry(conn, category, confidence)

        st.caption("Disclaimer: This tool provides general guidance only. Please follow your local municipal waste disposal rules.")

with tab2:
    st.subheader("Classification History")
    history = get_history(conn)

    if history.empty:
        st.write("No classifications yet. Classify a waste item to see history here.")
    else:
        st.dataframe(history, use_container_width=True)

        st.subheader("Category Distribution")
        counts = history["predicted_category"].value_counts()
        st.bar_chart(counts)

        csv = history.to_csv(index=False).encode("utf-8")
        st.download_button("Download history as CSV", csv, "waste_history.csv", "text/csv")
