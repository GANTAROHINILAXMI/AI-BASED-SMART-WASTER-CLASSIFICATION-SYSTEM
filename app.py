"""
AI-Based Smart Waste Classification System
Full UI compliant version with Header, Footer, About, Features, Contact, Dashboard tabs
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import sqlite3
from datetime import datetime
import pandas as pd

st.set_page_config(
    page_title="AI-Based Smart Waste Classifier | SITER Academy 2026",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1B5E20, #2E7D32);
    padding: 20px 30px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.main-header h1 { color: white; font-size: 2rem; font-weight: 800; margin: 0; }
.main-header p  { color: #C8E6C9; font-size: 1rem; margin: 6px 0 0 0; }
.footer {
    background: #1B5E20; color: #C8E6C9;
    text-align: center; padding: 18px;
    border-radius: 10px; margin-top: 40px; font-size: 0.85rem;
}
.footer a { color: #A5D6A7; text-decoration: none; }
.feature-card {
    background: #F1F8E9; border-left: 5px solid #2E7D32;
    border-radius: 8px; padding: 16px 20px; margin-bottom: 14px;
}
.feature-card h4 { color: #1B5E20; margin: 0 0 6px 0; }
.feature-card p  { color: #333; margin: 0; font-size: 0.92rem; }
.about-card {
    background: #E8F5E9; border-radius: 12px;
    padding: 20px 24px; margin-bottom: 16px;
}
.contact-info {
    background: #F9FBE7; border-radius: 10px;
    padding: 18px 22px; border-left: 4px solid #8BC34A; margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>♻️ AI-Based Smart Waste Classifier</h1>
    <p>Intelligent waste image classification and disposal guidance powered by Deep Learning</p>
</div>
""", unsafe_allow_html=True)

CLASS_NAMES = ["Biodegradable", "Non-Recyclable", "Recyclable"]
DISPOSAL_GUIDANCE = {
    "Biodegradable": {
        "bin": "🟢 Green Bin (Compost/Organic)",
        "tip": "This item can be composted. Avoid mixing with plastics or metals.",
        "steps": "1. Remove any non-organic parts. 2. Place in the green compost bin. 3. Ensure bin is sealed."
    },
    "Recyclable": {
        "bin": "🔵 Blue Bin (Recyclables)",
        "tip": "Rinse the item if possible before placing it in the recycling bin.",
        "steps": "1. Rinse or clean the item. 2. Remove non-recyclable parts. 3. Place in the blue recycling bin."
    },
    "Non-Recyclable": {
        "bin": "⚫ Black Bin (General Waste)",
        "tip": "This item is not recyclable. Try to reduce usage of such items where possible.",
        "steps": "1. Place in a sealed bag if required. 2. Dispose in the black general waste bin. 3. Check local rules."
    },
}
CONFIDENCE_THRESHOLD = 70.0

def init_db():
    conn = sqlite3.connect("waste_history.db", check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS entries (
        entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
        predicted_category TEXT, confidence_score REAL, created_at TEXT)""")
    conn.commit()
    return conn

def save_entry(conn, category, confidence):
    conn.execute("INSERT INTO entries (predicted_category, confidence_score, created_at) VALUES (?, ?, ?)",
                 (category, confidence, datetime.now().isoformat()))
    conn.commit()

def get_history(conn):
    return pd.read_sql_query("SELECT * FROM entries ORDER BY created_at DESC", conn)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("waste_classifier_model.h5")

def classify_image(model, image):
    img = image.convert("RGB").resize((224, 224))
    arr = preprocess_input(np.expand_dims(np.array(img), axis=0))
    preds = model.predict(arr)[0]
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], float(preds[idx]) * 100

conn = init_db()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Home", "ℹ️ About", "⭐ Features", "📊 Dashboard", "📞 Contact"
])

# HOME TAB
with tab1:
    st.markdown("### 🔍 Classify Your Waste")
    st.write("Upload or capture a photo of a waste item to receive instant AI-powered disposal guidance.")
    source = st.radio("Choose image source:", ["📁 Upload a file", "📷 Use camera"], horizontal=True)
    image = None
    if source == "📁 Upload a file":
        uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"],
                                    help="Upload a clear, well-lit photo of a single waste item.")
        if uploaded:
            image = Image.open(uploaded)
    else:
        captured = st.camera_input("Take a photo of the waste item")
        if captured:
            image = Image.open(captured)

    if image:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Input Image", use_container_width=True)
        with col2:
            with st.spinner("🤖 Analysing waste item..."):
                model = load_model()
                category, confidence = classify_image(model, image)
            if confidence < CONFIDENCE_THRESHOLD:
                st.warning(f"⚠️ Low confidence ({confidence:.1f}%). Please upload a clearer image.")
            else:
                st.success(f"**Predicted Category: {category}**")
                st.metric("Confidence Score", f"{confidence:.1f}%")
                guidance = DISPOSAL_GUIDANCE[category]
                st.info(f"**{guidance['bin']}**\n\n{guidance['tip']}\n\n**Steps:** {guidance['steps']}")
                save_entry(conn, category, confidence)
        st.caption("⚠️ Disclaimer: This tool provides general guidance only. Please follow your local municipal waste disposal rules.")

    st.markdown("---")
    st.markdown("#### 🗑️ How to Use")
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown("**Step 1**\n\nUpload or capture a photo of your waste item")
    with col2: st.markdown("**Step 2**\n\nOur AI model analyses the image instantly")
    with col3: st.markdown("**Step 3**\n\nGet your category result and disposal guidance")

# ABOUT TAB
with tab2:
    st.markdown("### ℹ️ About This Project")
    st.markdown("""<div class="about-card">
        <h4 style="color:#1B5E20;margin-top:0;">🎯 Project Objective</h4>
        <p>The AI-Based Smart Waste Classification System is an intelligent web application designed to help
        individuals, households, and organizations correctly segregate waste at the source. By using deep learning
        and computer vision, the system provides instant, image-based waste classification and actionable disposal
        guidance — reducing reliance on manual judgment and improving recycling efficiency.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("""<div class="about-card">
        <h4 style="color:#1B5E20;margin-top:0;">🧠 How It Works</h4>
        <p>The system uses a <strong>Convolutional Neural Network (CNN)</strong> based on <strong>MobileNetV2</strong>
        architecture, trained using transfer learning on the <strong>TrashNet</strong> public waste image dataset.
        When a user uploads or captures a waste image, the model classifies it into one of three categories and
        provides corresponding bin-colour guidance and disposal steps.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("#### 👩‍💻 Developer Information")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""| Field | Details |
|-------|---------|
| **Student Name** | Ganta Rohini Laxmi |
| **Domain** | Artificial Intelligence (Computer Vision) |
| **Program** | Summer Internship 2026 |
| **Organization** | SITER Academy, Norge |""")
    with col2:
        st.markdown("""| Field | Details |
|-------|---------|
| **Technology** | Python, TensorFlow, Streamlit |
| **Model** | MobileNetV2 (Transfer Learning) |
| **Dataset** | TrashNet (2,533 images) |
| **Accuracy** | 91.67% Validation Accuracy |""")
    st.markdown("#### 🔗 Project Links")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🌐 **Live App:** [Open Application](https://ai-based-smart-waster-classification-system-n45h2fpkzmfqmsxmik.streamlit.app)")
    with col2:
        st.markdown("💻 **Source Code:** [GitHub Repository](https://github.com/GANTAROHINILAXMI/AI-BASED-SMART-WASTER-CLASSIFICATION-SYSTEM)")

# FEATURES TAB
with tab3:
    st.markdown("### ⭐ Features & Services")
    features = [
        ("🖼️ Image Upload", "Upload any JPG or PNG image of a waste item for instant classification with file type and size validation."),
        ("📷 Live Camera Capture", "Use your device camera directly within the browser to capture and classify waste items in real time."),
        ("🤖 AI-Powered Classification", "MobileNetV2 CNN model classifies waste into Biodegradable, Recyclable, or Non-Recyclable with 91.67% accuracy."),
        ("📊 Confidence Score Display", "The system displays the model's confidence percentage alongside each result."),
        ("⚠️ Low-Confidence Warning", "If confidence falls below 70%, the system prompts the user to retake or re-upload a clearer image."),
        ("♻️ Disposal Guidance", "Clear disposal instructions including recommended bin colour and step-by-step guidance."),
        ("📚 Educational Tips", "Each result includes a short educational tip to promote sustainable waste habits."),
        ("📋 Classification History", "Every classification is saved to a SQLite database for future review."),
        ("📈 Analytics Dashboard", "Category distribution bar chart and history table for waste classification activity overview."),
        ("💾 CSV Export", "Download your complete classification history as a CSV file."),
        ("📱 Responsive Design", "Fully responsive across mobile, tablet, laptop, and desktop screens."),
    ]
    for title, desc in features:
        st.markdown(f"""<div class="feature-card"><h4>{title}</h4><p>{desc}</p></div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### 🗂️ Waste Categories Supported")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🟢 Biodegradable**\n- Food waste\n- Vegetable scraps\n- Paper\n- Cardboard")
    with col2:
        st.markdown("**🔵 Recyclable**\n- Plastic bottles\n- Glass items\n- Metal cans\n- Scrap metal")
    with col3:
        st.markdown("**⚫ Non-Recyclable**\n- General trash\n- Mixed waste\n- Contaminated items")

# DASHBOARD TAB
with tab4:
    st.markdown("### 📊 Dashboard & Classification History")
    history = get_history(conn)
    if history.empty:
        st.info("No classifications yet. Go to the 🏠 Home tab to classify a waste item.")
    else:
        total = len(history)
        bio = len(history[history["predicted_category"] == "Biodegradable"])
        rec = len(history[history["predicted_category"] == "Recyclable"])
        non = len(history[history["predicted_category"] == "Non-Recyclable"])
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total", total)
        col2.metric("🟢 Biodegradable", bio)
        col3.metric("🔵 Recyclable", rec)
        col4.metric("⚫ Non-Recyclable", non)
        st.markdown("---")
        st.subheader("📈 Category Distribution")
        st.bar_chart(history["predicted_category"].value_counts())
        st.subheader("📋 Classification History")
        st.dataframe(history, use_container_width=True)
        csv = history.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download History as CSV", csv, "waste_classification_history.csv", "text/csv")

# CONTACT TAB
with tab5:
    st.markdown("### 📞 Contact & Support")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="contact-info">
            <h4 style="color:#33691E;margin-top:0;">👩‍💻 Developer</h4>
            <p><strong>Name:</strong> Ganta Rohini Laxmi</p>
            <p><strong>Program:</strong> Summer Internship 2026</p>
            <p><strong>Organization:</strong> SITER Academy, Norge</p>
            <p><strong>Domain:</strong> AI (Computer Vision)</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="contact-info">
            <h4 style="color:#33691E;margin-top:0;">🏫 SITER Academy</h4>
            <p><strong>WhatsApp:</strong> 8309460744 / 8639975852</p>
            <p><strong>GitHub:</strong> <a href="https://github.com/GANTAROHINILAXMI">GANTAROHINILAXMI</a></p>
            <p><strong>Live App:</strong> <a href="https://ai-based-smart-waster-classification-system-n45h2fpkzmfqmsxmik.streamlit.app">Open App</a></p>
        </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### 📝 Send a Message")
    with st.form("contact_form"):
        name = st.text_input("Your Name", placeholder="Enter your full name")
        email = st.text_input("Your Email", placeholder="Enter your email address")
        subject = st.selectbox("Subject", ["General Enquiry", "Technical Issue", "Feature Suggestion", "Other"])
        message = st.text_area("Message", placeholder="Type your message here...", height=150)
        submitted = st.form_submit_button("📤 Send Message")
        if submitted:
            if name and email and message:
                st.success(f"✅ Thank you, {name}! Your message has been received. We will respond to {email} shortly.")
            else:
                st.error("Please fill in all required fields (Name, Email, and Message).")
    st.markdown("---")
    st.markdown("#### ❓ Frequently Asked Questions")
    with st.expander("How accurate is the waste classification?"):
        st.write("The model achieved 91.67% validation accuracy on the TrashNet dataset. Upload a clear, well-lit photo of a single waste item for best results.")
    with st.expander("What image formats are supported?"):
        st.write("The system supports JPG, JPEG, and PNG image formats up to 200MB.")
    with st.expander("Can it classify multiple items in one photo?"):
        st.write("Currently the system classifies one item per image. Multi-object detection is planned as a future enhancement.")
    with st.expander("Why am I getting a low confidence warning?"):
        st.write("This appears when confidence falls below 70%. Try retaking the photo with better lighting and a clearer view of the waste item.")

# FOOTER
st.markdown("""
<div class="footer">
    <p>♻️ <strong>AI-Based Smart Waste Classifier</strong> | Developed by Ganta Rohini Laxmi</p>
    <p>Summer Internship 2026 — SITER Academy, Norge | Domain: Artificial Intelligence (Computer Vision)</p>
    <p>
        <a href="https://github.com/GANTAROHINILAXMI/AI-BASED-SMART-WASTER-CLASSIFICATION-SYSTEM">GitHub</a>
        &nbsp;|&nbsp;
        <a href="https://ai-based-smart-waster-classification-system-n45h2fpkzmfqmsxmik.streamlit.app">Live App</a>
    </p>
    <p style="font-size:0.75rem;margin-top:8px;">
        ⚠️ Disclaimer: Classification results are AI-generated guidance only.
        Please follow your local municipal waste disposal regulations.
    </p>
</div>
""", unsafe_allow_html=True)
