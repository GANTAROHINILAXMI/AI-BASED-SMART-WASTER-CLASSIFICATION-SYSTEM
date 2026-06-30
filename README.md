# ♻️ AI-Based Smart Waste Classifier

An AI-powered web application that classifies waste images into **Biodegradable**, **Recyclable**, and **Non-Recyclable** categories using a Convolutional Neural Network (CNN), and provides instant disposal guidance to promote correct waste segregation.

---

## 🌐 Live Demo
[Click here to open the app](https://ai-based-smart-waster-classification-system-n45h2fpkzmfqmsxmik.streamlit.app)

---

## 📌 Project Overview

- **Student:** Ganta Rohini Laxmi
- **Domain:** Artificial Intelligence (Computer Vision)
- **Program:** Summer Internship 2026 — SITER Academy, Norge
- **Technology:** Python, TensorFlow, MobileNetV2, Streamlit, SQLite

---

## 🎯 Objective

To develop an AI-powered web application that:
- Classifies waste images into Biodegradable, Recyclable, or Non-Recyclable categories
- Provides clear, actionable disposal guidance based on the predicted category
- Tracks classification history with an analytics dashboard
- Promotes environmental sustainability through accessible AI technology

---

## 🧠 How It Works

1. User uploads or captures a photo of a waste item
2. The image is pre-processed (resized to 224x224, normalized)
3. A fine-tuned MobileNetV2 CNN model classifies the image
4. The predicted category and confidence score are displayed
5. Disposal guidance is shown based on the category
6. The entry is saved to a SQLite database for history tracking

---

## 🗂️ Waste Categories

| Category | Examples | Bin Color |
|---|---|---|
| Biodegradable | Food waste, paper, cardboard | 🟢 Green Bin |
| Recyclable | Plastic bottles, metal, glass | 🔵 Blue Bin |
| Non-Recyclable | General trash | ⚫ Black Bin |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| AI Model | TensorFlow / Keras — MobileNetV2 (Transfer Learning) |
| Dataset | TrashNet (remapped into 3 categories) |
| Web Interface | Streamlit |
| Database | SQLite |
| Deployment | Streamlit Community Cloud |
| Version Control | Git / GitHub |

---

## 📊 Model Performance

- **Architecture:** MobileNetV2 + custom classification head
- **Training epochs:** 10
- **Validation Accuracy:** 91.67%
- **Dataset:** TrashNet — 2,533 images remapped into 3 classes
  - Biodegradable: 997 images
  - Recyclable: 1,399 images
  - Non-Recyclable: 137 images

---

## ✨ Features

- Image upload (JPG/PNG) or live camera capture
- AI classification with confidence score display
- Disposal guidance and bin-color recommendations
- Low-confidence warning (below 70%) with retry prompt
- Classification history stored in SQLite database
- Dashboard with category distribution bar chart
- CSV export of classification history
- Responsive design for mobile, tablet, and desktop
- Disclaimer note for local municipal waste rule compliance

---

## 📁 Project Structure
