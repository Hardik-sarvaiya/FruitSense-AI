# 🍃 FruitSense AI

**FruitSense AI** is a stylish Streamlit web app that detects whether a fruit image looks **Fresh** or **Rotten** using a trained deep learning model.

---

## ✨ Features

- 🧠 AI-powered fruit freshness detection
- 📤 Single image prediction
- 📦 Batch upload mode for multiple images
- 🎨 Modern dark UI with custom CSS styling
- 📊 Confidence score, certainty level, and inference speed
- 🔒 Privacy-first design
- 🖼️ Supports `JPG`, `JPEG`, `PNG`, and `WEBP`

---

## 📸 How It Works

1. Upload a fruit image
2. The model analyzes the photo
3. The app returns:
   - **Fresh** or **Rotten**
   - Confidence percentage
   - Certainty level
   - Smart suggestion message

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **TensorFlow / Keras**
- **NumPy**
- **Pillow**

---

## 📁 Project Structure

```bash
FruitSense-AI/
├── app.py
├── fruit_fresh_rotten_classifier.keras
├── fruit_fresh_rotten_classifier.h5
├── requirements.txt
└── README.md