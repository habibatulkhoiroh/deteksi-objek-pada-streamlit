from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np

# Judul halaman
st.set_page_config(page_title="Deteksi Objek YOLOv8", layout="centered")
st.title("🚀 Deteksi Objek Menggunakan YOLOv8 + Streamlit")

# Pilihan model: model default atau model custom
use_custom_model = st.sidebar.checkbox("Gunakan Model Sendiri (best.pt)")

if use_custom_model:
    model_path = "best (1).pt"
    st.sidebar.success("Menggunakan model: best.pt")
else:
    model_path = "yolov8n.pt"  # model bawaan dari Ultralytics
    st.sidebar.info("Menggunakan model: yolov8n.pt (default)")

# Load model YOLOv8
try:
    model = YOLO(model_path)
except FileNotFoundError:
    st.error(f"❌ File model tidak ditemukan: {model_path}")
    st.stop()

# Upload gambar dari user
uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar yang diunggah", use_column_width=True)

    # Tombol untuk mulai deteksi
    if st.button("🔍 Deteksi Objek"):
        with st.spinner("🔎 Mendeteksi objek..."):
            results = model.predict(img)
            result_img = results[0].plot()  # gambar dengan bounding box

            st.image(result_img, caption="✅ Hasil Deteksi", use_column_width=True)
            st.success("🎉 Deteksi selesai!")

        # Menampilkan label dan confidence score
        st.subheader("📋 Label Terdeteksi:")
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            st.write(f"• {label} ({conf:.2%})")
