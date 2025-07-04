from ultralytics import YOLO
import streamlit as st
from PIL import Image

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Deteksi Objek YOLOv8", layout="centered")
st.title("🚀 Deteksi Objek Menggunakan YOLOv8 + Streamlit")

# Load model YOLOv8 bawaan (yolov8n.pt)
st.info("Menggunakan model default: yolov8n.pt")
model = YOLO("yolov8n.pt")

# Upload gambar dari pengguna
uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar yang Diunggah", use_column_width=True)

    # Tombol untuk mendeteksi objek
    if st.button("🔍 Deteksi Objek"):
        with st.spinner("🔎 Mendeteksi objek..."):
            results = model.predict(img)
            result_img = results[0].plot()  # Gambar dengan bounding box

            st.image(result_img, caption="✅ Hasil Deteksi", use_column_width=True)
            st.success("🎉 Deteksi selesai!")

        # Menampilkan label dan skor confidence
        st.subheader("📋 Label Terdeteksi:")
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            st.write(f"• {label} ({conf:.2%})")
