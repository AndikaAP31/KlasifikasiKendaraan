import streamlit as st
import numpy as np
import cv2
import pickle
from hog import hog
from cg import color_histogram
from glcm import glcm

st.set_page_config(page_title="Vehicle Type Recognition using SVM", layout="centered")

# fungsi untuk memuat model dan label encoder
@st.cache_resource
def load_model():
    with open("svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_model()

# judul dan deskripsi utama
st.title("Vehicle Type Prediction using SVM")
st.write("Upload gambar kendaraan dan inputkan label asli, lalu tekan tombol prediksi untuk melihat hasilnya.")

# input file gambar dan label asli
uploaded_file = st.file_uploader("📤 Upload gambar kendaraan", type=["jpg", "jpeg", "png"])
true_label = st.selectbox("🏷️ Pilih label asli kendaraan", ["bus", "car", "truck", "motorcycle"])

# tombol prediksi
if st.button("🔍 Prediksi", key="predict_button"):
    if uploaded_file is not None:

        # membaca gambar dan preprocessing
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.resize(img, (128, 128))

        # ekstraksi fitur
        hog_feat = hog(img)
        color_feat = color_histogram(img)
        glcm_feat = glcm(img)
        full_feat = np.hstack([hog_feat, color_feat, glcm_feat]).reshape(1, -1)

        # prediksi
        prediction = model.predict(full_feat)[0]
        predicted_label = le.inverse_transform([prediction])[0]

        # tampilkan gambar dan hasil
        st.image(img, channels="BGR", caption="Gambar yang Diunggah", use_container_width=True)
        st.markdown(f"### ✅ Prediksi: `{predicted_label}`")
        st.markdown(f"### 🎯 Label Asli: `{true_label}`")

        # evaluasi prediksi
        if predicted_label.lower() == true_label.lower():
            st.success("🎉 Prediksi benar!")
        else:
            st.error("❌ Prediksi salah.")
    else:
        st.warning("⚠️ Silahkan upload gambar terlebih dahulu sebelum memprediksi.")

st.markdown("""<hr style="margin-top:50px;">""", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
        Dibuat oleh: Kelompok C4 Informatika Udayana Angkatan 2023<br>
        <br>
        I Putu Satria Dharma Wibawa (2308561045)<br>
        I Putu Andika Arsana Putra (2308561063)<br>
        Christian Valentino (2308561081<br>
        Anak Agung Gede Angga Putra Wibawa (2308561099)<br>
        <br>
        © 2025 - All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)