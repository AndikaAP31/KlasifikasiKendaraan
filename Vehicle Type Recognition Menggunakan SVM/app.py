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
st.write("Upload gambar kendaraan untuk mendapatkan prediksi jenis kendaraan beserta skor confidence.")

# input file gambar
uploaded_file = st.file_uploader("📤 Upload gambar kendaraan", type=["jpg", "jpeg", "png"])

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

        # prediksi dan confidence score
        prediction = model.predict(full_feat)[0]
        predicted_label = le.inverse_transform([prediction])[0]
        
        # mendapatkan confidence score untuk SVM custom
        try:
            if len(model.classes_) == 2:
                # Binary classification
                decision = model._decision_function(model.models[0][0], full_feat)
                confidence_score = abs(float(decision[0]))
            else:
                # Multiclass classification - hitung voting scores dan decision values
                votes = np.zeros((full_feat.shape[0], len(model.classes_)))
                decision_values = []
                
                for svm_model, c1, c2 in model.models:
                    decision = model._decision_function(svm_model, full_feat)
                    decision_values.append(abs(float(decision[0])))
                    
                    idx1 = np.where(model.classes_ == c1)[0][0]
                    idx2 = np.where(model.classes_ == c2)[0][0]
                    votes[:, idx1] += (decision < 0).astype(int)
                    votes[:, idx2] += (decision >= 0).astype(int)
                
                # Cari kelas yang diprediksi
                predicted_class_idx = np.argmax(votes[0])
                max_votes = votes[0][predicted_class_idx]
                
                # Hitung confidence berdasarkan:
                # 1. Proporsi vote yang didapat kelas terpilih
                # 2. Rata-rata decision values (seberapa yakin tiap binary classifier)
                vote_confidence = max_votes / len(model.models)
                avg_decision_value = np.mean(decision_values)
                
                # Gabungkan kedua faktor confidence
                confidence_score = vote_confidence * (1 + avg_decision_value)
                
        except Exception as e:
            st.error(f"Error calculating confidence score: {e}")
            confidence_score = 0.5  # default moderate confidence
        
        # normalisasi confidence score ke rentang 0-100%
        # Sesuaikan dengan karakteristik SVM custom
        if confidence_score >= 2.0:
            confidence_percentage = 95.0
        elif confidence_score >= 1.5:
            confidence_percentage = 85.0 + (confidence_score - 1.5) * 20
        elif confidence_score >= 1.0:
            confidence_percentage = 70.0 + (confidence_score - 1.0) * 30
        elif confidence_score >= 0.7:
            confidence_percentage = 55.0 + (confidence_score - 0.7) * 50
        elif confidence_score >= 0.5:
            confidence_percentage = 40.0 + (confidence_score - 0.5) * 75
        else:
            confidence_percentage = confidence_score * 80
        
        # Pastikan confidence dalam range 0-100
        confidence_percentage = min(max(confidence_percentage, 0), 100)

        # tampilkan gambar dan hasil
        st.image(img, channels="BGR", caption="Gambar yang Diunggah", use_container_width=True)
        
        # tampilkan hasil prediksi dengan styling yang lebih menarik
        st.markdown("### 📊 Hasil Prediksi")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**🚗 Jenis Kendaraan:**")
            st.markdown(f"<h3 style='color: #1f77b4; margin-top: 0;'>{predicted_label.upper()}</h3>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**📈 Confidence Score:**")
            st.markdown(f"<h3 style='color: #ff7f0e; margin-top: 0;'>{confidence_percentage:.2f}%</h3>", unsafe_allow_html=True)
        
        # progress bar untuk confidence score
        st.progress(confidence_percentage / 100)
        
        # interpretasi confidence score
        if confidence_percentage >= 80:
            st.success("🎯 Prediksi dengan tingkat kepercayaan tinggi!")
        elif confidence_percentage >= 60:
            st.info("📊 Prediksi dengan tingkat kepercayaan sedang.")
        else:
            st.warning("⚠️ Prediksi dengan tingkat kepercayaan rendah.")
            
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
        Christian Valentino (2308561081)<br>
        Anak Agung Gede Angga Putra Wibawa (2308561099)<br>
        <br>
        © 2025 - All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)