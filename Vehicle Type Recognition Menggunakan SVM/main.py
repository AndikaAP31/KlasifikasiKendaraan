# import libary yang akan digunakan
import os
import cv2
import numpy as np
import pickle
from hog import hog
from cg import color_histogram
from glcm import glcm
from other import LabelEncoder, train_test_split
from svm import SVM, GridSearchCV
from report import evaluate, visualize_predictions

# buat variabel untuk menyimpan fitur, label, gambar, dan jumlah gambar
features = []
labels = []
images = []
category_counts = {}

# buat direktori untuk menyimpan gambar
base_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_path, "Dataset")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Folder Dataset tidak ditemukan di: {dataset_path}")

categories = ["bus", "car", "truck", "motorcycle"]

for category in categories:
    folder = os.path.join(dataset_path, category)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder kategori tidak ditemukan: {folder}")

# fungsi untuk melakukan augmentasi data
def augment_image(img):
    augmented = [img]

    flipped = cv2.flip(img, 1)
    augmented.append(flipped)

    h, w = img.shape[:2]
    M_pos15 = cv2.getRotationMatrix2D((w // 2, h // 2), 15, 1.0)
    rotated_pos15 = cv2.warpAffine(img, M_pos15, (w, h))
    augmented.append(rotated_pos15)

    M_neg15 = cv2.getRotationMatrix2D((w // 2, h // 2), -15, 1.0)
    rotated_neg15 = cv2.warpAffine(img, M_neg15, (w, h))
    augmented.append(rotated_neg15)

    return augmented

# melakukan ekstraksi fitur dan augmentasi gambar
print("Memulai ekstraksi fitur dan augmentasi gambar...")

for category in categories:
    folder = os.path.join(dataset_path, category)
    count = 0

    for filename in os.listdir(folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Gagal membaca gambar {img_path}")
            continue
        img = cv2.resize(img, (128, 128))

        for aug_img in augment_image(img):
            hog_feat = hog(aug_img)
            color_feat = color_histogram(aug_img)
            glcm_feat = glcm(aug_img)
            full_feat = np.hstack([hog_feat, color_feat, glcm_feat])
            features.append(full_feat)
            labels.append(category)
            images.append(aug_img)
            count += 1

    category_counts[category] = count
    print(f"{category.capitalize():<12}: {count} gambar berhasil diproses")

# mengubah list fitur menjadi numpy array
X = np.array(features)
y_labels = labels.copy()

# mengubah daftar label menjadi daftar indeks numerik
le = LabelEncoder()
y = le.fit_transform(y_labels)
images = np.array(images)

# membagi dataset menjadi data train dan data test
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(X, y, images, test_size=0.2, random_state=42)

# melakukan training menggunakan model kernel RBF
print("\nMemulai training menggunakan model kernel RBF...")

param_grid_rbf = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}
svm_rbf = GridSearchCV(SVM(), param_grid_rbf, cv=3)
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)

print("\nParameter terbaik (RBF):", svm_rbf.best_params_)

# evaluasi dan visualisasi prediksi model kernel RBF
evaluate(y_test, y_pred_rbf, "SVM RBF", le)
visualize_predictions(y_test, y_pred_rbf, img_test, "SVM RBF", le)

# melakukan training menggunakan model kernel Linear
print("\nMemulai training menggunakan model kernel Linear...")

param_grid_linear = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'kernel': ['linear']
}
svm_linear = GridSearchCV(SVM(), param_grid_linear, cv=3)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)

print("\nParameter terbaik (Linear):", svm_linear.best_params_)

# evaluasi dan visualisasi prediksi model kernel Linear
evaluate(y_test, y_pred_linear, "SVM Linear", le)
visualize_predictions(y_test, y_pred_linear, img_test, "SVM Linear", le)

# menyimpan hasil training dan label encoder dari model terbaik
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm_rbf.best_estimator_, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\nProses selesai dan model telah disimpan.")