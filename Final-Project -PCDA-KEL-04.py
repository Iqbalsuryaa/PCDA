import streamlit as st
import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog
import matplotlib.pyplot as plt
import seaborn as sns

# Daftar kategori ikan
kategori = ['Clown_fish', 'Angle_fish', 'Surgeon_fish']

# Direktori dataset
base_dir = "/content/drive/MyDrive/PCDA/UasPCD"

# Fungsi untuk preprocessing data
def preprocess_data(base_dir, kategori):
    data, label = [], []
    for idx, kategori_item in enumerate(kategori):
        path_kategori = os.path.join(base_dir, kategori_item)
        for nama_gambar in os.listdir(path_kategori):
            path_gambar = os.path.join(path_kategori, nama_gambar)
            gambar = cv2.imread(path_gambar)
            
            if gambar is None:
                continue
            
            gambar = cv2.resize(gambar, (50, 50))
            hsv = cv2.cvtColor(gambar, cv2.COLOR_BGR2HSV)
            fitur_hog, _ = hog(cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY),
                               orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), visualize=True,
                               block_norm='L2-Hys')
            fitur = np.hstack([hsv.flatten(), fitur_hog])
            data.append(fitur)
            label.append(idx)
    return np.array(data), np.array(label)

# Fungsi untuk visualisasi distribusi data
def visualisasi_distribusi_data(label, kategori):
    fig, ax = plt.subplots()
    sns.countplot(x=label, ax=ax, palette="viridis")
    ax.set_title("Distribusi Data Antar Kategori")
    ax.set_xticks(range(len(kategori)))
    ax.set_xticklabels(kategori, rotation=45)
    ax.set_xlabel("Kategori")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)

# Fungsi untuk Confusion Matrix
def plot_confusion_matrix(y_test, y_pred, kategori):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=kategori, yticklabels=kategori, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Sebenarnya")
    st.pyplot(fig)

# Muat data
st.title("Klasifikasi Ikan Laut dengan SVM")
st.subheader("1. Memuat dan Preprocessing Data")
data, label = preprocess_data(base_dir, kategori)
st.write(f"Jumlah data: {len(data)}")
visualisasi_distribusi_data(label, kategori)

# Split data
st.subheader("2. Melatih Model")
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
st.write("Data dibagi menjadi:")
st.write(f"- Data latih: {len(X_train)}")
st.write(f"- Data uji: {len(X_test)}")

# Latih model
model_svm = SVC(kernel='linear', random_state=42)
model_svm.fit(X_train, y_train)

# Simpan model
with open('model_svm.pkl', 'wb') as file:
    pickle.dump(model_svm, file)
st.write("Model telah dilatih dan disimpan sebagai 'model_svm.pkl'.")

# Evaluasi model
st.subheader("3. Evaluasi Model")
y_pred = model_svm.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)
st.write(f"Akurasi Model: {akurasi:.2f}")
st.text("Laporan Klasifikasi:")
st.text(classification_report(y_test, y_pred, target_names=kategori))

# Visualisasi Confusion Matrix
plot_confusion_matrix(y_test, y_pred, kategori)
