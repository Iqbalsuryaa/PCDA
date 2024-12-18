import streamlit as st
import cv2
import numpy as np
import pickle
from skimage.feature import hog
from PIL import Image
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Direktori dasar untuk dataset
base_dir = "/content/drive/MyDrive/PCDA/UasPCD"
kategori = ['Clown_fish', 'Angle_fish', 'Surgeon_fish']

# Fungsi untuk memproses gambar
def preprocess_image(image):
    image_resized = cv2.resize(image, (50, 50))  # Resize image
    hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)  # Convert to HSV
    features_hog, _ = hog(cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')  # Extract HOG features
    features = np.hstack([hsv.flatten(), features_hog])  # Combine HSV and HOG features
    return features

# Fungsi untuk memproses dataset
def preprocess_data(base_dir, kategori):
    data = []
    label = []
    for idx, kategori_name in enumerate(kategori):
        kategori_path = os.path.join(base_dir, kategori_name)
        for img_name in os.listdir(kategori_path):
            img_path = os.path.join(kategori_path, img_name)
            img = cv2.imread(img_path)
            img_features = preprocess_image(img)
            data.append(img_features)
            label.append(idx)  # Menambahkan label kategori
    return np.array(data), np.array(label)

# Visualisasi distribusi data
def periksa_jumlah_gambar(base_dir, kategori):
    for kategori_name in kategori:
        kategori_path = os.path.join(base_dir, kategori_name)
        st.write(f"{kategori_name}: {len(os.listdir(kategori_path))} gambar")

def visualisasi_distribusi_data(label, kategori):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=label)
    plt.xticks(ticks=np.arange(len(kategori)), labels=kategori)
    plt.title("Distribusi Data")
    st.pyplot()

# Load model SVM
with open('model_svm.pkl', 'rb') as file:
    model_svm = pickle.load(file)

# Muat dan preprocess data
data, label = preprocess_data(base_dir, kategori)

# Visualisasi distribusi data
periksa_jumlah_gambar(base_dir, kategori)
visualisasi_distribusi_data(label, kategori)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

# Latih model
model_svm = SVC(kernel='linear', random_state=42)
model_svm.fit(X_train, y_train)

# Simpan model
with open('model_svm.pkl', 'wb') as file:
    pickle.dump(model_svm, file)
st.write("Model telah disimpan sebagai model_svm.pkl")

# Evaluasi model
y_pred = model_svm.predict(X_test)
st.write("Akurasi:", accuracy_score(y_test, y_pred))
st.write("Laporan Klasifikasi:\n", classification_report(y_test, y_pred, target_names=kategori))

# Confusion Matrix
def plot_confusion_matrix(y_test, y_pred, kategori):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=kategori, yticklabels=kategori)
    plt.title("Confusion Matrix")
    plt.xlabel("Prediksi")
    plt.ylabel("Sebenarnya")
    st.pyplot()

plot_confusion_matrix(y_test, y_pred, kategori)

# Streamlit UI
st.title("Aplikasi Klasifikasi Ikan Laut")

# Upload Image
uploaded_file = st.file_uploader("Unggah Gambar Ikan Laut", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Baca gambar yang diunggah
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Tampilkan gambar
    st.image(image, caption='Gambar yang Diupload', use_column_width=True)

    # Prediksi
    if st.button("Prediksi"):
        features = preprocess_image(image)
        prediksi = model_svm.predict([features])[0]
        st.write(f"Prediksi: {kategori[prediksi]}")
