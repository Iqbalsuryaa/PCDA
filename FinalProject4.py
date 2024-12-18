import streamlit as st
import cv2
import numpy as np
import pickle
from skimage.feature import hog
from PIL import Image
import os

# Load model SVM
with open('model_svm.pkl', 'rb') as file:
    model_svm = pickle.load(file)

# Kategori ikan
kategori = ['Clown_fish', 'Angle_fish', 'Surgeon_fish']

def preprocess_image(image):
    # Resize image
    image_resized = cv2.resize(image, (50, 50))

    # Convert to HSV
    hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)

    # Extract HOG features
    features_hog, _ = hog(cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY),
                          orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True,
                          block_norm='L2-Hys')

    # Combine HSV and HOG features
    features = np.hstack([hsv.flatten(), features_hog])
    return features

def predict(image):
    features = preprocess_image(image)
    prediction = model_svm.predict([features])[0]
    return kategori[prediction]

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
        prediksi = predict(image)
        st.write(f"Prediksi: {prediksi}")
