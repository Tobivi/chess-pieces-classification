import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('history.h5')

def preprocess_image(image_path, size):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (size, size))
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)  
    return img

def main():
    st.title("Your Model Deployment with Streamlit")

    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        size =  your_target_size  
        img_array = preprocess_image(uploaded_file, size)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        st.write(f"Prediction: Class {predicted_class}")
        st.write(f"Confidence: {prediction[0][predicted_class]:.2%}")

if __name__ == "__main__":
    main()
