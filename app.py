
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model (ensure this path is correct)
model = load_model('D:/PROGRAMMING/RespiraScan/pneumonia_detection_model.h5')

# Streamlit app title and description
st.title("RespiraScan 🫁")

# Add description with some more style
st.markdown('<p class="header">Detect COVID-19 & Pneumonia from Chest X-ray</p>', unsafe_allow_html=True)
st.markdown('<p class="description">Upload a chest X-ray image to predict whether the image shows signs of Normal, Pneumonia, or COVID-19. Stay healthy and let the AI assist in early detection!</p>', unsafe_allow_html=True)
st.title("Where's my Waffle ??")
# Upload image
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)  # Corrected here

    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))  # Resize image to 224x224
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Get prediction
    prediction = model.predict(img_array)
    classes = ['NORMAL', 'PNEUMONIA', 'COVID19']
    predicted_class = classes[np.argmax(prediction)]
    
    st.write(f"Prediction: {predicted_class}")

