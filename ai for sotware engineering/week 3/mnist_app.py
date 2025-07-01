import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- IMPORTANT: Ensure the model is trained and saved --- #
# Before running this app, make sure you have run mnist_cnn.py
# and that it successfully saved 'mnist_cnn_model.h5'.
# -------------------------------------------------------- #

st.set_page_config(page_title="MNIST Digit Classifier", page_icon=":pencil:")

st.title("MNIST Handwritten Digit Classifier :pencil:")
st.write("Upload an image of a handwritten digit (0-9) and let the CNN model predict it!")

# Load the pre-trained model
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_model():
    try:
        model = tf.keras.models.load_model("mnist_cnn_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model. Please ensure 'mnist_cnn_model.h5' exists in the 'week 3' folder. Error: {e}")
        return None

model = load_model()

if model:
    st.header("1. Upload Your Digit Image")
    uploaded_file = st.file_uploader("Choose an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("L") # Convert to grayscale
        st.image(image, caption="Uploaded Image", width=150)

        st.write("### Analyzing Image...")

        # Preprocess the image
        img_array = np.array(image)
        
        # Resize to 28x28 if not already
        if img_array.shape != (28, 28):
            img_array = Image.fromarray(img_array)
            img_array = img_array.resize((28, 28), Image.Resampling.LANCZOS) # Use LANCZOS for high-quality downsampling
            img_array = np.array(img_array)

        img_array = img_array.reshape(1, 28, 28, 1).astype('float32') # Reshape for model input
        img_array = img_array / 255.0 # Normalize pixel values

        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        predicted_digit = np.argmax(prediction)
        
        # Get prediction probabilities
        probabilities = prediction[0]

        st.header("2. Prediction Result")
        st.success(f"The predicted digit is: **{predicted_digit}**")

        st.subheader("Prediction Probabilities")
        # Create a bar chart of probabilities
        fig, ax = plt.subplots(figsize=(8, 4))
        digits = range(10)
        ax.bar(digits, probabilities)
        ax.set_xticks(digits)
        ax.set_xlabel("Digit")
        ax.set_ylabel("Probability")
        ax.set_title("Probability Distribution for Each Digit")
        st.pyplot(fig)

        st.write("--- Developed using TensorFlow and Streamlit --- ") 