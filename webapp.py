import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np

st.header("Flower Classification CNN Model:")
flower_name = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

model = load_model("iris_flower_classifier.h5")


def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    prediction = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(prediction[0])

    outcome = f"The image belongs to '{flower_name[np.argmax(result)]}' with a score of {np.max(result)*100:.2f}%"
    return outcome


uploaded_file = st.file_uploader("Upload an Image: ")
if uploaded_file is not None:
    os.makedirs("Uploaded", exist_ok=True)

    with open(os.path.join("Uploaded", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, width=200)
    st.markdown(classify_images(uploaded_file))
