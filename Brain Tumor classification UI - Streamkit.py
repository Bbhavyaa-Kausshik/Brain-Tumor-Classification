#We have to import streamkit at st
import streamlit as st
st.title("MRI Classification of Brain Tumor")
st.header("Upload a brain MRI Image for image classification as tumor or no-tumor")

import keras
from PIL import Image, ImageOps
import numpy as np

keras_model = (r"<model path>")


def teachable_machine_classification(img, weights_file):
    # So we load the model
    model = keras.models.load_model(weights_file)

    # Now we create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #Next, turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run it
    prediction = model.predict(data)
    return np.argmax(prediction) #we did this so that it can return position of the highest probability

import time #this is to show the progress bar when its classyfying

uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg")
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI.', use_column_width=True)
        st.write("")
        st.write("Classifying... \n Please wait for a moment")
        label = teachable_machine_classification(image, r"<model path>")
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)
        if label == 0:
            st.write("The MRI scan is healthy")
            st.balloons() #fills the screen with balloon for 3 seconds
        else:
            st.write("The MRI scan has a brain tumor")
        