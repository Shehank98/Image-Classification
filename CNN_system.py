import numpy as np
from PIL import Image
import cv2
import streamlit as st
import tensorflow as tf

# st.set_option('deprecation.showfileuploadreEncoding', False)


# @st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(
        r'C:\Users\aduser\Desktop\SYSTEM\CNN_APP\model.h5')
    return model


model = load_model()

st.write("""
         # Drone Bird Classifer
         """)

file = st.file_uploader("Please upload an image", type=['jpg', 'png'])


def import_and_predict(file, model):
    if file is not None:
        image = Image.open(file)
        st.image(image, use_column_width=True)

        image = tf.image.resize(image, (28, 28))
        scaled_image = tf.image.convert_image_dtype(image, tf.float32) / 255.0
        scaled_image = np.expand_dims(scaled_image, axis=0)
        pred = model.predict(scaled_image)

        # class_labels = ['Bird', 'Drone']
        # predicted_class_index = np.argmax(pred)
        # string = "This image is a: "+class_names[np.argmax(pred)]
        confidence = np.max(pred) * 100
        if pred >= 0.5:
            string = f"This image is a Bird with {confidence:.2f}% confidence."
        else:
            string = f"This image is a Drone with {confidence:.2f}% confidence."

        # string = f"This image is a {class_labels[predicted_class_index]} with {confidence:.2f}% confidence."
        st.success(string)


if file is None:
    st.text("Please upload an image file")
else:
    import_and_predict(file, model)
