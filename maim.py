# streamlit_app.py
import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import os
CATEGORIES = ['negative', 'positive', 'surprise']

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(CATEGORIES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

st.title('PTSD Prediction')
st.write('By IDRIS SALAU GATA 21D/47CS/2017')

uploaded_file = st.file_uploader('Upload an image...', type=['bmp'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    image = image.convert('RGB')
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    model = create_model()
    model.load_weights('ptsd_cnn_model.h5')

    prediction = model.predict(image)
    class_idx = np.argmax(prediction, axis=1)[0]
    class_name = CATEGORIES[class_idx]

    st.write(f'Prediction: {class_name}')

