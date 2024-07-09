# streamlit_app.py
import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import numpy as np
from PIL import Image
import os

# Define paths
DATA_DIR = 'dataset'
CATEGORIES = ['negative', 'positive', 'surprise']

# Function to load and preprocess images
def load_images(data_dir, categories, img_size=(64, 64)):
    data = []
    labels = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                image = Image.open(img_path).convert('RGB')
                image = image.resize(img_size)
                image = np.array(image)
                data.append(image)
                labels.append(class_num)
            except Exception as e:
                pass
    data = np.array(data) / 255.0
    labels = np.array(labels)
    return data, labels

# Load and preprocess data
data, labels = load_images(DATA_DIR, CATEGORIES)
labels = to_categorical(labels, num_classes=len(CATEGORIES))
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define the CNN model
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

# Train the model
model = create_model()
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save('ptsd_cnn_model1.h5')

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(y_true, y_pred_classes)
class_report = classification_report(y_true, y_pred_classes, target_names=CATEGORIES)

print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
# Streamlit app
st.title('PTSD Prediction App')

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

