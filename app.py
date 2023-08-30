import streamlit as st
import numpy as np
import pickle
import io
import os
import gc
from PIL import Image
from keras.models import load_model
from keras.utils.image_utils import img_to_array, load_img
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Model

gc.collect()

# Load the image caption generator model
icg_model = load_model("icg_model_v18_2.h5", compile=False)
if icg_model is not None:
    icg_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


tokenizer = Tokenizer()
# Load the captions tokenizer
with open('captions_tokenizer_new.pickle', 'rb') as t:
    tokenizer = pickle.load(t)

vgg16 = VGG16()
vgg16 = Model(inputs=vgg16.inputs, outputs=vgg16.layers[-2].output)


def save_uploaded_image(uploaded_file):
    # Save the uploaded image and return the image path
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True) 
    image_path = os.path.join("uploads", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return image_path

# Function to extract features from images using VGG16
def extract_features(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = vgg16.predict(x, verbose=0) # type: ignore
    return features

def generate_caption(model, tokenizer, img_features, max_caption_length):
    caption = 'sseq'
    for i in range(max_caption_length):
        caption_sequence = tokenizer.texts_to_sequences([caption])[0]
        caption_sequence = pad_sequences([caption_sequence], maxlen=max_caption_length)
        predictions = model.predict([img_features, caption_sequence], verbose=0)
        predicted_index = np.argmax(predictions)
        word = tokenizer.index_word[predicted_index]
        caption += ' ' + word
        if word == 'eseq':
            break   
    return caption

def main():
    st.title("Image Caption Generator")
    uploaded_file = st.file_uploader("Choose an image..", type=["jpg", "jpeg", "png", "gif"])
    
    if uploaded_file is not None:
        
        image = Image.open(io.BytesIO(uploaded_file.read()))
        image_path = save_uploaded_image(uploaded_file)
        
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Generate Caption'):
            if icg_model is None:
                return st.write("Model: failed to load the model")
            gc.collect()
            img_features = extract_features(image_path)
            caption = generate_caption(icg_model, tokenizer, img_features, 34)
            # Trim the caption by removing 'sseq' and 'eseq'
            caption = caption.replace('sseq', '').replace('eseq', '')
            # Remove leading and trailing spaces
            caption = caption.strip()
            st.write("### ", caption)


if __name__ == '__main__':
    main()