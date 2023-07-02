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
from keras.utils import pad_sequences
from keras.models import Model

# Load the image caption generator model
icg_model = load_model("icg_model_v15.h5")

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

def indexToWord(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def preprocess_image(img_path):
    image = load_img(img_path, target_size=(224, 224))
    # convert image pixels to numpy array
    image = img_to_array(image)
    # reshape image
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # preprocess image for vgg
    image = preprocess_input(image)
    # extract features
    image_features = vgg16.predict(image, verbose='0')
    return image_features

def generate_caption(img_path, model, tokenizer, maxCaptionLength):
    imageFeatures = preprocess_image(img_path)
    # add start tag for generation process
    caption = 'sseq'
    # iterate over the max length of sequence
    for i in range(maxCaptionLength):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([caption])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], maxCaptionLength)
        
        # Check if the shapes and data types match the model's inputs
        if imageFeatures.shape != (1, 4096) or sequence.shape != (1, 35):
            return "Error: Invalid input shape or data type"
        
        # predict next word
        predictions = model.predict([imageFeatures, sequence], verbose=0)
        # get word index with the highest probability
        outputWordIndex = np.argmax(predictions)
        # convert index to word
        word = indexToWord(outputWordIndex, tokenizer)
        # stop if word is not found in vocabulary
        if word is None:
            break
        # append word as input for generating next word
        caption += " " + word
        # stop if we reach the end of sequence
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
            caption = generate_caption(image_path, icg_model, tokenizer, 35)
            # Trim the caption by removing 'sseq' and 'eseq'
            caption = caption.replace('sseq', '').replace('eseq', '')
            # Remove leading and trailing spaces
            caption = caption.strip()
            st.write("### ", caption)


if __name__ == '__main__':
    main()
    
#if icg_model is not None:
#        # Capture the model summary
#        buffer = io.StringIO()
#        icg_model.summary(print_fn=lambda x: buffer.write(x + '\n'))  # Capture the summary output
#
#        # Display the model summary
#        summary = buffer.getvalue()
#        st.code(summary)
