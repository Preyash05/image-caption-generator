import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

MAX_LENGTH = 35
IMG_SIZE = 224

@st.cache_resource
def load_resources():
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    model = load_model('model.keras')
    
    feature_extractor = DenseNet201(weights='imagenet', include_top=False, pooling='avg')
    
    return tokenizer, model, feature_extractor

tokenizer, model, feature_extractor = load_resources()

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption(image, tokenizer, model, feature_extractor):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    features = feature_extractor.predict(img, verbose=0)
    
    in_text = 'startseq'
    
    for i in range(MAX_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH, padding='post')
        
        yhat = model.predict([features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        
        word = idx_to_word(yhat, tokenizer)
        
        if word is None:
            break
            
        in_text += " " + word
        
        if word == 'endseq':
            break
            
    return in_text.replace('startseq', '').replace('endseq', '')

st.title("AI Image Caption Generator")
st.markdown("Upload an image to generate a caption.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    
    if st.button('Generate Caption'):
        with st.spinner('Generating caption...'):
            caption = generate_caption(image, tokenizer, model, feature_extractor)
            st.success(f"Caption: {caption}")

st.markdown("---")
st.markdown("Built by **Preyash Baveja**")