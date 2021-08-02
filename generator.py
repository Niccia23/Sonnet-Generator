import streamlit as st
from tensorflow import keras
from predict_sonnets import load_predict_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences




#keras.models.load_model(path)
model = load_predict_model()

st.title('Sonnets Generator')



