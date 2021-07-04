import streamlit as st
from tensorflow import keras
from predict_sonnets import load_predict_model


#keras.models.load_model(path)
model = load_predict_model()

st.title('Sonnets Generator')


seed_text = "you are my unicorn, my night, my sun"
next_words = 80

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)
