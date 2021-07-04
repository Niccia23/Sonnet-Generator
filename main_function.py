import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.utils as ku
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import re

from data_processing import corpus
from data_processing import removing_characters
from data_processing import splitting_text

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import pickle,os
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping



data = open("shakespeare.txt").read()

path = os.path.join('..','nn_model','model')

root_path = os.path.dirname(os.path.dirname(__file__))

tokenizer_path = os.path.join(root_path,'nn_model','model')


data_clean = corpus(data)

data_removing_characters = removing_characters(data)

data_tokenizer = splitting_text(data)

tokenizer = Tokenizer()
total_words = len(tokenizer.word_index)


input_sequences = []
for line in data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences], default=1)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

train, target = input_sequences[:,:-1],input_sequences[:,-1]
target = ku.to_categorical(target, num_classes=total_words+1)







def model(train, target, epochs=160, verbose=1):
    model = Sequential()
    model.add(Embedding(total_words+1, 100, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(150, return_sequences = True)))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(total_words+1/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(total_words+1, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train, target, epochs=160, verbose=1)

    return model



def save_model(pipeline):
 #   with open(path, "wb") as file:
 #           pickle.dump(pipeline, file)
    pipeline.save(path)





