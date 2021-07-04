import pandas as pd

import numpy as np
import tensorflow.keras.utils as ku

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

import pickle,os
import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from train_function import model, save_model





data = open("shakespeare.txt").read()

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

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

train, target = input_sequences[:,:-1],input_sequences[:,-1]
target = ku.to_categorical(target, num_classes=total_words+1)



nn_model = model(train, target, epochs=160, verbose=1)

save_model(nn_model)
