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

tokenizer = Tokenizer()


def corpus(data):
    corpus = data.lower().split("\n")
    return corpus


def removing_characters(corpus):
    mycorpus = [re.compile(r"by william shakespeare").sub("", m) for m in corpus]
    finalcorpus = [re.compile(r"the sonnets").sub("", m) for m in mycorpus]
    return finalcorpus

def splitting_text(corpus):
    tokenizer = Tokenizer()
    corpus = tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index)
    return corpus
