import os
from tensorflow import keras



def load_predict_model():
    root_path = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(root_path,'new_model','nn_model')
    return keras.models.load_model(path)



