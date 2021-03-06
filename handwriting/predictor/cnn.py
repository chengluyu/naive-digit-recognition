import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)
from keras.models import load_model
from .predictor import Predictor


class CNNPredictor(Predictor):
    def __init__(self, model_file_name):
        super().__init__()
        self.model = load_model(model_file_name)

    def predict(self, data):
        reshaped = data.reshape(1, 1, 28, 28).astype('float32') / 255
        result = self.model.predict(np.array(reshaped))
        return np.argmax(result)
