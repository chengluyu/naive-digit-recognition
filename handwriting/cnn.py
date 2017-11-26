import numpy as np
import os
from keras.models import load_model
from predictor import Predictor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CNNPredictor(Predictor):
    def __init__(self, model_file_name):
        super().__init__()
        self.model = load_model(model_file_name)

    def predict(self, data):
        result = self.model.predict(np.array([[data]]))
        print('CNN prediction result vector:', result)
        return np.argmax(result)
