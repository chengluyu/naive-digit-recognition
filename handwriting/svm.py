from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
from predictor import Predictor

class SVMPredictor(Predictor):
    def __init__(self, model_file_name):
        super().__init__()
        self.clf, self.pp = joblib.load(model_file_name)

    def predict(self, data):
        hog_fd = hog(data, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        hog_fd = self.pp.transform(np.array([hog_fd], 'float64'))
        return self.clf.predict(hog_fd)
