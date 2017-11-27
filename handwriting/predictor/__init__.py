from .svm import SVMPredictor
from .cnn import CNNPredictor

def select_predictor(method):
    if method == 'cnn':
        return CNNPredictor('../mnist/model.h5')
    elif method == 'hog':
        return SVMPredictor('./digits_cls.pkl')
    raise RuntimeError('unsupported method "%s"' % method)
