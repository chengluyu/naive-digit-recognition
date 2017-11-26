import argparse
import cv2
import numpy as np
from svm import SVMPredictor
from cnn import CNNPredictor

def validate_size(width, height):
    return 5 < width < 100 or 5 < height < 100

RECT_COLOR = (31, 16, 255)
TEXT_COLOR = (9, 8, 9)

def deskew(img, size):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * size * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (size, size), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def normalize(roi, width, height):
    side_length = int(1.4 * max(width, height))
    top = int(side_length - height) // 2
    bottom = side_length - height - top
    left = int(side_length - width) // 2
    right = side_length - width - left
    roi = cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT, 255)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    roi = deskew(roi, 28)
    return roi

def show_single_prediction(roi, result):
    cv2.namedWindow('debug', cv2.WINDOW_NORMAL)
    cv2.setWindowTitle('debug', 'Prediction Result: ' + str(int(result)))
    cv2.imshow('debug', roi)
    cv2.waitKey()

def parse_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--method', dest='method')
    parser.add_argument('--show-digits', dest='show_digits', action='store_true')
    return parser.parse_args()

def select_predictor(method):
    if method == 'cnn':
        return CNNPredictor('../mnist/model.h5')
    elif method == 'hog':
        return SVMPredictor('./digits_cls.pkl')
    raise RuntimeError('unsupported method')

def preprocess(origin):
    image = cv2.fastNlMeansDenoisingColored(origin, None, 10, 10, 7, 21)
    image = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    ret, image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)
    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return image

def segmentation(image):
    ret, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return map(cv2.boundingRect, contours)

if __name__ == '__main__':
    args = parse_argv()
    predictor = select_predictor(args.method)
    raw = cv2.imread(args.image)
    cooked = preprocess(raw)
    for (x, y, w, h) in segmentation(cooked):
        if not validate_size(w, h):
            continue
        cv2.rectangle(raw, (x, y), (x + w, y + h), RECT_COLOR, 1)
        roi = normalize(cooked[y:y+h,x:x+w], w, h)
        result = predictor.predict(roi)
        if args.show_digits:
            show_single_prediction(roi, result)
        cv2.putText(raw, str(int(result)), (x + w // 2, y), cv2.FONT_HERSHEY_PLAIN, 2, TEXT_COLOR, 2)
    cv2.namedWindow('Resulting Image with Rectangular ROIs', cv2.WINDOW_NORMAL)
    cv2.imshow('Resulting Image with Rectangular ROIs', raw)
    cv2.waitKey()
