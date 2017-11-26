import cv2
import numpy as np
from svm import SVMPredictor
from cnn import CNNPredictor

def validate_size(width, height):
    return 5 < width < 100 and 5 < height < 100

# predictor = SVMPredictor('digits_cls.pkl')
predictor = CNNPredictor('../mnist/model.h5')

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


def normalize(roi, weight, height):
    margin_x = int(w * 0.4 if w > h else (h - w) / 2)
    margin_y = int(h * 0.4 if h > w else (w - h) / 2)
    roi = cv2.copyMakeBorder(roi, margin_y, margin_y, margin_x, margin_x, cv2.BORDER_CONSTANT, 255)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    roi = deskew(roi, 28)
    return roi

def show_single_prediction(roi, result):
    cv2.namedWindow('debug', cv2.WINDOW_NORMAL)
    cv2.setWindowTitle('debug', 'Prediction Result: ' + str(int(result)))
    cv2.imshow('debug', roi)
    cv2.waitKey()

if __name__ == '__main__':
    origin = cv2.imread('IMG_2988.jpg')
    origin = cv2.resize(origin, (origin.shape[1] // 2, origin.shape[0] // 2))
    image = cv2.fastNlMeansDenoisingColored(origin, None, 10, 10, 7, 21)
    image = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    _, image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('Binarized Image', image)
    _, contours, hier = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for (x, y, w, h) in map(cv2.boundingRect, contours):
        if not validate_size(w, h):
            continue
        cv2.rectangle(origin, (x, y), (x + w, y + h), RECT_COLOR, 1)
        roi = normalize(image[y:y+h,x:x+w], w, h)
        result = predictor.predict(roi)
        show_single_prediction(roi, result)
        cv2.putText(origin, str(int(result)), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, TEXT_COLOR, 2)

    cv2.namedWindow('Resulting Image with Rectangular ROIs', cv2.WINDOW_NORMAL)
    cv2.imshow('Resulting Image with Rectangular ROIs', origin)
    cv2.waitKey()
