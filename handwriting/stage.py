import argparse
import cv2
import math
import time
import numpy as np
from predictor import select_predictor

"""
Divide the whole process into many stages.
"""
class Stage:
    def __init__(self, name):
        self.name = name

    def run(self, input):
        pass

class Loader(Stage):
    def __init__(self, filename):
        super().__init__('loader')
        self.filename = filename

    def run(self, input):
        return cv2.imread(self.filename)

class Thresholding(Stage):
    def __init__(self):
        super().__init__('thresholding')

    def run(self, input):
        image = cv2.fastNlMeansDenoisingColored(input, None, 10, 10, 7, 21)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        # ret, image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 9)
        return input, image

class Scissor(Stage):
    def __init__(self):
        super().__init__('scissor')

    @staticmethod
    def remove_outliers(regions):
        def is_not_outlier(rect):
            width, height = rect[2], rect[3]
            if width < 15 and height < 15: # very small, resizing may cause distortion
                return False
            if width > 300 or height > 300: # very large, maybe abnormal thresholding
                return False
            return True
        return filter(is_not_outlier, regions)

    def run(self, input):
        origin, image = input
        ret, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = Scissor.remove_outliers(map(cv2.boundingRect, contours))
        return origin, image, list(filtered)

def deskew(img, size):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * size * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (size, size), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def padding(roi, width, height, ratio=1.3):
    side_length = int(ratio * max(width, height))
    top = int(side_length - height) // 2
    bottom = side_length - height - top
    left = int(side_length - width) // 2
    right = side_length - width - left
    return cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT, 255)

def normalize(roi, width, height):
    roi = padding(roi, width, height)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = deskew(roi, 28)
    roi = cv2.dilate(roi, (3, 3), iterations=2)
    return roi

class Recognizer(Stage):
    def __init__(self, method):
        super().__init__('recognizer')
        self.predictor = select_predictor(method)

    def run(self, input):
        origin, image, regions = input
        results = []
        for rect in regions:
            x, y, w, h = rect
            roi = image[y:y + h, x:x + w]
            roi = normalize(roi, w, h)
            result = self.predictor.predict(roi)
            results.append(int(result))
        return origin, image, regions, results

class Marker(Stage):
    def __init__(self):
        super().__init__('marker')

    def run(self, input):
        RECT_COLOR = (31, 16, 255)
        TEXT_COLOR = (31, 16, 255)
        origin, image, regions, results = input
        for rect, result in zip(regions, results):
            x, y, w, h = rect
            cv2.rectangle(origin, (x, y), (x + w, y + h), RECT_COLOR, 1)
            cv2.putText(origin, str(int(result)), (x + w // 2, y), cv2.FONT_HERSHEY_PLAIN, 2, TEXT_COLOR, 2)
        return origin

def parse_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--method', dest='method')
    parser.add_argument('--show-digits', dest='show_digits', action='store_true')
    return parser.parse_args()

def launch(pipeline, input):
    result = input
    for stage in pipeline:
        print('Running ' + stage.name, end='')
        start_time = time.perf_counter()
        result = stage.run(result)
        end_time = time.perf_counter()
        print(' (elapsed %f seconds)' % (end_time - start_time))
    return result

if __name__ == '__main__':
    args = parse_argv()
    pipeline = [
        Loader(args.image),
        Thresholding(),
        Scissor(),
        Recognizer(args.method),
        Marker()
    ]
    result = launch(pipeline, None)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('result', result)
    cv2.waitKey()
