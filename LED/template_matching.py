import cv2
from matplotlib import pyplot as plt
import numpy as np

DIRECTORY = "resources/1.Direction_indicator_left"

PATH_FULL = f"{DIRECTORY}/positive.jpg"
PATH_FULL_NEGATIVE = f"{DIRECTORY}/negative.jpg"

PATH_TEMPLATE = f"{DIRECTORY}/template.jpg"

SCALE = 30


def read_image(path):
    return cv2.imread(path)


def resize(img):
    width = int(img.shape[1] * SCALE / 100)
    height = int(img.shape[0] * SCALE / 100)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA if SCALE > 100 else cv2.INTER_LINEAR)


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def template_match(img, template, method=cv2.TM_CCORR_NORMED):
    w = template.shape[1]
    h = template.shape[0]

    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, 255, 2)


def template_match_testing(img2, template):
    w = template.shape[1]
    h = template.shape[0]
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    for meth in methods:
        img = img2.copy()
        method = eval(meth)
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img, top_left, bottom_right, 255, 2)

        cv2.imshow("template:  " + meth, resize(img))

    cv2.waitKey(0)


def display_image():
    positive_img = to_gray(read_image(PATH_FULL))
    negative_img = to_gray(read_image(PATH_FULL_NEGATIVE))
    template = to_gray(read_image(PATH_TEMPLATE))

    output_positive = positive_img.copy()
    output_negative = negative_img.copy()

    template_match(output_positive, template)
    template_match(output_negative, template)

    print(output_positive.shape)
    print(output_negative.shape)
    #template_match_testing(positive_img, template)
    busy = True
    while busy:
        cv2.imshow("output positive", resize(output_positive))
        cv2.imshow("output negative", resize(output_negative))

        # cv2.imshow("output MASKED", grab_color(output))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            busy = False
    cv2.destroyAllWindows()


def main():
    display_image()


if __name__ == "__main__":
    main()


