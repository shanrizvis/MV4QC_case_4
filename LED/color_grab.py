import cv2
import numpy as np
import json

from utils import cv_utils as utils

DIRECTORY = "resources/11.Oil_Pressure"

PATH_FULL = f"{DIRECTORY}/positive.jpg"
PATH_FULL_NEGATIVE = f"{DIRECTORY}/negative.jpg"
SCALE = 20

PATH_TEMPLATE = f"{DIRECTORY}/template.jpg"

TEMPLATE_METHODS = [cv2.TM_SQDIFF_NORMED, cv2.TM_SQDIFF, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED]
JSON_FILE = open(f"{DIRECTORY}/specs.json")
SPECS = json.load(JSON_FILE)

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 650, 240)
cv2.createTrackbar("Hue min", "Trackbars", 0, 360, lambda x: None)
cv2.createTrackbar("Hue max", "Trackbars", 360, 360, lambda x: None)
cv2.createTrackbar("sat min", "Trackbars", 0, 255, lambda x: None)
cv2.createTrackbar("sat max", "Trackbars", 255, 255, lambda x: None)
cv2.createTrackbar("val min", "Trackbars", 0, 255, lambda x: None)
cv2.createTrackbar("val max", "Trackbars", 255, 255, lambda x: None)


def convert_coords(start_point, w, h):
    y_start = start_point[1] - SPECS["y_padding"][0]
    x_start = start_point[0] - SPECS["x_padding"][0]
    y_end = start_point[1] + h + SPECS["y_padding"][1]
    x_end = start_point[0] + w + SPECS["x_padding"][1]
    return x_start, x_end, y_start, y_end


def grab_color(img):
    hue_min = cv2.getTrackbarPos("Hue min", "Trackbars")
    hue_max = cv2.getTrackbarPos("Hue max", "Trackbars")
    sat_min = cv2.getTrackbarPos("sat min", "Trackbars")
    sat_max = cv2.getTrackbarPos("sat max", "Trackbars")
    val_min = cv2.getTrackbarPos("val min", "Trackbars")
    val_max = cv2.getTrackbarPos("val max", "Trackbars")

    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])
    mask = cv2.inRange(img, lower, upper)

    return cv2.bitwise_and(img, img, mask=mask)


def main():
    img = utils.read_image(PATH_FULL)
    img_negative = utils.read_image(PATH_FULL_NEGATIVE)

    template = utils.to_gray(utils.read_image(PATH_TEMPLATE))

    gray = utils.to_gray(img)
    gray_negative = utils.to_gray(img_negative)

    start_point, w, h = utils.template_match(gray, template)
    x_start, x_end, y_start, y_end = convert_coords(start_point, w, h)
    output = img[y_start: y_end, x_start:x_end]

    start_point, w, h = utils.template_match(gray_negative, template)
    x_start, x_end, y_start, y_end = convert_coords(start_point, w, h)
    output_negative = img_negative[y_start: y_end, x_start:x_end]

    def show_func():
        utils.display_image(grab_color(output), "output MASKED")
        utils.display_image(grab_color(output_negative), "output negative MASKED")

    utils.display_loop(show_func)


if __name__ == "__main__":
    main()
