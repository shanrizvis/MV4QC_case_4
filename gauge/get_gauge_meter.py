import cv2 as cv
import numpy as np
import math
from utils import cv_utils as utils

img = utils.read_image("./temp.jpg")

blur = cv.GaussianBlur(img, (1, 1), 0)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

edges = cv.Canny(img, 1, 200)

cv.namedWindow("Trackbars")
cv.resizeWindow("Trackbars", 650, 240)
cv.createTrackbar("P1_x", "Trackbars", 1, 360, lambda x: None)
cv.createTrackbar("P1_y", "Trackbars", 110, 360, lambda x: None)


def show():
    X = cv.getTrackbarPos("P1_x", "Trackbars")
    y = cv.getTrackbarPos("P1_y", "Trackbars")

    X = 1 if X < 1 else X
    hough_lines = cv.HoughLinesP(edges, X, np.pi / 180, y, None, 0, 0)

    out = img.copy()
    if hough_lines is not None:
        for i in range(0, len(hough_lines)):
            line = hough_lines[i][0]
            cv.line(out, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 3, cv.LINE_AA)

    utils.display_image(edges, "edges")
    utils.display_image(out, "lines")


def main():
    utils.display_loop(show)


if "__main__" == __name__:
    main()
