import cv2 as cv
import numpy as np
import math
from utils import cv_utils as utils

img = utils.read_image("./temp.jpg")

out = img.copy()

blur = cv.GaussianBlur(img, (1, 1), 0)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

edges = cv.Canny(img, 1, 200)
hough_lines = cv.HoughLines(edges, 1, np.pi / 180, 110, None, 0, 0)

cv.namedWindow("Trackbars")
cv.resizeWindow("Trackbars", 650, 240)
cv.createTrackbar("P1_x", "Trackbars", 0, img.shape[0], lambda x: None)
cv.createTrackbar("P1_y", "Trackbars", img.shape[1], img.shape[1], lambda x: None)

polar_coords = []

if hough_lines is None:
    raise Exception("houghlines is None")

for i in range(0, len(hough_lines)):
    print(hough_lines[i])
    print("------------")

    # convert polar to cart
    rho = hough_lines[i][0][0]
    theta = hough_lines[i][0][1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    polar_coords.append((rho, theta))

    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    cv.line(out, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

print(polar_coords)

print(f"Angle 1 is then {math.atan(polar_coords[0][0] / 10)}")
print(f"Angle 2 is then {math.atan(polar_coords[0][1] / 10)}")

def show():
    utils.display_image(edges, "edges")
    utils.display_image(out, "lines")


def main():
    utils.display_loop(show)


if "__main__" == __name__:
    main()
