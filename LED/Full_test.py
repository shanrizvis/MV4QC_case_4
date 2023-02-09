import cv2
import numpy as np
import json
from utils import cv_utils as utils

DIRECTORY = "resources/12.Automatic_PDF"

PATH_FULL = f"{DIRECTORY}/positive.jpg"
PATH_FULL_NEGATIVE = f"{DIRECTORY}/negative.jpg"
JSON_FILE = open(f"{DIRECTORY}/specs.json")

SPECS = json.load(JSON_FILE)
print(SPECS)

PATH_TEMPLATE = f"{DIRECTORY}/template.jpg"

TEMPLATE_METHODS = [cv2.TM_SQDIFF_NORMED, cv2.TM_SQDIFF, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED]


def convert_coords(start_point, w, h):
    y_start = start_point[1] - SPECS["y_padding"][0]
    x_start = start_point[0] - SPECS["x_padding"][0]
    y_end = start_point[1] + h + SPECS["y_padding"][1]
    x_end = start_point[0] + w + SPECS["x_padding"][1]
    return x_start, x_end, y_start, y_end


def draw_region_of_interest(image, top_left, bottom_right):
    img = image.copy()
    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    return img


def merge_horizontal(img, img_2, title):
    res = np.concatenate((img, img_2), axis=1)
    cv2.putText(res, title, org=(100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0),
                thickness=3)
    return res


def merge_vertical(img, img_2):
    res = np.concatenate((img, img_2), axis=2)
    return res


def main():
    img = utils.read_image(PATH_FULL)
    img_negative = utils.read_image(PATH_FULL_NEGATIVE)

    # todo blur image?

    template = utils.to_gray(utils.read_image(PATH_TEMPLATE))
    gray = utils.to_gray(img)
    gray_negative = utils.to_gray(img_negative)

    start_point, w, h = utils.template_match(gray, template)
    x_start, x_end, y_start, y_end = convert_coords(start_point, w, h)
    output = img[y_start: y_end, x_start:x_end]

    start_point, w, h = utils.template_match(gray_negative, template)
    x_start, x_end, y_start, y_end = convert_coords(start_point, w, h)
    output_negative = img_negative[y_start: y_end, x_start:x_end]

    color_positive = utils.grab_color(output, np.array(SPECS["hsv_lower"]), np.array(SPECS["hsv_uper"]))
    color_negative = utils.grab_color(output_negative, np.array(SPECS["hsv_lower"]), np.array(SPECS["hsv_uper"]))

    print(f"{PATH_FULL} is a {'black image' if utils.is_image_black(color_positive) else 'not black'}")
    print(f"{PATH_FULL_NEGATIVE} is a {'black image' if utils.is_image_black(color_negative) else 'not black'}")

    def show_func():
        utils.display_image(color_positive, "output MASKED")
        utils.display_image(color_negative, "output negative MASKED")

    utils.display_loop(show_func)


if __name__ == "__main__":
    main()
