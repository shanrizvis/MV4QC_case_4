import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

DIRECTORY = "resources/12.Automatic_PDF"

PATH_FULL = f"{DIRECTORY}/positive.jpg"
PATH_FULL_NEGATIVE = f"{DIRECTORY}/negative.jpg"
JSON_FILE = open(f"{DIRECTORY}/specs.json")

SPECS = json.load(JSON_FILE)

PATH_TEMPLATE = f"{DIRECTORY}/template.jpg"

TEMPLATE_METHODS = [cv2.TM_SQDIFF_NORMED, cv2.TM_SQDIFF, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED]


def read_image(path):
    return cv2.imread(path)


def resize(img, scale=20):
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA if scale > 100 else cv2.INTER_LINEAR)


def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def template_match(img, template, method_index=2):
    method = TEMPLATE_METHODS[method_index]
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_loc
    return top_left, template.shape[1], template.shape[0]


def convert_coords(start_point, w, h):
    y_start = start_point[1] - SPECS["y_padding"][0]
    x_start = start_point[0] - SPECS["x_padding"][0]
    y_end = start_point[1] + h + SPECS["y_padding"][1]
    x_end = start_point[0] + w + SPECS["x_padding"][1]
    return x_start, x_end, y_start, y_end


def grab_color(img):
    lower = np.array(SPECS["hsv_lower"])
    upper = np.array(SPECS["hsv_uper"])
    mask = cv2.inRange(img, lower, upper)
    return cv2.bitwise_and(img, img, mask=mask)


def check_if_black(img):
    return cv2.countNonZero(to_gray(img)) == 0


def merge_horizontal(img, img_2, title):
    res = np.concatenate((img, img_2), axis=1)
    cv2.putText(res, title, org=(100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0),
                thickness=3)
    return res


def merge_vertical(img, img_2):
    res = np.concatenate((img, img_2), axis=2)
    return res


def show_multiple_images(images, rows=2, cols=2):
    fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(10, 6), constrained_layout=True)
    for ind, content in enumerate(images):
        axes.ravel()[ind].imshow(cv2.cvtColor(content["image"], cv2.COLOR_BGR2RGB))
        axes.ravel()[ind].set_title(content["title"])
        axes.ravel()[ind].set_axis_off()
    fig.tight_layout()
    plt.show()


def display_image():
    img = read_image(PATH_FULL)
    img_negative = read_image(PATH_FULL_NEGATIVE)

    positives = [{"title": "positive", "image": img}]
    negatives = [{"title": "negative", "image": img_negative}]

    template = to_gray(read_image(PATH_TEMPLATE))
    gray = to_gray(img)
    gray_negative = to_gray(img_negative)

    positives.append({"title": "gray_positive", "image": gray})
    negatives.append({"title": "gray_negative", "image": gray_negative})

    start_point, w, h = template_match(gray, template)
    x_start, x_end, y_start, y_end = convert_coords(start_point, w, h)
    output = img[y_start: y_end, x_start:x_end]

    start_point, w, h = template_match(gray_negative, template)
    x_start, x_end, y_start, y_end = convert_coords(start_point, w, h)
    output_negative = img_negative[y_start: y_end, x_start:x_end]

    positives.append({"title": "zoomed_positive", "image": output})
    negatives.append({"title": "zoomed_negative", "image": output_negative})

    color_positive = grab_color(output)
    color_negative = grab_color(output_negative)

    positives.append({"title": "filtered_positive", "image": color_positive})
    negatives.append({"title": "filtered_negative", "image": color_negative})

    show_multiple_images(positives + negatives, cols=int(len(positives + negatives) / 2))


def main():
    display_image()


if __name__ == "__main__":
    main()
