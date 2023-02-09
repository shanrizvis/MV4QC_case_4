import cv2
import json

specs = {
    "hsv_lower": [
        7,
        236,
        220
    ],
    "hsv_uper": [
        256,
        255,
        255
    ]}

DIRECTORY = "resources/12.Automatic_PDF"

PATH_FULL = f"{DIRECTORY}/positive.jpg"
PATH_FULL_NEGATIVE = f"{DIRECTORY}/negative.jpg"
SCALE = 20

PATH_TEMPLATE = f"{DIRECTORY}/template.jpg"

TEMPLATE_METHODS = [cv2.TM_SQDIFF_NORMED, cv2.TM_SQDIFF, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED]

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 650, 240)
cv2.createTrackbar("X left", "Trackbars", 0, 360, lambda x: None)
cv2.createTrackbar("X right", "Trackbars", 0, 360, lambda x: None)
cv2.createTrackbar("Y up", "Trackbars", 0, 360, lambda x: None)
cv2.createTrackbar("Y down", "Trackbars", 5, 360, lambda x: None)


def read_image(path):
    return cv2.imread(path)


def resize(img):
    width = int(img.shape[1] * SCALE / 100)
    height = int(img.shape[0] * SCALE / 100)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA if SCALE > 100 else cv2.INTER_LINEAR)


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
    X_left = cv2.getTrackbarPos("X left", "Trackbars")
    X_right = cv2.getTrackbarPos("X right", "Trackbars")
    Y_left = cv2.getTrackbarPos("Y up", "Trackbars")
    Y_right = cv2.getTrackbarPos("Y down", "Trackbars")

    y_start = start_point[1] - Y_left
    x_start = start_point[0] - X_left
    y_end = start_point[1] + h + Y_right
    x_end = start_point[0] + w + X_right
    return x_start, x_end, y_start, y_end


def check_if_black(img):
    return cv2.countNonZero(to_gray(img)) == 0


def display_image():
    img = read_image(PATH_FULL)
    img_negative = read_image(PATH_FULL_NEGATIVE)
    template = to_gray(read_image(PATH_TEMPLATE))

    gray = to_gray(img)
    gray_negative = to_gray(img_negative)

    pos_start_point, pos_w, pos_h = template_match(gray, template)
    neg_start_point, neg_w, neg_h = template_match(gray_negative, template)

    busy = True
    while busy:
        x_start, x_end, y_start, y_end = convert_coords(pos_start_point, pos_w, pos_h)
        output = img[y_start: y_end, x_start:x_end]

        x_start, x_end, y_start, y_end = convert_coords(neg_start_point, neg_w, neg_h)
        output_negative = img_negative[y_start: y_end, x_start:x_end]

        cv2.imshow("output positive", output)
        cv2.imshow("output negative", output_negative)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            busy = False

    X_left = cv2.getTrackbarPos("X left", "Trackbars")
    X_right = cv2.getTrackbarPos("X right", "Trackbars")
    Y_left = cv2.getTrackbarPos("Y up", "Trackbars")
    Y_right = cv2.getTrackbarPos("Y down", "Trackbars")
    cv2.destroyAllWindows()

    specs["x_padding"] = [X_left, X_right]
    specs["y_padding"] = [Y_left, Y_right]
    json_obj = json.dumps(specs)

    with open(f"{DIRECTORY}/specs.json", "w+") as out:
        out.write(json_obj)


def main():
    display_image()


if __name__ == "__main__":
    main()
