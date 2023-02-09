import cv2
import matplotlib.pyplot as plt


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


def template_match(img, template, method=cv2.TM_SQDIFF):
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_loc
    return top_left, template.shape[1], template.shape[0]


def grab_color(img, lower_lim, upper_lim):
    mask = cv2.inRange(img, lower_lim, upper_lim)
    return cv2.bitwise_and(img, img, mask=mask)


def get_square(start, width, height):
    return start, (start[0] + width, start[1] + height)


def draw_square(img, top_left, bottom_right):
    cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)


def display_image(img, title="output"):
    cv2.imshow(title, img)


def is_image_black(img):
    return cv2.countNonZero(to_gray(img)) == 0


def display_loop(display_func):
    busy = True
    while busy:
        display_func()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            busy = False
    cv2.destroyAllWindows()


def show_plt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
