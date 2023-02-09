from utils import cv_utils, cv_utils as utils

FUEL_PATH = "../resources/19.Fuel_Gauge/fuel_template.jpg"
TEMP_PATH = "../resources/20.Motor_temp/motor_temprature.jpg"

TEST_PATH = "../resources/19.Fuel_Gauge/IMG_20230113_073820.jpg"

PADDING = {
    "x_left": 101,
    "x_right": 575,
    "y_up": 275,
    "y_down": 100,
}


def main():
    template = utils.to_gray(utils.read_image(TEMP_PATH))

    test = utils.read_image(TEST_PATH)

    gray_test = utils.to_gray(test.copy())
    top, bottom = utils.get_square(*utils.template_match(gray_test, template))

    # out = test.copy()
    # cv_utils.draw_square(out, top, bottom)

    out = test[top[1] - PADDING["y_up"]: bottom[1] + PADDING["y_down"],
          top[0] - PADDING["x_left"]: bottom[0] + PADDING["x_right"]]

    def display_func():
        # utils.display_image(utils.resize(out))
        utils.display_image(out)

    utils.display_loop(display_func)
    import cv2
    cv2.imwrite("./temp2.jpg", out)


if __name__ == "__main__":
    main()
