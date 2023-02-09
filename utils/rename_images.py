import cv2
import os

DIR_NAMES = [
    "Direction_indicator_left",
    "Differential",
    "Forward Traction",
    "lights",
    "Signalisation_one",
    "Signalisation_two",
    "Forward_suspention",
    "Battery_indicator",
    "direction_indicator_right",
    "Preheating",
    "Oil_Pressure",
    "Automatic_PDF",
]

IMAGE_NAMES = ["postive", "negative", "template"]

SCALE = 20


def resize(img):
    width = int(img.shape[1] * SCALE / 100)
    height = int(img.shape[0] * SCALE / 100)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA if SCALE > 100 else cv2.INTER_LINEAR)


for i, n in enumerate(DIR_NAMES):
    full_dir = f"resources/{i + 1}.{n}"
    names = [f"{full_dir}/{n}" for n in os.listdir(full_dir)]
    for im in names:
        if "json" in im:
            continue

        img = cv2.imread(im)
        if img is None:
            continue
        print("Reading " + im)
        name = "foo"
        busy = True
        while busy:
            cv2.imshow(n, resize(img))
            delay = 1
            if cv2.waitKey(delay) & 0xFF == ord('p'):
                name = "positive"
                busy = False
            if cv2.waitKey(delay) & 0xFF == ord('n'):
                name = "negative"
                busy = False
            if cv2.waitKey(delay) & 0xFF == ord('t'):
                name = "template"
                busy = False

        cv2.destroyAllWindows()
        os.rename(im, f"{full_dir}/{name}.jpg")
        print(f"Renamed {im} to {full_dir}/{name}.jpg")
        print("-----------------")
    print("==============================")
