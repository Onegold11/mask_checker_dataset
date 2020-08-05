import os
import cv2

size = 300
type = "mask"
source_path = "../images/" + type + "/"
result_path = "../resize/" + type + "/"


def resize():
    file_list = os.listdir(source_path)

    num = 0
    for image_name in file_list:
        img = cv2.imread(source_path + image_name)

        resize = cv2.resize(img, dsize=(300, 300), interpolation=cv2.INTER_AREA)
        cv2.imwrite(result_path + "no_mask_" + str(num) + ".png", resize)

        print(result_path + type + "_" + str(num) + ".png" + " complete")
        num += 1


if __name__ == "__main__":
    resize()
