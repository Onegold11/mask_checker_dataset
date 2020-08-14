import os
import cv2

# 변환 크기
size = 300
# 이미지 타입
type = "mask"
# 이미지 원본 경로
source_path = "../images/" + type + "/"
# 변경 이미지 저장 경로
result_path = "../resize/" + type + "/"


def resize():
    # 이미지 목록 가져오기
    file_list = os.listdir(source_path)

    num = 0
    for image_name in file_list:
        # 이미지 읽기
        img = cv2.imread(source_path + image_name)

        # 이미지 크기 변환
        resize = cv2.resize(img, dsize=(300, 300), interpolation=cv2.INTER_AREA)

        # 이미지 저장
        cv2.imwrite(result_path + "no_mask_" + str(num) + ".png", resize)

        # 이미지 이름 출력
        print(result_path + type + "_" + str(num) + ".png" + " complete")
        num += 1


if __name__ == "__main__":
    resize()
