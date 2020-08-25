from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import os

# 버전(v1, v2, v3)
VERSION = "v3"
# 이미지 원본 경로
SOURCE_PATH = "../../result/" + VERSION + "/"
# 이미지 타입
CLASSES = ["mask", "no_mask_2"]
# 이미지 크기
image_w = 64
image_h = 64


# mask, no_mask 이미지 파일을 배열로 변환
def get_data_set(x, y):
    for idx, type in enumerate(CLASSES):

        # 이미지 파일 목록 가져오기
        file_path = SOURCE_PATH + type + "/"
        files = os.listdir(file_path)

        # 이미지 타입을 원-핫 인코딩
        label = [0 for i in range(len(CLASSES))]
        label[idx] = 1

        for file in files:
            # 이미지 불러오기
            img = Image.open(file_path + file)
            # 이미지 크기 변경
            img = img.resize((image_w, image_h))
            # 배열로 변환
            data = np.asarray(img)

            x.append(data)
            y.append(label)


if __name__ == "__main__":
    x, y = [], []
    get_data_set(x, y)

    X = np.array(x)
    Y = np.array(y)

    # 학습 전용 데이터와 테스트 전용 데이터 구분
    print("학습, 테스트 데이터 분류...")
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    xy = (X_train, X_test, y_train, y_test)
    print("분류 완료")

    # 이미지 데이터 셋 저장
    print('데이터 저장중 ...')
    print(xy)

    np.save("./dataset/images_" + VERSION + ".npy", xy)
    print("저장 완료")
