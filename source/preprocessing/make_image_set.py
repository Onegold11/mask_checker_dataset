from sklearn.model_selection import train_test_split
import numpy as np
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input

# 이미지 원본 경로
SOURCE_PATH = "C:/Users/ujini/Desktop/face-mask-detector/dataset/"
# 이미지 타입
CLASSES = ["mask", "no_mask"]
# 이미지 크기
image_w = 128
image_h = 128


# mask, no_mask 이미지 파일을 배열로 변환
def get_data_set(x, y):
    for idx, type1 in enumerate(CLASSES):

        # 이미지 파일 목록 가져오기
        file_path = SOURCE_PATH + type1 + "/"
        files = os.listdir(file_path)

        # 이미지 타입을 원-핫 인코딩
        label = [0 for i in range(len(CLASSES))]
        label[idx] = 1

        for file in files:
            # 이미지 불러오기
            img = load_img(file_path + file, target_size=(image_w, image_h))
            # 이미지 numpy 배열로 변환
            img = img_to_array(img)
            # 이미지 정규화 (-1 ~ 1)
            # x /= 127.5
            # x -= 1.
            img = preprocess_input(img)

            print("{} / {}".format(file, label))
            x.append(img)
            y.append(label)


if __name__ == "__main__":
    x, y = [], []
    get_data_set(x, y)
    
    X = np.array(x, dtype='float32')
    Y = np.array(y)

    # 학습 전용 데이터와 테스트 전용 데이터 구분
    print("학습, 테스트 데이터 분류...")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
    xy = (X_train, X_test, y_train, y_test)
    print("분류 완료")

    # 이미지 데이터 셋 저장
    print(xy[0][0][0])
    print(xy[2])

    print('데이터 저장중 ...')
    np.save("./dataset/images.npy", xy)
    print("저장 완료")
