from sklearn.model_selection import train_test_split
from PIL import Image
import keras
import numpy as np
import os

# 모델 경로
MODEL_FINAL_PATH = './models/'
# 모델 이름
MODEL_NAME = 'mask_detection_model.h5'
# 마스크 이미지 폴더
MASK_IMAGE_PATH = '../image/validation/mask/'
# 얼굴 이미지 폴더
FACE_IMAGE_PATH = '../image/validation/no_mask/'


# 얼굴 이미지 불러오기
def load_face_image():
    files, x = load_image(FACE_IMAGE_PATH)
    return files, x


# 마스크 이미지 불러오기
def load_mask_image():
    files, x = load_image(MASK_IMAGE_PATH)
    return files, x


# 이미지 불러오기
def load_image(path):
    files = os.listdir(path)

    x = []
    for file in files:
        # 이미지 불러오기
        img = Image.open(path + file)
        # 이미지 크기 변경
        img = img.resize((64, 64))
        # 배열로 변환
        data = np.asarray(img)
        data = data.astype("float") / 255
        x.append(data)

    x = np.asarray(x)
    return files, x


if __name__ == "__main__":
    mask_files, mask = load_mask_image()
    face_files, face = load_face_image()

    # 모델 불러오기
    model = keras.models.load_model(MODEL_FINAL_PATH + MODEL_NAME)
    print(MODEL_FINAL_PATH + MODEL_NAME)

    # 마스크 이미지 예측
    print(np.shape(mask))
    predictions = model.predict(mask)

    mask_count = 0
    for i in range(len(predictions)):
        print("==========")
        print("Name : {0}".format(mask_files[i]))
        print(predictions[i])
        print(np.argmax(predictions[i]))
        print("==========")
        if np.argmax(predictions[i]) == 0:
            mask_count += 1

    # 얼굴 이미지 예측
    print(np.shape(face))
    predictions = model.predict(face)
    face_count = 0
    for i in range(len(predictions)):
        print("==========")
        print("Name : {0}".format(face_files[i]))
        print(predictions[i])
        print(np.argmax(predictions[i]))
        print("==========")
        if np.argmax(predictions[i]) == 1:
            face_count += 1

    # 맞춘 개수
    print("mask : {0}, face : {1}".format(mask_count, face_count))