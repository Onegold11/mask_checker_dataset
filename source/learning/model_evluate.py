from sklearn.model_selection import train_test_split
from PIL import Image
import keras
import numpy as np
import os

# 이미지 원본 경로
MODEL_PATH = "./mask_detection_v1.h5"
# 이미지 폴더
#IMAGE_PATH = "./validation/"
IMAGE_PATH = "../../result/v1/no_mask/"


files = os.listdir(IMAGE_PATH)
x = []
for file in files:
    # 이미지 불러오기
    img = Image.open(IMAGE_PATH + file)
    # 이미지 크기 변경
    img = img.resize((64, 64))
    # 배열로 변환
    data = np.asarray(img)
    x.append(data)

X = np.array(x)
img = Image.fromarray(X[0])
img.show()
print(X.shape)
model = keras.models.load_model(MODEL_PATH)
predictions = model.predict(X)
for i in range(len(predictions)):
    if np.argmax(predictions[i]) == 1:
        img = Image.fromarray(X[i])
        img.show()