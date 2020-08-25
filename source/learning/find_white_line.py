from sklearn.model_selection import train_test_split
from PIL import Image
import keras
import numpy as np
import os

# 모델 경로
MODEL_FINAL_PATH = './models/final/MobileNet/'
# 모델 이름
MODEL_NAME = 'mask_detection_v3.h5'



# 저장 이미지 폴더
SAVE_IMAGE_PATH = '../../result/v3/no_mask/'

# 얼굴 이미지 폴더
FACE_IMAGE_PATH = '../../result/v2/no_mask/'

# 이미지 불러오기
def load_image(path):
    files = os.listdir(path)

    x = []
    num = 1
    for file in files:
        # 이미지 불러오기
        img = Image.open(path + file)

      #  img.show()

        datas = np.asarray(img)
        amen = datas[2:-2, 2:-2]

        im = Image.fromarray(amen)
       # im.show()

        for data in datas:
            print(data)

        im.save(SAVE_IMAGE_PATH + str(num) + ".png")
        num += 1

'''
        # 이미지 크기 변경
        img = img.resize((64, 64))
        # 배열로 변환
        data = np.asarray(img)
        data = data.astype("float") / 255
        x.append(data)

    x = np.asarray(x)
    return files, x
'''

if __name__ == "__main__":

    load_image(FACE_IMAGE_PATH)

#    mask_files, mask = load_mask_image()
#    face_files, face = load_face_image()

