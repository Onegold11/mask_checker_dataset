from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import os

source_path = "../../result/v1/"
classes = ["mask", "no_mask"]
image_w = 64
image_h = 64


# mask, no_mask 이미지 파일을 배열로 변환
def get_data_set(x, y):
    for idx, type in enumerate(classes):
        file_path = source_path + type + "/"
        files = os.listdir(file_path)

        label = [0 for i in range(len(classes))]
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


# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28, 28, 1), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# MODEL_DIR = './model/'
# if not os.path.exists(MODEL_DIR):
#     os.mkdir(MODEL_DIR)
#
# modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
# checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
# early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback, checkpointer])
# print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

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

    print('데이터 저장중 ...')
    print(xy)
    np.save("./dataset/images.npy", xy)
    print("저장 완료")