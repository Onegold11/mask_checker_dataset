from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, AveragePooling2D
from keras.callbacks import EarlyStopping
from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import BaseLogger
from keras import metrics
import matplotlib.pyplot as plt
import numpy as np

# 데이터 셋 경로
DATASET_PATH = './dataset/images.npy'
# 모델 중간 파일 저장 경로
MODEL_PATH = './models/'
# 모델 최종 파일 저장 경로
MODEL_FINAL_PATH = './models/'
# 모델 이름
MODEL_NAME = 'mask_detection.h5'
# 이미지 크기
image_w = 128
image_h = 128


def create_model(X_train, X_test, Y_train, Y_test):
    # 데이터 생성기
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # MobileNet 모델 생성
    transfer_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(image_w, image_h, 3))
    for layer in transfer_model.layers:
        layer.trainable = False

    # 모델 생성
    model = Sequential()

    # MobileNet 모델 연결
    model.add(transfer_model)

    # 완전 연결 계층
    # fc1
    model.add(AveragePooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    # fc2
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # fc3
    model.add(Dense(2, activation='softmax'))

    model.summary()
    opt = Adam(lr=1e-4, decay=1e-4 / 20)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=[metrics.binary_accuracy])

    # 조기 멈춤
    early_stopping_callback = EarlyStopping(monitor='loss', patience=2, mode='auto')
    # 학습
    history = model.fit(aug.flow(X_train, Y_train, batch_size=32), validation_data=(X_test, Y_test),
                        batch_size=32, epochs=20,
                        callbacks=[early_stopping_callback])
    print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

    # 학습 과정 손실 값 그래프
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    y_vloss = history.history['val_loss']
    y_loss = history.history['loss']

    print("acc_train\n{0}".format(acc))
    print("acc_test\n{0}".format(val_acc))
    print("loss_train\n{0}".format(y_loss))
    print("loss_test\n{0}".format(y_vloss))

    x_len = np.arange(len(y_loss))
    plt.plot(x_len, acc, marker='.', c='red', label='Trainset_acc')
    plt.plot(x_len, val_acc, marker='.', c='lightcoral', label='Testset_acc')
    plt.plot(x_len, y_vloss, marker='.', c='cornflowerblue', label='Testset_loss')
    plt.plot(x_len, y_loss, marker='.', c='blue', label='Trainset_loss')

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    # 모델 저장
    model.save(MODEL_FINAL_PATH + MODEL_NAME)


if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = np.load(DATASET_PATH, allow_pickle=True)
    print("{} {} {} {}".format(len(X_train), len(X_train[0]), len(X_train[0][0]), len(X_train[0][0][0])))
    create_model(X_train, X_test, Y_train, Y_test)
