from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import numpy as np

# 데이터 셋 경로
DATASET_PATH = './dataset/images.npy'
# 모델 파일 저장 경로
MODEL_PATH = './models/VGG16/'


def get_data_set():
    X_train, X_test, y_train, y_test = np.load(DATASET_PATH, allow_pickle=True)

    # 데이터 정규화(0~1)
    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256
    print('X_train shape:', X_train.shape)
    print('Y_train shape:', y_train.shape)
    print('X_test shape:', X_test.shape)
    print('Y_test shape:', y_test.shape)
    return X_train, X_test, y_train, y_test


def create_model(X_train, X_test, Y_train, Y_test):
    # VGG16 모델 생성
    transfer_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    transfer_model.trainable = False
    transfer_model.summary()

    # 모델 생성
    model = Sequential()

    # VGG16 모델 연결
    model.add(transfer_model)

    # 완전 연결 계층
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 모델 중간 세이브
    model_path = MODEL_PATH + '{epoch:02d}-{val_loss:.4f}.hdf5'
    check_pointer = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)

    # 조기 멈춤
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, mode='auto')

    # 학습
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=32, verbose=0,
                        callbacks=[early_stopping_callback, check_pointer])
    print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

    # 학습 과정 손실 값 그래프
    y_vloss = history.history['val_loss']
    y_loss = history.history['loss']

    x_len = np.arange(len(y_loss))
    plt.plot(x_len, y_vloss, marker='.', c='red', label='Testset_loss')
    plt.plot(x_len, y_loss, marker='.', c='blue', label='Trainset_loss')

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data_set()
    create_model(X_train, X_test, y_train, y_test)