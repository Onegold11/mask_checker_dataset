import tensorflow as tf
import keras

# 모델 경로
MODEL_FINAL_PATH = './models/'
# 모델 이름
MODEL_NAME = 'mask_detection_v3_binary.h5'
# TFLite 파일 저장 경로
TFLITE_PATH = './models/TFLite/'
# TFLite 모델 이름
TFLITE_NAME = 'MobileNet_binary_20_30.tflite'

if __name__ == "__main__":
    # 모델 불러오기
    model = keras.models.load_model(MODEL_FINAL_PATH + MODEL_NAME)

    # TFLite로 변환
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # TFLite 파일로 저장
    with tf.io.gfile.GFile(TFLITE_PATH + TFLITE_NAME, 'wb') as f:
        f.write(tflite_model)