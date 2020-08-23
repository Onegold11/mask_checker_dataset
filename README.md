Mask checker dataset
====================

Intro
-----
[MaskChecker][MaskChecker_Android] 앱에 사용할 데이터셋과 모델를 위한 저장소입니다.

[MaskChecker_Android]: https://github.com/Onegold11/MaskChecker_Android

Data Source
-----------
+ https://www.kaggle.com/wobotintelligence/face-mask-detection-dataset/data?select=train.csv
+ https://www.kaggle.com/atulanandjha/lfwpeople?select=lfw-funneled.tgz
+ https://public.roboflow.ai/object-detection/mask-wearing

Using Models
------------
+ VGG16
+ MobileNetV2

Required Version
-----------
+ Tensorflow 2.2.0: https://github.com/tensorflow/tensorflow
+ Keras 2.4.3: https://github.com/keras-team/keras
+ scikit-learn 0.23.2: https://github.com/scikit-learn/scikit-learn
+ numpy 1.19.0: https://github.com/numpy/numpy
+ dlib: https://github.com/davisking/dlib
+ opencv-python 4.3.0.36: https://github.com/skvark/opencv-python

Contents
--------
+ example
  - 테스트 코드 모음
+ image
  - 전처리 전 이미지 모음
  - mask : 마스크 이미지 모음
  - no_mask : 얼굴 이미지 모음
  - validation : 검증용 이미지 모음
+ result
  - 전처리 후 이미지 모음
+ source
  - 소스 코드 모음
  - learning : 모델 학습 소스 코드 몽므
  - preprocessing : 전처리 소스 코드 모음

Contributor
-----------
