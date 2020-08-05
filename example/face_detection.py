import cv2
import dlib
import sys
import numpy as np

scaler = 0.7

# 얼굴 디텍터 모듈 초기화
detector = dlib.get_frontal_face_detector()
# 얼굴 특징점 모듈 초기화
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture('mycam.avi')

while True:
    ret, img = cap.read()
    if not ret:
        break

#    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler )))
    ori = img.copy()
    # detect faces
    faces = detector(img)

    try:
        face = faces[0]
    except:
        pass

    dlib_shape = predictor(img, face)

    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    # visualize
    img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()),
                        color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    for s in shape_2d:
        cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)


    cv2.imshow('Face', img)
    cv2.waitKey(1)
