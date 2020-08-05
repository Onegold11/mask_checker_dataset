import os
import cv2
import dlib
import numpy as np

def make_image_directory(name):
    dir_path = "../images/" + name
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_face():
    #make_image_directory("mask_converted")
    #detect_faces("../new_image/")
    detect_faces("../new_image/no_mask/")




def detect_faces(dir_path):
    file_list = os.listdir(dir_path)

    imgNum = 0

    for image_name in file_list:

        if image_name[-8:-4] != '0001':
            continue

        # 얼굴 디텍터 모듈 초기화
        detector = dlib.get_frontal_face_detector()
        # 얼굴 특징점 모듈 초기화
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


        img = cv2.imread(dir_path + image_name)
        try:
            faces = detector(img)
        except:
            print("왜없어")
            continue


        face = faces[0]

        dlib_shape = predictor(img, face)


        # 68개 파트에 동그라미
#        shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

        # 얼굴에 사각형 그려줌
        # visualize
        img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()),
                         color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # 눈 코입에 동그라미 그려주는 것
#        for s in shape_2d:
#            cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        cv2.imshow("Image view", img)

        cropped = img[face.top():face.bottom(), face.left():face.right()]
        cv2.imwrite(str(imgNum) + ".jpg", cropped)
        imgNum += 1



        input = cv2.waitKey(0)
        if input == 120 or input == 88:
            cv2.destroyAllWindows()
            break

        cv2.destroyAllWindows()
        # image = cv2.imread(dir_path + image_name)
        # faces, confidences = cv.detect_face(image)
        #
        # for face in faces:
        #     (startX, startY) = face[0], face[1]
        #     (endX, endY) = face[2], face[3]
        #     cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 2)
        #
        # cv2.imshow("Image view", image)
        #
        # input = cv2.waitKey(0)
        # if input == 120 or input == 88:
        #     cv2.destroyAllWindows()
        #     break
        #
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    get_face()