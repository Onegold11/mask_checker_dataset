import os
import cv2
import dlib

def make_image_directory(name):
    dir_path = "../images/" + name
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_face():
    detect_faces("../image/no_mask/")

def detect_faces(dir_path):
    file_list = os.listdir(dir_path)
    result_path = "../result/no_mask_+20/"
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
            face = faces[0]
        except:
            print("no_face")
            continue

        # 얼굴에 사각형 그려줌
        img = cv2.rectangle(img, pt1=(face.left() - 30, face.top() - 50), pt2=(face.right() + 20, face.bottom() + 25),
                            color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        # 이미지 확인
        #cv2.imshow("Image view", img)

        # 사각형 얼굴 이미지만 저장
        cropped = img[face.top() - 49:face.bottom() + 24, face.left() - 29:face.right() + 19]
        print(result_path + str(imgNum) + ".jpg")
        try:
            cv2.imwrite(result_path + str(imgNum) + ".jpg", cropped)
        except:
            continue
        imgNum += 1

        input = cv2.waitKey(0)
        if input == 120 or input == 88:
            cv2.destroyAllWindows()
            break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    get_face()