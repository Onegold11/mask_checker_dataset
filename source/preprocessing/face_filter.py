import os
import cv2

# 이미지 타입, ! type 확인 !
type = "mask"
# 이미지 원본 경로(변경 해야함)
source_path = "C:/Users/ujini/Desktop/mask_checker_dataset-master/images/mixed/"
# 이미지 저장 경로(변경 해야함)
result_path = "../../images/result2/"


def get_face():
    detect_faces(source_path)


def detect_faces(dir_path):
    # 이미지 목록 가져오기
    file_list = os.listdir(dir_path)

    num = 0
    for image_name in file_list:
        # 얼굴 인식 모델 생성
        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

        # 이미지 불러오기
        img = cv2.imread(dir_path + image_name)
        # 이미지 흑백으로 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 이미지에서 얼굴 위치 추출
        faces = face_cascade.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            # 이미지에서 얼굴 영역만 추출
            cropped = img[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]

            try:
                # 이미지 저장, ! type 확인 !
                cv2.imwrite(result_path + type + "_" + str(num) + ".png", cropped)
            except:
                print("오류")
            # 이미지 이름 출력
            print(result_path + type + "_" + str(num) + ".png" + " complete")
            num += 1


if __name__ == "__main__":
    get_face()
