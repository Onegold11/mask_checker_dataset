import os
import cv2

type = "mask"
#source_path = "../../resize/" + type + "/"
#result_path = "../../resize/result/" + type + "/"
source_path = "C:/Users/ujini/Desktop/mask_checker_dataset-master/images/mixed/"
result_path = "../../images/result2/"


def get_face():
    detect_faces(source_path)


def detect_faces(dir_path):
    file_list = os.listdir(dir_path)

    num = 0
    for image_name in file_list:
        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

        img = cv2.imread(dir_path + image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cropped = img[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]

            print(len(cropped))
            try:
                #cv2.imwrite(result_path + type + "_" + str(num) + ".png", cropped)
                cv2.imwrite(result_path + str(num) + ".png", cropped)
            except:
                print("오류")
            print(result_path + type + "_" + str(num) + ".png" + " complete")
            num += 1


if __name__ == "__main__":
    get_face()
