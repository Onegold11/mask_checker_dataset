import os
import cv2

def make_image_directory(name):
    dir_path = "../images/" + name
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_face():
    #make_image_directory("mask_converted")
    #detect_faces("../new_image/")
    detect_faces("../images/mask/")


def detect_faces(dir_path):
    file_list = os.listdir(dir_path)

    for image_name in file_list:
        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

        img = cv2.imread(dir_path + image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Image view", img)

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