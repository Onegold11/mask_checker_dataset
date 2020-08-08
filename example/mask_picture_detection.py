import os
import cv2
import dlib
import pandas as pd

def make_image_directory(name):
    dir_path = "../images/" + name
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_face():
    #make_image_directory("mask_converted")
    #detect_faces("../new_image/")
    detect_faces("../image/mask/train/")

def detect_faces(dir_path):
    file_list = os.listdir(dir_path)
    result_path = "../result/v2/mask/"
    imgNum = 0

    annotations = pd.read_csv(dir_path + file_list[-1])


    for image_name in file_list:


        img = cv2.imread(dir_path + image_name)

        face_pos = []

        for i in range(len(annotations)):
            if image_name == annotations['filename'][i] and annotations['class'][i] == 'mask':
                face_pos.append([annotations.iloc[i]['xmin'], annotations.iloc[i]['ymin'], annotations.iloc[i]['xmax'], annotations.iloc[i]['ymax']])

        for face in face_pos:
            print(face[0].item())
            print(face[1].item())
            print(face[2].item())
            print(face[3].item())

            cropped = img[face[1].item():face[3].item(), face[0].item():face[2].item()]

            try:
                resize = cv2.resize(cropped, dsize=(64, 64), interpolation=cv2.INTER_AREA)
            except:
                continue
            #cv2.imshow("image view", resize)

            cv2.imwrite(result_path + str(imgNum) + '.jpg', resize)

            imgNum += 1

            input = cv2.waitKey(0)
            if input == 120 or input == 88:
                cv2.destroyAllWindows()
                break

            cv2.destroyAllWindows()


if __name__ == "__main__":
    get_face()