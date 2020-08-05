import cv2
import numpy as np

def showImage():
    imgfile = '0022.jpg'
    #img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
    img = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)

#    cv2.namedWindow('0022', cv2.WINDOW_NORMAL)

    cv2.imshow('0022', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


showImage()