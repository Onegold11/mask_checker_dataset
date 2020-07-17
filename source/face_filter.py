import cv2

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_casecade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

img = cv2.imread('../images/no_mask/0227.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
eyes = eye_casecade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

for (ex, ey, ew, eh) in eyes:
    cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

cv2.imshow("Image view", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
