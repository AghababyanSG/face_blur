import cv2

img = cv2.imread('Textures/11.jpg', 0)

upperBody_cascade = cv2.CascadeClassifier('Haar_Files/haar_upperbody.xml')

arrUpperBody = upperBody_cascade.detectMultiScale(img)
if arrUpperBody != ():
    for (x, y, w, h) in arrUpperBody:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    print('body found')

cv2.imshow('image', cv2.resize(img, (1152, 822)))
cv2.waitKey(0)
cv2.destroyAllWindows()
