import cv2
# import imutils

# Initializing the HOG person
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


#FIXME: esi a et sharpeni methody
# kernel = np.array([[0, -1, 0],
#                    [-1, 5,-1],
#                    [0, -1, 0]])
# image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

# Reading the Image
# image = cv2.imread('Textures/11.jpg', 0)
image = cv2.resize(cv2.imread('Textures/11.jpg', 0), [1260, 914])

# Resizing the Image
# image = imutils.resize(image,
#                        width=min(500, image.shape[1]))

# Detecting all humans
(humans, _) = hog.detectMultiScale(image,
                                   winStride=(3, 3),
                                   padding=(3, 3),
                                   scale=1.21)
# getting no. of human detected
print('Human Detected : ', len(humans))

# Drawing the rectangle regions
for (x, y, w, h) in humans:
    cv2.rectangle(image, (x, y),
                  (x + w, y + h),
                  (0, 0, 255), 2)

# Displaying the output Image
cv2.imshow("Image", image)
cv2.waitKey(0)

cv2.destroyAllWindows()
