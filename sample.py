#import the cv2 liberary for the 

import cv2
img = cv2.imread("dogesh.jpg")
gray = cv2.imread('dogesh.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)
cv2.imshow('gray',gray)
#Wait for any key before image disappears
cv2.waitKey(0)
cv2.destroyAllWindows()