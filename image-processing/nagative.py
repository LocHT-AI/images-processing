import cv2

img = cv2.imread('1.jpg')
negative_img = 255 - img
# cv2.imwrite('negative_image.jpg', negative_img)
cv2.imshow('Original Image', img)
cv2.imshow('Negative Image', negative_img)

# Wait for a key event and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()