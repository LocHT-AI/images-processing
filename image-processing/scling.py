import cv2
import numpy as np

# Load the image
img = cv2.imread('1.jpg')

# Perform slicing transformation
sliced_img = img[100:300, 200:400]  # Specify the desired slice range

# Display the sliced image
cv2.imshow('Sliced Image', sliced_img)
cv2.waitKey(0)
cv2.destroyAllWindows()