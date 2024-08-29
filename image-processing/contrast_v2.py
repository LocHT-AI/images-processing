import cv2
import numpy as np

# Read the image
img1 = cv2.imread('4.jpg',0)
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# Create zeros array to store the stretched image
minmax_img = np.zeros((img1.shape[0],img1.shape[1]),dtype = 'uint8')
# print(range(img1.shape[0]))
# print(range(img1.shape[1]))

# Loop over the image and apply Min-Max formulae
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        minmax_img[i,j] = 255*(img1[i,j]-np.min(img1))/(np.max(img1)-np.min(img1))

# Displat the stretched image
cv2.imshow('Minmax',minmax_img)
cv2.waitKey(0)