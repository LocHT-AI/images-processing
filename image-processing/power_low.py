import cv2
import numpy as np

img = cv2.imread('3.jpg')

for gamma in [0.1, 0.5, 1.2, 2.2]:
    gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')
    # cv2.imwrite('gamma_transformed_image_' + str(gamma) + '.jpg', gamma_corrected)
    # cv2.imshow('gama Image', gamma_corrected)
    # cv2.imshow('Original Image', img)
    # Wait for a key event and close the windows
    # cv2.waitKey(0)
cv2.destroyAllWindows()