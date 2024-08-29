import cv2
import numpy as np
def sobel(images):
    # Read the image
    img = cv2.imread(images,0)

    # Apply Sobel filter
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the gradient magnitude
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Convert the result to uint8
    gradient_magnitude = np.uint8(gradient_magnitude)

    # Display the image
    cv2.imshow('Sobel Gradient', gradient_magnitude)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return gradient_magnitude