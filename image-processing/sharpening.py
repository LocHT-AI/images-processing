import cv2
import numpy as np
def sharpened(img):
    # Read the image
    img = cv2.imread(img)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian filter
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Add the original image and the Laplacian result
    sharpened = cv2.add(img, laplacian)

    # Display the image
    cv2.imshow('Sharpened', sharpened)
    cv2.imwrite('sharpened.jpg', sharpened)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return sharpened