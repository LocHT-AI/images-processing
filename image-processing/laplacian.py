import cv2
import numpy as np
def laplacian(images):
    # Read the image
    img = cv2.imread(images,0)
    # Apply Laplacian filter
    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    # Convert the result to uint8
    laplacian = np.uint8(np.absolute(laplacian))

    # Display the image
    cv2.imshow('Laplacian', laplacian)
    cv2.imwrite('laplacian.jpg', laplacian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return laplacian