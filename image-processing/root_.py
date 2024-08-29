import cv2
import numpy as np

# Load the image
img = cv2.imread('1.jpg', 0)  # Read the image in grayscale

# Define the root value
n = 10  # Change this value to the desired root value

# Apply the Nth Root Transformation
transformed_img = np.power(img, 1/n)

# Scale the transformed image to the range [0, 255]
transformed_img = np.uint8(255 * (transformed_img / np.max(transformed_img)))

# Save the transformed image
cv2.imwrite('transformed_img.jpg', transformed_img)
cv2.imshow("ori",img )
cv2.imshow("root_", transformed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()