import cv2
import numpy as np

def nth_power_transform(image, n):
    # Normalize the image to the range [0, 1]
    normalized_image = image.astype(float) / 255.0
    
    # Apply the Nth power transformation
    transformed_image = np.power(normalized_image, n)
    
    # Scale the transformed image back to the range [0, 255]
    scaled_image = (transformed_image * 255).astype(np.uint8)
    
    return scaled_image

# Load the image
image = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)

# Apply the Nth power transformation with n = 2
transformed_image = nth_power_transform(image, 10)

# Display the original and transformed images
cv2.imshow('Original Image', image)
cv2.imshow('Transformed Image', transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()