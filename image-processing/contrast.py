import cv2
import numpy as np

def contrast_stretching(gray):
    # Convert image to grayscale
    
    
    # Calculate minimum and maximum pixel values
    min_val = np.min(gray)
    max_val = np.max(gray)
    # stretched = ()
    # Perform contrast stretching
    stretched = ((gray - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)
    
    # Convert back to BGR color space
    # stretched_bgr = cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)
    
    return stretched

# Load image
image = cv2.imread('lowcontrast.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply contrast stretching
_, thresholded_image = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
stretched_image= contrast_stretching(gray)


# Display original and stretched images
cv2.imshow('Original Image', gray)
cv2.imshow('Stretched Image', stretched_image)
# cv2.imshow('Stretched Image_2', stretched)
cv2.imshow('Thresh holding', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
