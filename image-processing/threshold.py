import cv2

# Load the image
image = cv2.imread("5.jpg", 0)  # 0 flag loads the image in grayscale

# Apply binary thresholding
_, thresholded_image = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)

# Display the original and thresholded images
cv2.imshow("Original Image", image)
cv2.imshow("Thresholded Image", thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()