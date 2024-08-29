import cv2
import matplotlib.pyplot as plt
import numpy as np

def nth_power_transform(image, n):
    # Normalize the image to the range [0, 1]
    normalized_image = image.astype(float) / 255.0
    # Apply the Nth power transformation
    transformed_image = np.power(normalized_image, n)
    # Scale the transformed image back to the range [0, 255]
    scaled_image = (transformed_image * 255).astype(np.uint8)
    return scaled_image

img = cv2.imread('trangden.jpg', cv2.IMREAD_GRAYSCALE)


negative_img = 255 - img


n = 5  # Change this value to the desired root value

# Apply the Nth Root Transformation
transformed_img = np.power(img, 1/n)
# Scale the transformed image to the range [0, 255]
transformed_img = np.uint8(255 * (transformed_img / np.max(transformed_img)))


transformed_image = nth_power_transform(img, n)

# Display the original image
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

# Calculate the Laplacian

# Display the Laplacian image
plt.subplot(2, 2, 2)
plt.imshow(negative_img, cmap='gray')
plt.title('nagative of Image')

plt.subplot(2, 2, 3)
plt.imshow(transformed_img, cmap='gray')
plt.title('root Image')

plt.subplot(2, 2, 4)
plt.imshow(transformed_image, cmap='gray')
plt.title('power Gradient of Image')
plt.show()
