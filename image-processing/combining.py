import cv2
import matplotlib.pyplot as plt

# Load the original image
img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
laplacian_img = cv2.Laplacian(img, cv2.CV_64F)
laplacian_img = laplacian_img.astype(img.dtype)

sharpened_img = cv2.add(img, laplacian_img)
# Calculate the Sobel gradient
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_grad = cv2.magnitude(sobel_x, sobel_y)


# Display the original image
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

# Calculate the Laplacian

# Display the Laplacian image
plt.subplot(2, 2, 2)
plt.imshow(laplacian_img, cmap='gray')
plt.title('Laplacian of Image')

plt.subplot(2, 2, 3)
plt.imshow(sharpened_img, cmap='gray')
plt.title('Sharpened Image')
# Display the Sobel gradient image
plt.subplot(2, 2, 4)
plt.imshow(sobel_grad, cmap='gray')
plt.title('Sobel Gradient of Image')
plt.show()
