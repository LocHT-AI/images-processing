import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('4.jpg',cv2.IMREAD_GRAYSCALE)

# Define the kernel size for Gaussian blur (should be an odd number)
kernel_size = (5, 5)  # Adjust this according to the blurring effect you want
kernel_size2 = (11, 11)  # Adjust this according to the blurring effect you want
# kernel_size3 = (22, 22)  # Adjust this according to the blurring effect you want


# Apply Gaussian blur to the image
blurred_image = cv2.GaussianBlur(image, kernel_size, sigmaX=0)
blurred_image2 = cv2.GaussianBlur(image, kernel_size2, sigmaX=0)
# blurred_image3 = cv2.GaussianBlur(image, kernel_size3, sigmaX=0)


plt.subplot(221), plt.imshow(image, cmap='gray')
plt.title('Ảnh gốc'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(blurred_image, cmap='gray')
plt.title('Biến đổi Fourier'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(blurred_image2, cmap='gray')
plt.title('Centered spectrum'), plt.xticks([]), plt.yticks([])


# plt.subplot(224), plt.imshow(blurred_image2, cmap='gray')
# plt.title('Result after a log transformation'), plt.xticks([]), plt.yticks([])


plt.show()

