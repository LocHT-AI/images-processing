import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def apply_gaussian_noise(image, mean, std_dev):
    """
    Apply Gaussian noise to an image.

    Parameters:
    - image: Input image.
    - mean: Mean of the Gaussian noise.
    - std_dev: Standard deviation of the Gaussian noise.

    Returns:
    - noisy_image: Image with applied Gaussian noise.
    """
    # Generate Gaussian noise
    rows, cols = image.shape
    gaussian_noise = np.random.normal(mean, std_dev, (rows, cols))

    # Add Gaussian noise to the image
    noisy_image = np.clip(image + gaussian_noise, 0, 255).astype(np.uint8)

    return noisy_image

def arithmetic_mean_filter(image, kernel_size):
    """
    Apply arithmetic mean filter to an image.

    Parameters:
    - image: Input image.
    - kernel_size: Size of the square kernel.

    Returns:
    - filtered_image: Image with arithmetic mean filter applied.
    """
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image


# def arithmetic_mean_filter(image, kernel_size):
#     rows, cols = image.shape
#     pad_size = kernel_size // 2
#     padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT)
#     filtered_image = np.zeros_like(image, dtype=np.float32)
    
#     for i in range(pad_size, rows + pad_size):
#         for j in range(pad_size, cols + pad_size):
#             roi = padded_image[i - pad_size: i + pad_size + 1, j - pad_size: j + pad_size + 1]
#             filtered_image[i - pad_size, j - pad_size] = np.sum(roi) / (kernel_size * kernel_size)
    
#     return filtered_image.astype(np.uint8)

def geometric_mean_filter(image, kernel_size):
    rows, cols = image.shape
    padding = kernel_size // 2
    # pad_size = kernel_size // 2

    image_array = np.array(image)
    # padding = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT)
    filtered_image = np.zeros_like(image_array, dtype=np.float32)
    for i in range(padding, image_array.shape[0] - padding):
        for j in range(padding, image_array.shape[1] - padding):
            # Extract local neighborhood
            neighborhood = image_array[i - padding:i + padding + 1, j - padding:j + padding + 1]
            filtered_image[i, j] = np.exp(np.mean(np.log(neighborhood + 1)))


    restored_image_array = np.clip(filtered_image, 0, 255).astype(np.uint8)

    # Convert NumPy array back to image
    restored_image = Image.fromarray(restored_image_array)
    return restored_image

def median_filter(image, kernel_size=3):
    filtered_image = np.zeros_like(image)
    border = kernel_size // 2
    
    # Iterate over each pixel in the image
    for y in range(border, image.shape[0] - border):
        for x in range(border, image.shape[1] - border):
            # Extract the neighborhood around the current pixel
            neighborhood = image[y - border:y + border + 1, x - border:x + border + 1]
            # Apply the median filter to the neighborhood
            median_value = np.median(neighborhood)
            # Set the filtered pixel value to the median
            filtered_image[y, x] = median_value
    
    return filtered_image
# Load an image
image = cv2.imread('4.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian noise
mean = 0
std_dev = 20
noisy_image = apply_gaussian_noise(image, mean, std_dev)


# Apply arithmetic mean filtering
arithmetic_filtered_image = arithmetic_mean_filter(noisy_image, 5)

# Apply geometric mean filtering
# geometric_filtered_image = geometric_mean_filter(noisy_image, 3)


geometric_filtered_image = median_filter(noisy_image, 5)


# Hiển thị ảnh gốc và không gian tần số
plt.subplot(221), plt.imshow(image, cmap='gray')
plt.title('Ảnh gốc'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(noisy_image, cmap='gray')
plt.title('Gaussian_noise'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(arithmetic_filtered_image, cmap='gray')
plt.title('arithmetic_filtered_image'), plt.xticks([]), plt.yticks([])


plt.subplot(224), plt.imshow(geometric_filtered_image, cmap='gray')
plt.title('geometric_filtered_image'), plt.xticks([]), plt.yticks([])


plt.show()

