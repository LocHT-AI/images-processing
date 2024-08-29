import numpy as np
import cv2
import matplotlib.pyplot as plt

def add_pepper_noise(image, amount):
    h, w = image.shape
    num_pepper = int(amount * h * w)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    image[tuple(pepper_coords)] = 0
    return image

def contra_harmonic_mean_filter(image, kernel_size, Q):
    pad_size = kernel_size // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')

    filtered_image = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            numerator = np.sum(np.power(region, Q+1))
            denominator = np.sum(np.power(region, Q))
            filtered_image[i, j] = numerator / (denominator + 1e-6)

    return np.uint8(np.clip(filtered_image, 0, 255))

def add_salt_noise(image, amount):
    h, w = image.shape
    num_salt = int(amount * h * w)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    image[tuple(salt_coords)] = 255
    return image
# Example usage:
image = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)

# Add pepper noise to the image
noisy_image = add_pepper_noise(image.copy(), amount=0.1)
noisy_image_2 = add_salt_noise(image.copy(), amount=0.1)

# Applying the filter with kernel size 3 and Q=1.5
filtered_image = contra_harmonic_mean_filter(noisy_image, kernel_size=3, Q=1.5)
filtered_image_2 = contra_harmonic_mean_filter(noisy_image_2, kernel_size=3, Q=-1.5)


# # Displaying the original, noisy, and filtered images
# cv2.imshow('Original Image', image)
# cv2.imshow('Noisy Image', noisy_image)
# cv2.imshow('Filtered Image', filtered_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# Hiển thị ảnh gốc và không gian tần số
plt.subplot(221), plt.imshow(noisy_image, cmap='gray')
plt.title('Ảnh gốc'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(noisy_image_2, cmap='gray')
plt.title('noise'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(filtered_image, cmap='gray')
plt.title('peper'), plt.xticks([]), plt.yticks([])


plt.subplot(224), plt.imshow(filtered_image_2, cmap='gray')
plt.title('salt'), plt.xticks([]), plt.yticks([])


plt.show()
