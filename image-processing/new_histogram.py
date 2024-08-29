import cv2
import numpy as np
import matplotlib.pyplot as plt

def match_histogram(source, target):
    # Calculate histograms
    source_hist = cv2.calcHist([source], [0], None, [256], [0, 256])
    target_hist = cv2.calcHist([target], [0], None, [256], [0, 256])

    # Normalize histograms
    source_hist /= source.size
    target_hist /= target.size

    # Compute cumulative distribution functions (CDF)
    source_cdf = np.cumsum(source_hist)
    target_cdf = np.cumsum(target_hist)

    # Map pixel values from source to target
    mapping = np.interp(source_cdf, target_cdf, range(256)).astype(np.uint8)

    # Apply mapping to the target image
    matched_image = mapping[target]

    return matched_image

# Read the source and target images
# source_path = 'gray.jpeg'  # Replace with the actual path to your source image
# target_path = 'xam.jpg'  # Replace with the actual path to your target image
source_path = '4.jpg'  # Replace with the actual path to your source image
target_path = '5.jpg'  # Replace with the actual path to your target image

source_image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
target_image = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

# Match histograms
matched_image = match_histogram(source_image, target_image)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)), plt.title('Source Image')
plt.subplot(1, 3, 2), plt.imshow(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)), plt.title('Target Image')
plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)), plt.title('Matched Image')

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1), plt.imshow(source_image, cmap='gray'), plt.title('Source Image')
plt.subplot(1, 3, 2), plt.imshow(target_image, cmap='gray'), plt.title('Target Image')
plt.subplot(1, 3, 3), plt.imshow(matched_image, cmap='gray'), plt.title('Matched Image')

plt.show()