import cv2
import numpy as np
from scipy import fftpack, ndimage
import matplotlib.pyplot as plt

def ideal_bandpass_filter(shape, lowcut, highcut):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - highcut:crow + highcut, ccol - highcut:ccol + highcut] = 1
    mask[crow - lowcut:crow + lowcut, ccol - lowcut:ccol + lowcut] = 0
    return mask

def gaussian_bandpass_filter(shape, lowcut, highcut):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    X, Y = np.meshgrid(x, y)
    g = np.exp(-((X ** 2 + Y ** 2) / (2 * (highcut / 2.77258872224) ** 2)))
    g *= 1 - np.exp(-((X ** 2 + Y ** 2) / (2 * (lowcut / 2.77258872224) ** 2)))
    return g

def butterworth_bandpass_filter(shape, lowcut, highcut, order=1):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    X, Y = np.meshgrid(x, y)
    d = np.sqrt(X**2 + Y**2)
    bpf = 1 / (1 + (d / highcut) ** (2 * order)) * (1 / (1 + (lowcut / d) ** (2 * order)))
    return bpf

# Read the image
image = cv2.imread('trangden.jpg', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('5.jpg', cv2.IMREAD_GRAYSCALE)



# Apply bandpass filters
shape = image.shape
lowcut = 30
highcut = 80
order = 2

ideal_mask = ideal_bandpass_filter(shape, lowcut, highcut)
gaussian_mask = gaussian_bandpass_filter(shape, lowcut, highcut)
butterworth_mask = butterworth_bandpass_filter(shape, lowcut, highcut, order)


# Apply FFT
f_transform = fftpack.fft2(image)
f_shift = fftpack.fftshift(f_transform)


f_ideal = f_shift * ideal_mask
f_gaussian = f_shift * gaussian_mask
f_butterworth = f_shift * butterworth_mask

# Inverse FFT
ideal_image = np.abs(fftpack.ifft2(fftpack.ifftshift(f_ideal)))
gaussian_image = np.abs(fftpack.ifft2(fftpack.ifftshift(f_gaussian)))
butterworth_image = np.abs(fftpack.ifft2(fftpack.ifftshift(f_butterworth)))

# # Normalize images
# ideal_image = cv2.normalize(ideal_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# gaussian_image = cv2.normalize(gaussian_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# butterworth_image = cv2.normalize(butterworth_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# # # Display the images
# cv2.imshow('Original Image', image)
# cv2.imshow('Ideal Bandpass Filtered Image', ideal_image)
# cv2.imshow('Gaussian Bandpass Filtered Image', gaussian_image)
# cv2.imshow('Butterworth Bandpass Filtered Image', butterworth_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.subplot(221), plt.imshow(image, cmap='gray')
plt.title('Ảnh gốc'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(ideal_image, cmap='gray')
plt.title('ideal'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(gaussian_image, cmap='gray')
plt.title('gaussian'), plt.xticks([]), plt.yticks([])


plt.subplot(224), plt.imshow(butterworth_image, cmap='gray')
plt.title('butterworth_bandpass'), plt.xticks([]), plt.yticks([])


plt.show()