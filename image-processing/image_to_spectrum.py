import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
image = cv2.imread('4.jpg', cv2.IMREAD_GRAYSCALE)

# Thực hiện phép biến đổi Fourier
f_transform = np.fft.fft2(image)
f_shift = np.fft.fftshift(f_transform)

magnitude_spectrum = 20*np.log(np.abs(f_shift))

centered_spectrum = 20*np.log(np.abs(f_transform))

log_transformed_image = 20*np.log(magnitude_spectrum + 1)

# f_inverse = np.fft.ifftshift(f_shift)
# image_back = np.fft.ifft2(f_inverse)
# image_back = np.abs(image_back)

# Hiển thị ảnh gốc và không gian tần số
plt.subplot(221), plt.imshow(image, cmap='gray')
plt.title('Ảnh gốc'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Biến đổi Fourier'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(centered_spectrum, cmap='gray')
plt.title('Centered spectrum'), plt.xticks([]), plt.yticks([])


plt.subplot(224), plt.imshow(log_transformed_image, cmap='gray')
plt.title('Result after a log transformation'), plt.xticks([]), plt.yticks([])


plt.show()
