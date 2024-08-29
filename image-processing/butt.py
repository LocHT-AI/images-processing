import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import cv2
def butterworth_notch_reject_filter(shape, d0, u_k, v_k):
    P, Q = shape
    H = np.ones((P, Q))

    for u in range(P):
        for v in range(Q):
            D_uv = np.sqrt((u - P / 2 + u_k) ** 2 + (v - Q / 2 + v_k) ** 2)
            D_muv = np.sqrt((u - P / 2 - u_k) ** 2 + (v - Q / 2 - v_k) ** 2)

            if D_uv <= d0 or D_muv <= d0:
                H[u, v] = 0.0

    return H

# Load the image
image = plt.imread('s.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform Fourier transform
f = fftpack.fft2(image)
fshift = fftpack.fftshift(f)

# Define the parameters for the Butterworth notch reject filter
d0 = 50
u_k = 200
v_k = 200

# Create the Butterworth notch reject filter transfer function
H = butterworth_notch_reject_filter(fshift.shape, d0, u_k, v_k)

# Apply the filter to the Fourier transform
filtered_fshift = fshift * H

# Perform inverse Fourier transform
filtered_f = fftpack.ifftshift(filtered_fshift)
filtered_image = fftpack.ifft2(filtered_f).real

# Display the original and filtered images
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')

plt.show()