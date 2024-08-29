# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# from scipy import fftpack, ndimage


# # Load the image
# image = cv2.imread('b.jpg', 0)

# # Apply FFT
# f_transform = fftpack.fft2(image)
# f_shift = fftpack.fftshift(f_transform)

# spectrum = np.log(np.abs(f_shift))


# # Create a notch filter
# def create_notch_filter(shape, center, radius):
#     rows, cols = shape
#     crow, ccol = center
#     mask = np.ones((rows, cols), np.uint8)
#     for i in range(rows):
#         for j in range(cols):
#             if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) < radius:
#                 mask[i, j] = 0
#     return mask

# # Define the positions and radii of the notches
# notch_centers = [(110, 140), (130, 160)]  # Example notch center positions
# notch_radius = 2  # Example notch radius

# # Apply notch filters
# for center in notch_centers:
#     notch_filter = create_notch_filter(spectrum.shape, center, notch_radius)
#     f_shift = f_shift * notch_filter

# # Inverse FFT shift
# f_ishift = fftpack.ifftshift(f_shift)
# spectrum_2 = np.log(np.abs(f_shift))
# # Inverse FFT
# filtered_image = fftpack.ifft2(f_ishift)
# filtered_image = np.abs(filtered_image)

# plt.subplot(221), plt.imshow(image, cmap='gray')
# plt.title('Ảnh gốc'), plt.xticks([]), plt.yticks([])

# plt.subplot(222), plt.imshow(spectrum, cmap='gray')
# plt.title('frequence'), plt.xticks([]), plt.yticks([])

# plt.subplot(223), plt.imshow(spectrum_2, cmap='gray')
# plt.title('notch'), plt.xticks([]), plt.yticks([])


# plt.subplot(224), plt.imshow(filtered_image, cmap='gray')
# plt.title('filterd'), plt.xticks([]), plt.yticks([])


# plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------
def notch_reject_filter(shape, d0=9, u_k=0, v_k=0):
    P, Q = shape
    # Initialize filter with zeros
    H = np.zeros((P, Q))

    # Traverse through filter
    for u in range(0, P):
        for v in range(0, Q):
            # Get euclidean distance from point D(u,v) to the center
            D_uv = np.sqrt((u - P / 2 + u_k) ** 2 + (v - Q / 2 + v_k) ** 2)
            D_muv = np.sqrt((u - P / 2 - u_k) ** 2 + (v - Q / 2 - v_k) ** 2)

            if D_uv <= d0 or D_muv <= d0:
                H[u, v] = 0.0
            else:
                H[u, v] = 1.0

    return H
#-----------------------------------------------------

img = cv2.imread('b.jpg', 0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
phase_spectrumR = np.angle(fshift)
magnitude_spectrum = 20*np.log(np.abs(fshift))

img_shape = img.shape

H1 = notch_reject_filter(img_shape, 5, 8, 10)
# H2 = notch_reject_filter(img_shape, 4, -42, 27)
# H3 = notch_reject_filter(img_shape, 2, 80, 30)
# H4 = notch_reject_filter(img_shape, 2, -82, 28)
# H1 = notch_reject_filter(img_shape, 4, 38, 30)
# H = notch_reject_filter(img_shape, 4, 38, 30)
# H1 = notch_reject_filter(img_shape, 4, 38, 30)

NotchFilter = H1
NotchRejectCenter = fshift * NotchFilter 
NotchReject = np.fft.ifftshift(NotchRejectCenter)
inverse_NotchReject = np.fft.ifft2(NotchReject)  # Compute the inverse DFT of the result


Result = np.abs(inverse_NotchReject)

plt.subplot(222)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(221)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('magnitude spectrum')

plt.subplot(223)
plt.imshow(magnitude_spectrum*NotchFilter, "gray") 
plt.title("Notch Reject Filter")

plt.subplot(224)
plt.imshow(Result, "gray") 
plt.title("Result")


plt.show()