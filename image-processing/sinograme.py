import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon

def generate_sinogram(image_path, num_angles):
    # Read the input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Perform Radon transform
    sinogram = radon(image, theta=np.linspace(0, 180, num_angles))

    return sinogram

def plot_sinogram(sinogram):
    plt.figure(figsize=(10, 5))
    plt.title('Sinogram')
    plt.xlabel('Projection Angle (degrees)')
    plt.ylabel('Projection Position (pixels)')
    plt.imshow(sinogram, cmap='gray', aspect='auto')
    plt.colorbar(label='Intensity')
    plt.show()

if __name__ == "__main__":
    # Path to the input image
    image_path = '1.jpg'

    # Number of angles for Radon transform
    num_angles = 180

    # Generate sinogram
    sinogram = generate_sinogram(image_path, num_angles)

    # Plot sinogram
    plot_sinogram(sinogram)



# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.transform import radon

# def generate_sinogram(image_path, num_angles):
#     # Read the input image
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     # Perform Radon transform
#     sinogram = radon(image, theta=np.linspace(0, 180, num_angles))

#     return sinogram

# def plot_sinogram(image,sinogram):
#     plt.subplot(221), plt.imshow(image, cmap='gray')
#     plt.title('Ảnh gốc'), plt.xticks([]), plt.yticks([])

#     plt.subplot(222), plt.imshow(sinogram, cmap='gray')
#     plt.title('sinogram'), plt.xticks([]), plt.yticks([])

#     # plt.subplot(223), plt.imshow(gaussian_image, cmap='gray')
#     # plt.title('gaussian'), plt.xticks([]), plt.yticks([])


#     # plt.subplot(224), plt.imshow(butterworth_image, cmap='gray')
#     # plt.title('butterworth_bandpass'), plt.xticks([]), plt.yticks([])


#     plt.show()

# if __name__ == "__main__":
#     # Path to the input image
#     image_path = '3.jpg'
#     image=cv2.imread(image_path)

#     # Number of angles for Radon transform
#     num_angles = 100

#     # Generate sinogram
#     sinogram = generate_sinogram(image_path, num_angles)

#     # Plot sinogram
#     plot_sinogram(image,sinogram)
