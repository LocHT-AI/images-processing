from skimage.io import imread                         # load the image
from skimage.exposure import cumulative_distribution  # calculate the cumulative pixel value
import numpy as np                                    # reshape the image
import matplotlib.pyplot as plt                       # plot the result
import cv2

def compute_cumula(image):
    # compute the cumulative distribution of input image
    cdfImageInput, binsImageInput = cumulative_distribution(image)
    cdfImageInput = np.insert(cdfImageInput, 0, [0]*binsImageInput[0]) # fill 0 in index 0 - 17
    cdfImageInput = np.append(cdfImageInput, [1]*(255-binsImageInput[-1])) # fill 1 in index 247 - 255

    # plt.plot(cdfImageInput, linewidth=5)
    # plt.xlim(0,255)
    # plt.ylim(0,1)
    # plt.xlabel('Pixel Values')
    # plt.ylabel('Cumulative Probability')
    # plt.show()
    return cdfImageInput

def show_cdf(image,imageTemplate):
    plt.plot(cdf_images, linewidth=5, label='Input Image')
    plt.plot(cdf_imaTem, linewidth=5, label='Template')
    plt.xlim(0,255)
    plt.ylim(0,1)
    plt.xlabel('Pixel Values')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.show()

def histMatch(cdfInput, cdfTemplate, imageInput):
    pixels = np.arange(256)
    new_pixels = np.interp(cdfInput, cdfTemplate, pixels)
    imageMatch = (np.reshape(new_pixels[imageInput.ravel()], imageInput.shape)).astype(np.uint8)
    return imageMatch

def show(image,imageTemplate):
    plt.figure(figsize=(8,6))
    plt.subplot(1,2,1)
    plt.title('Input Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title('Template Image')
    plt.imshow(cv2.cvtColor(imageTemplate, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


if __name__== "__main__" :
    # read/load the input and template image
    image = (imread('xam.jpg', as_gray=True)*255).astype(np.uint8)
    imageTemplate = (imread('gray.jpeg', as_gray=True)*255).astype(np.uint8)

    show(image,imageTemplate)
    cdf_images=compute_cumula(image)
    cdf_imaTem=compute_cumula(imageTemplate)
    show_cdf(image,imageTemplate)

    imageResult = histMatch(cdf_images, cdf_imaTem, image)

    show(image,imageResult)
    # plt.imshow(imageResult, cmap='gray')
    # plt.axis('off')
    # plt.show()
