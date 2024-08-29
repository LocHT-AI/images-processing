import numpy as np
import cv2  # OpenCV for image manipulation


def apply_padding(image_array, pad_type='zero'):
    if pad_type == 'zero':
        return np.pad(image_array, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    elif pad_type == 'mirror':
        return np.pad(image_array, ((1, 1), (1, 1)), mode='reflect')
    elif pad_type == 'replicate':
        return np.pad(image_array, ((1, 1), (1, 1)), mode='edge')
    else:
        raise ValueError(f"Unsupported padding type: {pad_type}")
    

def lowpass_filter(image_array,pad_type ='zero'):
    # Define an 11x11 box kernel for the lowpass filter
    kernel_size = 11
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)  # Normalized box kernel
    

    padded_image = apply_padding(image_array, pad_type)
    # Apply the filter
    filtered_image = np.zeros_like(padded_image, dtype=np.float32)
    
    half_kernel = kernel_size // 2
    
    for i in range(half_kernel, image_array.shape[0] - half_kernel):
        for j in range(half_kernel, image_array.shape[1] - half_kernel):
            # Apply the kernel to the neighborhood
            neighborhood = image_array[i-half_kernel:i+half_kernel+1, j-half_kernel:j+half_kernel+1]
            filtered_pixel_value = np.sum(neighborhood * kernel)
            filtered_image[i, j] = filtered_pixel_value
    
    # Convert the filtered image to uint8 for display
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    
    return filtered_image

# Example usage
# Read an image using OpenCV
input_image = cv2.imread('4.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Apply the lowpass filter
# output_image = lowpass_filter(gray_image)
# Example with zero padding
output_array_zero = lowpass_filter(gray_image, pad_type='zero')

# Example with mirror padding
output_array_mirror = lowpass_filter(gray_image, pad_type='mirror')

# Example with replicate padding
output_array_replicate = lowpass_filter(gray_image, pad_type='replicate')

# Display the original and filtered images
cv2.imshow('Original Image', gray_image)
cv2.imshow('zero', output_array_zero)
cv2.imshow('mirror', output_array_mirror)
cv2.imshow('replicate', output_array_replicate)

cv2.waitKey(0)
cv2.destroyAllWindows()
