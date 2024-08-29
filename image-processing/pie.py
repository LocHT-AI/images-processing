import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = '1.jpg'
image = cv2.imread(image_path, 0)  # Read the image as grayscale

# Define the piecewise linear transformation function
def piecewise_linear_transform(image, a, b):
    # Normalize the image to the range [0, 1]
    normalized_image = image / 255.0
    
    # Apply the piecewise linear transformation
    transformed_image = np.piecewise(normalized_image, [
        normalized_image < a,
        (normalized_image >= a) & (normalized_image < b),
        normalized_image >= b
    ], [
        lambda x: x * a,
        lambda x: (x - a) * (1 / (b - a)) + a,
        lambda x: (x - b) * (1 / (1 - b)) + b
    ])
    
    # Convert the transformed image back to the range [0, 255]
    transformed_image = (transformed_image * 255).astype(np.uint8)
    
    return transformed_image

# Define the parameters for the piecewise linear transformation
a = 0.2
b = 0

# Apply the piecewise linear transformation
transformed_image = piecewise_linear_transform(image, a, b)

# Display the original and transformed images
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(transformed_image, cmap='gray')
plt.title('Transformed Image')

plt.show()