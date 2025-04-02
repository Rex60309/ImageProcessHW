import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Create image
def create_square_image(size, square_size):
    img = np.zeros((size, size), dtype=np.uint8)
    start = (size - square_size) // 2
    end = start + square_size
    img[start:end, start:end] = 255
    return img

# Create and display original image
size = 200
square_size = 80
original_img = create_square_image(size, square_size)
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(original_img, cmap='gray')
plt.title('Original Image')

angle = 30

# Nearest Neighbor Interpolation
rotated_nearest = ndimage.rotate(original_img, angle, order=0, reshape=False)
plt.subplot(132)
plt.imshow(rotated_nearest, cmap='gray')
plt.title('Rotation with Nearest Neighbor Interpolation')
plt.xlim([0, size])
plt.ylim([0, size])

# Bilinear Interpolation
rotated_bilinear = ndimage.rotate(original_img, angle, order=1, reshape=False)
plt.subplot(133)
plt.imshow(rotated_bilinear, cmap='gray')
plt.title('Rotation with Bilinear Interpolation')
plt.xlim([0, size])
plt.ylim([0, size])

plt.tight_layout()
plt.savefig('rotation_result.png')
plt.show()

# Calculate and display the difference between NN and Bilinear
diff_img = np.abs(rotated_nearest.astype(int) - rotated_bilinear.astype(int)).astype(np.uint8)
plt.figure(figsize=(5, 5))
plt.imshow(diff_img, cmap='hot')
plt.colorbar()
plt.title('Difference Between Two Rotations')
plt.xlim([0, size])
plt.ylim([0, size])
plt.savefig('interpolation_difference.png')
plt.show()