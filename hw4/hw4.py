import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2

# Step 1: Create an image with a white square on a black background
def create_square_image(size=200, square_size=60):
    """Create a black image with a white square in the center."""
    img = np.zeros((size, size), dtype=np.uint8)
    start = (size - square_size) // 2
    end = start + square_size
    img[start:end, start:end] = 255
    return img

# Create and display the original image
original_img = create_square_image()
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(original_img, cmap='gray')
plt.title('Original Image')

# Step 2: Rotate the image by 30 degrees with different interpolation methods
angle = 30

# (a) Nearest neighbor interpolation
rotated_nearest = ndimage.rotate(original_img, angle, order=0, reshape=False)
plt.subplot(132)
plt.imshow(rotated_nearest, cmap='gray')
plt.title('Rotation with Nearest Neighbor')

# (b) Bilinear interpolation
rotated_bilinear = ndimage.rotate(original_img, angle, order=1, reshape=False)
plt.subplot(133)
plt.imshow(rotated_bilinear, cmap='gray')
plt.title('Rotation with Bilinear Interpolation')

plt.tight_layout()
plt.savefig('rotation_comparison.png')
plt.show()

# Step 3: Compare the two results
print("Comparison of the two rotation methods:")
print("1. Nearest Neighbor Interpolation:")
print("   - Preserves the original pixel values")
print("   - Results in a blocky/pixelated appearance")
print("   - Does not introduce new intensity values")
print("   - Edges appear jagged")
print("\n2. Bilinear Interpolation:")
print("   - Creates smoother transitions between pixels")
print("   - Introduces new intermediate intensity values")
print("   - Results in smoother edges")
print("   - May slightly blur the image")

# Optional: Calculate and display the difference between the two rotated images
diff_img = np.abs(rotated_nearest.astype(int) - rotated_bilinear.astype(int)).astype(np.uint8)
plt.figure(figsize=(5, 5))
plt.imshow(diff_img, cmap='hot')
plt.colorbar()
plt.title('Difference Between Interpolation Methods')
plt.savefig('interpolation_difference.png')
plt.show()

# Display pixel values in a small region to see the differences
def show_pixel_values(region_img, title):
    """Display pixel values in a small region of the image."""
    plt.figure(figsize=(6, 6))
    plt.imshow(region_img, cmap='gray')
    plt.title(title)

    # Add pixel values as text
    for i in range(region_img.shape[0]):
        for j in range(region_img.shape[1]):
            plt.text(j, i, f'{region_img[i, j]}',
                     ha='center', va='center', color='red', fontsize=8)

    plt.grid(True, color='yellow', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.show()

# Extract a small region around an edge for comparison
def extract_region(img, size=10):
    """Extract a small region near the edge of the rotated square."""
    # Find the edge of the square (a transition from black to white)
    edges = cv2.Canny(img, 100, 200)
    y, x = np.where(edges > 0)
    if len(y) == 0 or len(x) == 0:
        # If no edge is found, use the center of the image
        center_y, center_x = img.shape[0] // 2, img.shape[1] // 2
    else:
        # Use the first detected edge point
        center_y, center_x = y[0], x[0]

    # Extract region around the edge
    half_size = size // 2
    y_min = max(0, center_y - half_size)
    y_max = min(img.shape[0], center_y + half_size)
    x_min = max(0, center_x - half_size)
    x_max = min(img.shape[1], center_x + half_size)

    return img[y_min:y_max, x_min:x_max]

# Show pixel values in a region for both interpolation methods
region_nearest = extract_region(rotated_nearest)
region_bilinear = extract_region(rotated_bilinear)

show_pixel_values(region_nearest, "Nearest Neighbor Region")
show_pixel_values(region_bilinear, "Bilinear Region")