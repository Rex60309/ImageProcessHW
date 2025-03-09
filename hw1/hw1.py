import cv2
import numpy as np

def dither_image(input_path):
    # Read the grayscale image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return

    # Get image dimensions
    height, width = img.shape

    # Define the 4x4 dithering matrix D as specified in the assignment
    D_base = np.array([
        [0, 128, 32, 160],
        [192, 64, 224, 96],
        [48, 176, 16, 144],
        [240, 112, 208, 80]
    ])

    # Create D array of the same size as the image by repeating the dithering matrix
    # Calculate how many repetitions we need in each dimension
    repeat_y = int(np.ceil(height / 4))
    repeat_x = int(np.ceil(width / 4))

    # Create the repeated matrix and then crop to the image size
    D = np.tile(D_base, (repeat_y, repeat_x))
    D = D[:height, :width]

    # Apply the dithering threshold (Step 2 in the assignment)
    # If I(i,j) ≥ D(i,j), then I₂(i,j) = 255, else I₂(i,j) = 0
    I2 = np.zeros_like(img)
    I2[img >= D] = 255

    # Save the result
    output_path = 'dithered_image.jpg'
    cv2.imwrite(output_path, I2)
    print(f"Dithered image saved to {output_path}")

    # Part B: Extend to n=4 gray values
    # Step 1: 255/3 = 85
    Q = np.floor(img / 85)

    # Step 3: Extend the dithering matrix D
    D_extend = np.array([
        [0, 56],
        [84, 28]
    ])

    # Repeat the extended dithering matrix
    repeat_y = int(np.ceil(height / 2))
    repeat_x = int(np.ceil(width / 2))
    D_ext = np.tile(D_extend, (repeat_y, repeat_x))
    D_ext = D_ext[:height, :width]

    # Step 4: Apply the extended dithering
    I_ext = np.zeros_like(img)

    # For each pixel: if I(i,j) - Q(i,j)*85 >= D(i,j), then I(i,j) = Q(i,j)+1, else I(i,j) = Q(i,j)
    condition = (img - Q * 85) >= D_ext
    I_ext = np.where(condition, Q + 1, Q)

    # Step 5: Scale values to [0, 255] for displaying
    I_ext = (I_ext * 85).astype(np.uint8)

    # Save the extended result
    output_path_extended = 'dithered_image_extended.jpg'
    cv2.imwrite(output_path_extended, I_ext)
    print(f"Extended dithered image saved to {output_path_extended}")

if __name__ == "__main__":
    # Process the grayscale image
    dither_image('grayscale_image.jpg')