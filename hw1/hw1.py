import cv2
import numpy as np

def dither_image_A(input_path):
    # Load the image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
    height, width = img.shape

    # Dithering matrix
    D_base = np.array([
        [0, 128, 32, 160],
        [192, 64, 224, 96],
        [48, 176, 16, 144],
        [240, 112, 208, 80]
    ])

    # Calculate how many repetitions
    repeat_y = int(np.ceil(height / 4))
    repeat_x = int(np.ceil(width / 4))

    # Repeat dithering matrix to generate D and fit to the image size
    D = np.tile(D_base, (repeat_y, repeat_x))
    D = D[:height, :width]

    # If I(i,j) > D(i,j), then I'(i,j) = 255, else I'(i,j) = 0
    # set all to 0
    I2 = np.zeros_like(img)
    # set 255
    I2[img > D] = 255

    # Save the result
    output_path = 'A_dithered_image.jpg'
    cv2.imwrite(output_path, I2)
    print(f"Dithered image saved to {output_path}")

def dither_image_B(input_path):
    # Load the image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
    height, width = img.shape

    # 255 / 3 = 85
    Q = np.floor(img / 85)

    # Extended dithering matrix
    D_extend = np.array([
        [0, 56],
        [84, 28]
    ])

    # Calculate how many repetitions
    repeat_y = int(np.ceil(height / 2))
    repeat_x = int(np.ceil(width / 2))

    # Repeat dithering matrix to generate D and fit to the image size
    D_ext = np.tile(D_extend, (repeat_y, repeat_x))
    D_ext = D_ext[:height, :width]

    # If I(i,j) - 85 * Q(i,j) > D(i,j), then I'(i,j) = Q(i,j) + 1
    # else I'(i,j) = Q(i,j) + 0
    # set all to 0
    I_ext = np.zeros_like(img)
    # set I'
    condition = (img - Q * 85) > D_ext
    I_ext = np.where(condition, Q + 1, Q)

    # Scale values to [0, 255] for displaying
    I_ext = (I_ext * 85).astype(np.uint8)

    # Save the result
    output_path_extended = 'B_extended_dithered_image.jpg'
    cv2.imwrite(output_path_extended, I_ext)
    print(f"Extended dithered image saved to {output_path_extended}")

if __name__ == "__main__":
    dither_image_A('grayscale_image.jpg')
    dither_image_B('grayscale_image.jpg')