import numpy as np
import cv2
import matplotlib.pyplot as plt

def rgb_to_hsv(rgb_img):
    hsv_img = np.zeros_like(rgb_img, dtype=np.float32)

    r = rgb_img[:, :, 0]
    g = rgb_img[:, :, 1]
    b = rgb_img[:, :, 2]

    # Calculate V
    v = np.max(rgb_img, axis=2)

    # Calculate S
    min_val = np.min(rgb_img, axis=2)
    s = np.zeros_like(v)
    non_zero_indices = (v != 0)
    s[non_zero_indices] = (v[non_zero_indices] - min_val[non_zero_indices]) / v[non_zero_indices]

    # Calculate H
    h = np.zeros_like(v)

    # V = R
    indices = (v == r) & (v != 0) & (v != min_val)
    if np.any(indices):
        h[indices] = (1/6) * ((g[indices] - b[indices]) / (v[indices] - min_val[indices]))

    # V = G
    indices = (v == g) & (v != 0) & (v != min_val)
    if np.any(indices):
        h[indices] = (1/6) * ((b[indices] - r[indices]) / (v[indices] - min_val[indices])) + (2/6)

    # V = B
    indices = (v == b) & (v != 0) & (v != min_val)
    if np.any(indices):
        h[indices] = (1/6) * ((r[indices] - g[indices]) / (v[indices] - min_val[indices])) + (4/6)

    h[h < 0] += 1.0

    hsv_img[:, :, 0] = h
    hsv_img[:, :, 1] = s
    hsv_img[:, :, 2] = v

    return hsv_img

def hsv_to_rgb(hsv_img):
    rgb_img = np.zeros_like(hsv_img, dtype=np.float32)

    h = hsv_img[:, :, 0]
    s = hsv_img[:, :, 1]
    v = hsv_img[:, :, 2]

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    # Fix NaN values and ensure h is in valid range
    h = np.nan_to_num(h, nan=0.0)
    h = np.clip(h, 0.0, 0.999999)

    h_i = np.floor(6 * h).astype(np.int32)
    f = 6 * h - h_i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    # h_i = 0
    indices = (h_i == 0)
    r[indices] = v[indices]
    g[indices] = t[indices]
    b[indices] = p[indices]

    # h_i = 1
    indices = (h_i == 1)
    r[indices] = q[indices]
    g[indices] = v[indices]
    b[indices] = p[indices]

    # h_i = 2
    indices = (h_i == 2)
    r[indices] = p[indices]
    g[indices] = v[indices]
    b[indices] = t[indices]

    # h_i = 3
    indices = (h_i == 3)
    r[indices] = p[indices]
    g[indices] = q[indices]
    b[indices] = v[indices]

    # h_i = 4
    indices = (h_i == 4)
    r[indices] = t[indices]
    g[indices] = p[indices]
    b[indices] = v[indices]

    # h_i = 5
    indices = (h_i == 5)
    r[indices] = v[indices]
    g[indices] = p[indices]
    b[indices] = q[indices]

    indices = (s == 0)
    r[indices] = v[indices]
    g[indices] = v[indices]
    b[indices] = v[indices]

    rgb_img[:, :, 0] = r
    rgb_img[:, :, 1] = g
    rgb_img[:, :, 2] = b

    return rgb_img

def histogram_equalization(v_channel):
    # Convert to 8-bit for histogram calculation
    v_8bit = np.round(v_channel * 255).astype(np.uint8)

    hist, bins = np.histogram(v_8bit.flatten(), 256, [0, 256])

    cdf = hist.cumsum()

    cdf_normalized = cdf * 255 / cdf[-1]

    # Apply histogram equalization
    v_equalized_8bit = np.interp(v_8bit.flatten(), np.arange(256), cdf_normalized).reshape(v_8bit.shape)

    # Convert back to 0-1
    v_equalized = v_equalized_8bit / 255.0

    return v_equalized

def enhance_contrast(input_img):
    # Scale to 0-1
    rgb_img = input_img.astype(np.float32) / 255.0

    hsv_img = rgb_to_hsv(rgb_img)

    v_equalized = histogram_equalization(hsv_img[:, :, 2])

    hsv_enhanced = hsv_img.copy()
    hsv_enhanced[:, :, 2] = v_equalized

    rgb_enhanced = hsv_to_rgb(hsv_enhanced)

    # Scale back to 0-255
    RGB_enhanced = np.clip(rgb_enhanced * 255.0, 0, 255).astype(np.uint8)

    return input_img, RGB_enhanced

def main():
    img_path = input("Image path: ")
    input_img = cv2.imread(img_path)

    # BGR to RGB
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    original_img, enhanced_img = enhance_contrast(input_img)

    # Display results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title('Original RGB Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_img)
    plt.title('Enhanced R\'G\'B\' Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()