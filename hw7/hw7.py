import numpy as np
import cv2
import matplotlib.pyplot as plt

def otsu_threshold(image):
    # if color image, covert to grayscale image
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    p = hist / hist.sum()
    
    best_threshold = 0
    max_variance = 0
    
    # find best threshold through all gray values
    for t in range(1, 256):
        left = np.sum(p[:t])
        right = np.sum(p[t:])
        
        if left == 0 or right == 0:
            continue

        left_avg = np.sum(np.arange(t) * p[:t]) / left
        right_avg = np.sum(np.arange(t, 256) * p[t:]) / right
        
        variance = left * right * ((left_avg - right_avg) ** 2)
                
        # update better threshold
        if variance > max_variance:
            max_variance = variance
            best_threshold = t
    
    return best_threshold

def apply_threshold(image, threshold):  
    # if pixel < threshold, then = 0 
    # else = 255
    otsu_image = np.zeros_like(image)
    otsu_image[image > threshold] = 255
    
    return otsu_image

def main():
    image_path = input("Input image path: ")
    
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Can't read image from '{image_path}'")
            raise ValueError("Invalid image path")
    except Exception as e:
        print(f"Error: {e}")

    threshold = otsu_threshold(image)
    print(f"Otsu's best threshold: {threshold}")
    
    otsu_image = apply_threshold(image, threshold)

    # result
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.title('Original Image (Grayscale)')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(132)
    plt.title('Probability Distribution Histogram')
    plt.hist(image.flatten(), 256, [0, 256])
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'best threshold = {threshold}')
    plt.xlim(0, 255)
    plt.legend()
    
    plt.subplot(133)
    plt.title('Otsu\'s Image')
    plt.imshow(otsu_image, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()