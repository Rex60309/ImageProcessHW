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
        
        # update threshold
        if variance > max_variance:
            max_variance = variance
            best_threshold = t
    
    return best_threshold

def apply_threshold(image, threshold):
    # if color image, covert to grayscale image
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # if pixel < threshold, then = 0 
    # else = 255
    segmented_image = np.zeros_like(image)
    segmented_image[image > threshold] = 255
    
    return segmented_image

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
    
    segmented_image = apply_threshold(image, threshold)
    
    _, opencv_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # result
    plt.figure(figsize=(12, 4))
    
    plt.subplot(141)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(142)
    plt.title('Histogram')
    plt.hist(image.flatten(), 256, [0, 256])
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'threshold = {threshold}')
    plt.xlim(0, 255)
    plt.legend()
    
    plt.subplot(143)
    plt.title('Otsu')
    plt.imshow(segmented_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(144)
    plt.title('OpenCV Otsu')
    plt.imshow(opencv_otsu, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("Complete!")