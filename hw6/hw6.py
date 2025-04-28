import numpy as np
import matplotlib.pyplot as plt

# Create an image g(x,y) with gray value = 100
def create_image(size=(256, 256), gray_value = 100):
    return np.ones(size, dtype=np.uint8) * gray_value

# Gaussian noise n(x,y) with mean = 0, var = 25
def generate_gaussian_noise(size=(256, 256), mean=0, var=25):
    sigma = np.sqrt(var)
    noise = np.zeros(size, dtype=np.float32)

    rows, cols = size
    for i in range(rows):
        j = 0
        while j < cols - 1:
            r = np.random.rand()
            phi = np.random.rand()

            z1 = sigma * np.cos(2 * np.pi * phi) * np.sqrt(-2 * np.log(r))
            z2 = sigma * np.sin(2 * np.pi * phi) * np.sqrt(-2 * np.log(r))

            noise[i, j] = z1
            noise[i, j+1] = z2

            j += 2

        # if odd column
        if j == cols - 1:
            r = np.random.rand()
            phi = np.random.rand()
            z1 = sigma * np.cos(2 * np.pi * phi) * np.sqrt(-2 * np.log(r))
            noise[i, j] = z1

    return noise.astype(np.int16)

# Generate noisy image f(x,y) = g(x,y) + n(x,y)
def add_noise_to_image(image, noise):
    G = 256
    # add noise
    f_prime = image.astype(np.int16) + noise
    # clip to 0-255
    f_xy = np.where(f_prime < 0, 0, np.where(f_prime > G-1, G-1, f_prime))

    return f_xy.astype(np.uint8)


# Display histogram
def plot_histogram(image, title='Histogram'):
    hist = np.zeros(256)
    for i in range(256):
        hist[i] = np.sum(image == i)

    plt.bar(range(256), hist)
    plt.title(title)
    plt.xlabel('Gray Level', labelpad=0)
    plt.ylabel('Frequency')
    plt.xlim([0, 255])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return hist

def main():
    image_size = (256, 256)

    # original g(x,y)
    g_xy = create_image(image_size)

    # noisy f(x,y)
    noise = generate_gaussian_noise(image_size)
    f_xy = add_noise_to_image(g_xy, noise)

    # result
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(g_xy, cmap='gray', vmin=0, vmax=255)
    plt.title('Original Image g(x,y)')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plot_histogram(g_xy, 'Histogram of g(x,y)')

    plt.subplot(2, 2, 3)
    plt.imshow(f_xy, cmap='gray', vmin=0, vmax=255)
    plt.title('Noisy Image f(x,y)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plot_histogram(f_xy, 'Histogram of f(x,y)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()