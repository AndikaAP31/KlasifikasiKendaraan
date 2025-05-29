import cv2
import numpy as np

img_height = 128
img_width = 128

# fungsi ekstraksi fitur dengan Histogram of Oriented Gradients (HOG)
def hog(img):
    img = cv2.resize(img, (img_width, img_height))
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    cell_size = 8
    block_size = 2
    nbins = 9
    
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = (np.arctan2(gy, gx) * (180 / np.pi)) % 180

    height, width = gray.shape
    n_cells_y = height // cell_size
    n_cells_x = width // cell_size
    histograms = np.zeros((n_cells_y, n_cells_x, nbins), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            mag = magnitude[y, x]
            angle = orientation[y, x]
            
            cell_x = x // cell_size
            cell_y = y // cell_size
            if cell_x >= n_cells_x or cell_y >= n_cells_y:
                continue

            bin_idx = angle / (180 / nbins)
            bin_lower = int(bin_idx) % nbins
            bin_upper = (bin_lower + 1) % nbins
            weight_upper = bin_idx - bin_lower
            weight_lower = 1 - weight_upper

            histograms[cell_y, cell_x, bin_lower] += mag * weight_lower
            histograms[cell_y, cell_x, bin_upper] += mag * weight_upper

    n_blocks_y = n_cells_y - block_size + 1
    n_blocks_x = n_cells_x - block_size + 1
    eps = 1e-5
    hog_vector = []

    for y in range(n_blocks_y):
        for x in range(n_blocks_x):
            block = histograms[y:y+block_size, x:x+block_size, :].flatten()
            norm = np.sqrt(np.sum(block ** 2) + eps ** 2)
            normalized_block = block / norm
            hog_vector.append(normalized_block)

    return np.concatenate(hog_vector)