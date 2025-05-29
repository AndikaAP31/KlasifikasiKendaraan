import cv2
import numpy as np

img_height = 128
img_width = 128

# fungsi ekstraksi fitur dengan Histogram Warna
def color_histogram(img, bins=(8, 8, 8)):
    img = cv2.resize(img, (img_width, img_height))
    hist = np.zeros(bins[0] * bins[1] * bins[2], dtype=np.float32)
    height, width = img.shape[:2]
    
    bin_width = [256 // bins[i] for i in range(3)]

    for y in range(height):
        for x in range(width):
            b, g, r = img[y, x]
            b_bin = min(b // bin_width[0], bins[0] - 1)
            g_bin = min(g // bin_width[1], bins[1] - 1)
            r_bin = min(r // bin_width[2], bins[2] - 1)

            idx = int(r_bin) * bins[1] * bins[0] + int(g_bin) * bins[0] + int(b_bin)
            hist[idx] += 1.0

    norm = np.sqrt(np.sum(hist ** 2) + 1e-10)
    hist /= norm
    
    return hist