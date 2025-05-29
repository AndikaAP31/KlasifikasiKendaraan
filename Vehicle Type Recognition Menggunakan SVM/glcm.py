import cv2
import numpy as np

img_height = 128
img_width = 128

# fungsi graycomatrix untuk ekstraksi fitur Gray-Level Co-occurrence Matrix (GLCM)
def graycomatrix(image, distances, angles, levels, symmetric, normed):
    image = cv2.resize(image, (img_width, img_height))
    height, width = image.shape
    result = np.zeros((levels, levels, len(distances), len(angles)), dtype=np.float64)
    
    if levels != 256:
        image = np.floor(image * (levels / 256)).astype(np.int32)
        image = np.clip(image, 0, levels - 1)

    for d_idx, distance in enumerate(distances):
        for a_idx, angle in enumerate(angles):
            dx, dy = int(round(np.cos(angle) * distance)), int(round(np.sin(angle) * distance))
            
            for y in range(height):
                for x in range(width):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        i, j = image[y, x], image[ny, nx]
                        result[i, j, d_idx, a_idx] += 1
                        if symmetric:
                            result[j, i, d_idx, a_idx] += 1

    if normed:
        for d_idx in range(len(distances)):
            for a_idx in range(len(angles)):
                total = np.sum(result[:, :, d_idx, a_idx])
                if total > 0:
                    result[:, :, d_idx, a_idx] /= total

    return result

# fungsi graycoprops untuk ekstraksi fitur Gray-Level Co-occurrence Matrix (GLCM)
def graycoprops(matrix, prop):
    num_distances = matrix.shape[2]
    num_angles = matrix.shape[3]
    results = np.zeros((num_distances, num_angles))
    
    I, J = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]), indexing='ij')
    
    for d in range(num_distances):
        for a in range(num_angles):
            P = matrix[:, :, d, a]
            if prop == 'contrast':
                results[d, a] = np.sum(P * ((I - J) ** 2))
            elif prop == 'correlation':
                mu_i = np.sum(I * P)
                mu_j = np.sum(J * P)
                sigma_i = np.sqrt(np.sum(P * ((I - mu_i) ** 2)))
                sigma_j = np.sqrt(np.sum(P * ((J - mu_j) ** 2)))
                results[d, a] = np.sum(P * (I - mu_i) * (J - mu_j) / (sigma_i * sigma_j + 1e-10))
            elif prop == 'energy':
                results[d, a] = np.sum(P ** 2)
            elif prop == 'homogeneity':
                results[d, a] = np.sum(P / (1 + np.abs(I - J)))
    return results

# fungsi ekstraksi fitur dengan Gray-Level Co-occurrence Matrix (GLCM)
def glcm(img):
    img = cv2.resize(img, (img_width, img_height))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )
    
    props = ['contrast', 'correlation', 'energy', 'homogeneity']
    feats = [graycoprops(glcm, p)[0, 0] for p in props]
    return np.array(feats)