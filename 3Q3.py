import cv2
import numpy as np
from math import log10

# Charger l'image
image = cv2.imread("M5.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Erreur : impossible de charger l'image")
    exit()

# Paramètres
block_size = 8
height, width = image.shape
max_gray = 255

# Fonction pour calculer le PSNR
def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * log10((max_gray ** 2) / mse)

# Compression DCT
threshold_percent = 0.3  # Exemple : 30 % de coefficients conservés
compressed_image = np.zeros_like(image, dtype=np.float32)
for i in range(0, height - block_size + 1, block_size):
    for j in range(0, width - block_size + 1, block_size):
        block = image[i:i+block_size, j:j+block_size].astype(np.float32)
        dct_block = cv2.dct(block)
        flat_dct = dct_block.flatten()
        threshold_value = np.percentile(np.abs(flat_dct), 100 * threshold_percent)
        dct_block[np.abs(dct_block) < threshold_value] = 0
        reconstructed_block = cv2.idct(dct_block)
        compressed_image[i:i+block_size, j:j+block_size] = reconstructed_block

psnr = calculate_psnr(image, np.clip(compressed_image, 0, 255).astype(np.uint8))
print(f"PSNR : {psnr:.2f} dB")

