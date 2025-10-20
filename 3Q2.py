import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import log10

# Charger l'image
image = cv2.imread("M5.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Erreur : impossible de charger l'image")
    exit()

# Paramètres
block_size = 8
max_gray = 255
psnr_values = []
compression_rates = []

# Fonction pour calculer le PSNR
def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * log10((max_gray ** 2) / mse)

# Tester différents taux de compression
for threshold_percent in np.arange(0.1, 1.0, 0.1):  # De 10 % à 90 %
    # Appliquer DCT et seuillage
    height, width = image.shape
    compressed_image = np.zeros_like(image, dtype=np.float32)
    for i in range(0, height - block_size + 1, block_size):
        for j in range(0, width - block_size + 1, block_size):
            block = image[i:i+block_size, j:j+block_size].astype(np.float32)
            dct_block = cv2.dct(block)
            # Seuillage : conserver les top N% des coefficients
            flat_dct = dct_block.flatten()
            threshold_value = np.percentile(np.abs(flat_dct), 100 * (1 - threshold_percent))
            dct_block[np.abs(dct_block) < threshold_value] = 0
            # Reconstruire le bloc
            reconstructed_block = cv2.idct(dct_block)
            compressed_image[i:i+block_size, j:j+block_size] = reconstructed_block

    # Calculer le PSNR
    psnr = calculate_psnr(image, compressed_image.astype(np.uint8))
    psnr_values.append(psnr)

    # Calculer le taux de compression approximatif (pourcentage de coefficients non nuls)
    total_coeffs = height * width
    non_zero_coeffs = np.count_nonzero(compressed_image)
    compression_rate = total_coeffs / non_zero_coeffs if non_zero_coeffs > 0 else float('inf')
    compression_rates.append(1 / compression_rate)  # Inverse pour taux de compression

# Tracer le graphique
plt.plot(compression_rates, psnr_values, marker='o')
plt.xlabel('Taux de compression (fraction de coefficients conservés)')
plt.ylabel('PSNR (dB)')
plt.title('Évolution du PSNR en fonction du taux de compression')
plt.grid()
plt.show()