import cv2
import numpy as np
import os

# Charger l'image en niveaux de gris
image_path = "M5.jpg"
if not os.path.exists(image_path):
    print(f"Erreur : fichier {image_path} non trouvé")
    exit()

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print(f"Erreur : impossible de charger {image_path}")
    exit()

# Paramètres
block_size = 8
height, width = image.shape
compression_threshold = 0.1  # Seuil pour conserver 10 % des coefficients (ajustable)

# Fonction pour appliquer la compression DCT
def compress_image(image, threshold):
    compressed_image = np.zeros_like(image, dtype=np.float32)
    for i in range(0, height - block_size + 1, block_size):
        for j in range(0, width - block_size + 1, block_size):
            # Extraire un bloc 8x8
            block = image[i:i+block_size, j:j+block_size].astype(np.float32)
            # Appliquer la DCT
            dct_block = cv2.dct(block)
            # Seuillage : conserver les top N% des coefficients
            flat_dct = dct_block.flatten()
            threshold_value = np.percentile(np.abs(flat_dct), 100 * (1 - threshold))
            dct_block[np.abs(dct_block) < threshold_value] = 0
            # Reconstruire le bloc avec IDCT
            reconstructed_block = cv2.idct(dct_block)
            compressed_image[i:i+block_size, j:j+block_size] = reconstructed_block
    return compressed_image

# Appliquer la compression
compressed_image = compress_image(image, compression_threshold)

# Convertir en uint8 pour sauvegarde/affichage
compressed_image_uint8 = np.clip(compressed_image, 0, 255).astype(np.uint8)

# Sauvegarder l'image compressée
output_path = "M5_compressed.jpg"
cv2.imwrite(output_path, compressed_image_uint8)
print(f"Image compressée sauvegardée sous : {output_path}")

# Calculer le taux de compression approximatif
total_coeffs = height * width
non_zero_coeffs = np.count_nonzero(compressed_image)
compression_rate = total_coeffs / non_zero_coeffs if non_zero_coeffs > 0 else float('inf')
print(f"Taux de compression approximatif : {compression_rate:.2f}:1")