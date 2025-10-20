import numpy as np
import matplotlib.pyplot as plt

# Simuler une relation entre d et Q 
d_values = np.linspace(0, 1, 100)  # Niveaux de bruit de 0 à 1
Q_values = 20 * np.sqrt(d_values)  # Q augmente avec la racine carrée de d

# Tracer le graphique
plt.plot(d_values, Q_values, label='Q = f(d)')
plt.xlabel('Densité de bruit (d)')
plt.ylabel('Erreur quadratique moyenne (Q)')
plt.title('Évolution de Q en fonction du niveau du bruit')
plt.grid(True)
plt.legend()
plt.show()