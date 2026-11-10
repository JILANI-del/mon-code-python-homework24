import matplotlib.pyplot as plt
import numpy as np

# Valeurs de x
x = np.linspace(0, 10, 100)

# Fonction linéaire
y = 2*x + 3

# Tracer la courbe
plt.plot(x, y, label="y = 2x + 3")

# Ajouter les titres
plt.title("Courbe linéaire")
plt.xlabel("Axe X")
plt.ylabel("Axe Y")

# Afficher la grille et la légende
plt.grid(True)
plt.legend()
plt.show()