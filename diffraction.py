
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.special import j1

# ==========================================
# CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Diffraction circulaire",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Simulation de diffraction circulaire")
st.markdown("### Montage : Laser → Ouverture circulaire → Écran")

# ==========================================
# PARAMETRES
# ==========================================

st.sidebar.header("1. Laser")

lambda_nm = st.sidebar.slider(
    "Longueur d'onde (nm)",
    380, 750, 550, 10
)

st.sidebar.header("2. Ouverture circulaire")

D_mm = st.sidebar.slider(
    "Diamètre D (mm)",
    0.05, 2.0, 0.5, 0.05
)

st.sidebar.header("3. Écran de projection")

L = st.sidebar.slider(
    "Distance écran-ouverture (m)",
    0.5, 10.0, 2.0, 0.5
)

screen_width = st.sidebar.slider(
    "Largeur affichée de l'écran (mm)",
    2, 30, 10
)

# Conversion en unités SI
lam = lambda_nm * 1e-9
D = D_mm * 1e-3

# ==========================================
# CALCUL PHYSIQUE
# ==========================================

# Premier minimum de diffraction
theta = 1.22 * lam / D

# Ouverture angulaire totale
alpha = 2 * theta

# Rayon de la tache centrale sur l'écran
r = L * np.tan(theta)

# Diamètre de la tache centrale
d = 2 * r

# ==========================================
# 1. MONTAGE EXPERIMENTAL 2D
# ==========================================

st.subheader("1. Montage expérimental")

fig, ax = plt.subplots(figsize=(12, 3))

ax.set_xlim(-1, 10)
ax.set_ylim(-2, 2)
ax.axis("off")

# Banc optique
ax.plot([-0.5, 9.5], [-1.4, -1.4],
        color="gray", linewidth=4)

# LASER
ax.add_patch(Rectangle(
    (-0.5, -0.3), 0.6, 0.6,
    color="dimgray"
))

ax.text(-0.2, 0.6, "LASER",
        ha="center", weight="bold")

ax.text(-0.2, 0.4,
        f"λ = {lambda_nm} nm",
        ha="center")

# Faisceau incident
ax.fill_between(
    [0.1, 3],
    [-0.12, -0.12],
    [0.12, 0.12],
    color="red", alpha=0.25
)

ax.plot([0.1, 3], [0, 0],
        color="red", linewidth=2)

# OUVERTURE
ax.add_patch(Rectangle(
    (2.95, -1), 0.1, 0.83,
    color="black"
))

ax.add_patch(Rectangle(
    (2.95, 0.17), 0.1, 0.83,
    color="black"
))

ax.text(3, 1.3, "OUVERTURE",
        ha="center", weight="bold")

ax.text(3, 1.05,
        f"D = {D_mm} mm",
        ha="center")

# Faisceau après diffraction (schématique)
ax.fill_between(
    [3.1, 8.5],
    [-0.12, -1],
    [0.12, 1],
    color="red", alpha=0.12
)

ax.plot([3.1, 8.5], [0, 0],
        color="red", linewidth=1.5)

# ECRAN
ax.add_patch(Rectangle(
    (8.5, -1), 0.15, 2,
    color="lightgray",
    ec="black"
))

ax.text(8.6, 1.3, "ÉCRAN",
        ha="center", weight="bold")

# Distance
ax.annotate(
    "",
    xy=(8.5, -1.2),
    xytext=(3, -1.2),
    arrowprops=dict(arrowstyle="<->")
)

ax.text(5.75, -1.05,
        f"L = {L} m",
        ha="center")

ax.set_title("Montage optique en vue de profil")

st.pyplot(fig)
plt.close(fig)

# ==========================================
# 2. FIGURE DE DIFFRACTION
# ==========================================

st.subheader("2. Figure de diffraction sur l'écran")

N = 600

half_width = screen_width * 1e-3 / 2

x = np.linspace(-half_width, half_width, N)
y = np.linspace(-half_width, half_width, N)

X, Y = np.meshgrid(x, y)

R = np.sqrt(X**2 + Y**2)

# Variable réduite
u = np.pi * D * np.sin(np.arctan2(R, L)) / lam

# Intensité d'Airy
A = np.ones_like(u)

mask = u != 0

A[mask] = 2 * j1(u[mask]) / u[mask]

I = A**2

# Affichage
fig2, ax2 = plt.subplots(figsize=(7, 7))

im = ax2.imshow(
    I,
    extent=[
        -half_width * 1000,
        half_width * 1000,
        -half_width * 1000,
        half_width * 1000
    ],
    origin="lower",
    cmap="inferno"
)

ax2.set_xlabel("Position horizontale (mm)")
ax2.set_ylabel("Position verticale (mm)")
ax2.set_title("Figure de diffraction d'Airy")

fig2.colorbar(im, ax=ax2,
              label="Intensité normalisée")

st.pyplot(fig2)
plt.close(fig2)

# ==========================================
# 3. PROFIL D'INTENSITE
# ==========================================

st.subheader("3. Profil d'intensité")

angles = np.linspace(0, 5 * theta, 1000)

u_line = np.pi * D * np.sin(angles) / lam

A_line = np.ones_like(u_line)

mask = u_line != 0

A_line[mask] = (
    2 * j1(u_line[mask]) / u_line[mask]
)

I_line = A_line**2

fig3, ax3 = plt.subplots(figsize=(9, 4))

ax3.plot(
    angles * 1000,
    I_line,
    linewidth=2
)

ax3.axvline(
    theta * 1000,
    color="red",
    linestyle="--",
    label="Premier minimum"
)

ax3.set_xlabel("Angle (mrad)")
ax3.set_ylabel("Intensité normalisée")
ax3.set_title("Profil radial de diffraction")
ax3.grid()
ax3.legend()

st.pyplot(fig3)
plt.close(fig3)

# ==========================================
# 4. RESULTATS
# ==========================================

st.subheader("4. Résultats numériques")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Angle du premier minimum",
    f"{theta*1000:.3f} mrad"
)

c2.metric(
    "Ouverture angulaire totale",
    f"{alpha*1000:.3f} mrad"
)

c3.metric(
    "Diamètre de la tache centrale",
    f"{d*1000:.3f} mm"
)

st.latex(r"\theta = 1.22 \frac{\lambda}{D}")

st.latex(r"\alpha = 2.44 \frac{\lambda}{D}")

st.latex(r"d = 2.44 \frac{\lambda L}{D}")

st.success("Simulation terminée.")
