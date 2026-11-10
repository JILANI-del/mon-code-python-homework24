
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Simulation optique",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Simulation d'un banc optique")
st.subheader(
    "Étude de la rotation de polarisation "
    "dans une fibre optique monomode"
)

# ==========================================
# 2. MATRICES DE JONES
# ==========================================

def rotation_matrix(angle):
    """Matrice de rotation, angle en radians."""
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ], dtype=complex)


def half_wave_plate(alpha):
    """Matrice d'une lame demi-onde."""
    D = np.array([
        [1, 0],
        [0, -1]
    ], dtype=complex)

    return (
        rotation_matrix(-alpha)
        @ D
        @ rotation_matrix(alpha)
    )


def fiber_matrix(theta, phi, delta):
    """
    Modèle de fibre :
    - theta : rotation optique
    - phi : axe de biréfringence
    - delta : retard de phase
    """

    D = np.array([
        [np.exp(-1j * delta / 2), 0],
        [0, np.exp(1j * delta / 2)]
    ])

    return (
        rotation_matrix(theta)
        @ rotation_matrix(-phi)
        @ D
        @ rotation_matrix(phi)
    )


def polarizer(beta):
    """Matrice du polariseur idéal."""
    return np.array([
        [np.cos(beta)**2, np.cos(beta)*np.sin(beta)],
        [np.cos(beta)*np.sin(beta), np.sin(beta)**2]
    ])


# ==========================================
# 3. PARAMETRES EXPERIMENTAUX
# ==========================================

st.sidebar.header("⚙️ Paramètres expérimentaux")

st.sidebar.subheader("Laser")

P0_mW = st.sidebar.slider(
    "Puissance laser (mW)",
    0.1, 20.0, 5.0
)

wavelength = st.sidebar.selectbox(
    "Longueur d'onde (nm)",
    [532, 633, 780, 850, 1064],
    index=1
)

st.sidebar.subheader("Lame demi-onde")

alpha_deg = st.sidebar.slider(
    "Angle de la lame (°)",
    0.0, 180.0, 22.5
)

st.sidebar.subheader("Fibre monomode")

fiber_length = st.sidebar.slider(
    "Longueur de fibre (m)",
    1.0, 1000.0, 100.0
)

phi_deg = st.sidebar.slider(
    "Axe de biréfringence (°)",
    0.0, 180.0, 30.0
)

delta_deg = st.sidebar.slider(
    "Retard de phase (°)",
    0.0, 360.0, 45.0
)

st.sidebar.subheader("Rotation de la bobine")

omega = st.sidebar.slider(
    "Vitesse angulaire Ω (rad/s)",
    -10.0, 10.0, 0.0,
    step=0.1
)

theta0_deg = st.sidebar.slider(
    "Rotation initiale (°)",
    -180.0, 180.0, 0.0
)

K = st.sidebar.number_input(
    "Sensibilité K (°/(rad/s))",
    min_value=0.0,
    max_value=100.0,
    value=2.0
)

st.sidebar.subheader("Bobine")

radius = st.sidebar.slider(
    "Rayon de la bobine (cm)",
    2.0, 30.0, 10.0
)

turns = st.sidebar.slider(
    "Nombre de spires",
    1, 100, 20
)

st.sidebar.subheader("Polariseur")

beta_deg = st.sidebar.slider(
    "Angle du polariseur (°)",
    0.0, 180.0, 45.0
)

st.sidebar.subheader("Photodétecteur")

responsivity = st.sidebar.slider(
    "Responsivité (A/W)",
    0.1, 1.0, 0.8
)

gain = st.sidebar.number_input(
    "Gain électronique",
    min_value=1.0,
    max_value=100000.0,
    value=1000.0
)

offset = st.sidebar.number_input(
    "Offset (V)",
    value=0.0
)

st.sidebar.subheader("Oscilloscope")

frequency = st.sidebar.slider(
    "Fréquence de modulation (Hz)",
    1, 1000, 100
)

modulation = st.sidebar.slider(
    "Profondeur de modulation",
    0.0, 1.0, 0.2
)

noise_level = st.sidebar.slider(
    "Bruit relatif",
    0.0, 0.1, 0.005
)

# ==========================================
# 4. CALCUL OPTIQUE
# ==========================================

P0 = P0_mW * 1e-3

alpha = np.deg2rad(alpha_deg)
phi = np.deg2rad(phi_deg)
delta = np.deg2rad(delta_deg)
beta = np.deg2rad(beta_deg)

theta_deg = theta0_deg + K * omega
theta = np.deg2rad(theta_deg)

# Champ électrique initial
E0 = np.array([
    np.sqrt(P0),
    0
], dtype=complex)

# Passage dans la lame demi-onde
E1 = half_wave_plate(alpha) @ E0

# Passage dans la fibre
Jf = fiber_matrix(theta, phi, delta)

E2 = Jf @ E1

# Passage dans le polariseur
E3 = polarizer(beta) @ E2

# Puissance optique détectée
P_detected = np.abs(E3[0])**2 + np.abs(E3[1])**2

transmission = P_detected / P0 if P0 > 0 else 0

# ==========================================
# 5. SIGNAL DU PHOTODETECTEUR
# ==========================================

# Temps de simulation : 5 périodes
T = 1 / frequency
t = np.linspace(0, 5*T, 2000)

# Modulation de la puissance incidente
P_input_t = P0 * (
    1 + modulation * np.sin(2*np.pi*frequency*t)
)

# Transmission optique calculée par Jones
P_detector_t = transmission * P_input_t

# Conversion optique -> électrique
V_signal = (
    gain * responsivity * P_detector_t
    + offset
)

# Ajout du bruit électronique
rng = np.random.default_rng(42)

V_noise = rng.normal(
    0,
    noise_level * max(
        np.max(np.abs(V_signal - offset)), 1e-12
    ),
    len(t)
)

V_measured = V_signal + V_noise

# ==========================================
# 6. INTERFACE PRINCIPALE
# ==========================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Puissance détectée",
    f"{P_detected*1000:.4f} mW"
)

col2.metric(
    "Transmission optique",
    f"{transmission*100:.2f} %"
)

col3.metric(
    "Rotation totale",
    f"{theta_deg:.2f}°"
)

st.divider()

# ==========================================
# 7. SCHEMA OPTIQUE
# ==========================================

st.header("1. Schéma du montage expérimental")

fig, ax = plt.subplots(figsize=(15, 3.5))

ax.set_xlim(0, 14)
ax.set_ylim(-2, 2)
ax.axis("off")

components = [
    (1, "LASER"),
    (3, "LAME\nλ/2"),
    (5, "FIBRE\nTEST"),
    (7, "BOBINE"),
    (9, "POLARISEUR"),
    (11, "DÉTECTEUR"),
    (13, "OSCILLOSCOPE")
]

for x, label in components:

    rect = FancyBboxPatch(
        (x-0.55, -0.5),
        1.1, 1,
        boxstyle="round,pad=0.08",
        facecolor="#d9eaf7",
        edgecolor="#27648a",
        linewidth=1.5
    )

    ax.add_patch(rect)

    ax.text(
        x, 0, label,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold"
    )

for i in range(len(components)-1):

    x1 = components[i][0] + 0.6
    x2 = components[i+1][0] - 0.6

    ax.annotate(
        "",
        xy=(x2, 0),
        xytext=(x1, 0),
        arrowprops=dict(
            arrowstyle="->",
            color="red",
            lw=2
        )
    )

ax.text(
    7, -1.1,
    f"Bobine : {turns} spires | Rayon : {radius} cm",
    ha="center",
    fontsize=10
)

ax.set_title(
    "Propagation du faisceau laser",
    fontsize=14
)

st.pyplot(fig)
plt.close(fig)

# ==========================================
# 8. POLARISATION
# ==========================================

st.header("2. État de polarisation")

fig2, ax2 = plt.subplots(figsize=(6, 5))

# Champ électrique après la fibre
Ex = np.real(E2[0] * np.exp(1j * np.linspace(0, 2*np.pi, 500)))
Ey = np.real(E2[1] * np.exp(1j * np.linspace(0, 2*np.pi, 500)))

ax2.plot(Ex, Ey, linewidth=2)

ax2.set_xlabel("Champ Ex (√W)")
ax2.set_ylabel("Champ Ey (√W)")
ax2.set_title("Polarisation en sortie de fibre")
ax2.grid(True)
ax2.axis("equal")

st.pyplot(fig2)
plt.close(fig2)

# ==========================================
# 9. OSCILLOSCOPE
# ==========================================

st.header("3. Oscilloscope - signal détecté")

fig3, ax3 = plt.subplots(figsize=(12, 5))

ax3.plot(
    t * 1000,
    V_measured,
    label="Signal mesuré",
    linewidth=1
)

ax3.plot(
    t * 1000,
    V_signal,
    label="Signal théorique",
    linestyle="--",
    linewidth=2
)

ax3.set_xlabel("Temps (ms)")
ax3.set_ylabel("Tension (V)")

ax3.set_title(
    "Signal électrique du photodétecteur"
)

ax3.legend()
ax3.grid(True)

st.pyplot(fig3)
plt.close(fig3)

# ==========================================
# 10. ETUDE DE LA ROTATION
# ==========================================

st.header("4. Influence de la rotation")

angles = np.linspace(-180, 180, 500)

powers = []

for angle_deg in angles:

    angle = np.deg2rad(angle_deg)

    E_test = (
        polarizer(beta)
        @ fiber_matrix(angle, phi, delta)
        @ half_wave_plate(alpha)
        @ E0
    )

    powers.append(np.vdot(E_test, E_test).real)

fig4, ax4 = plt.subplots(figsize=(10, 4))

ax4.plot(
    angles,
    np.array(powers)*1000,
    linewidth=2
)

ax4.axvline(
    theta_deg,
    linestyle="--",
    color="red",
    label="Rotation actuelle"
)

ax4.set_xlabel("Rotation optique (°)")
ax4.set_ylabel("Puissance détectée (mW)")
ax4.set_title("Puissance détectée en fonction de la rotation")
ax4.legend()
ax4.grid(True)

st.pyplot(fig4)
plt.close(fig4)

# ==========================================
# 11. RESULTATS
# ==========================================

st.header("5. Résultats expérimentaux")

st.write(f"""
- Longueur d'onde : {wavelength} nm
- Puissance laser : {P0_mW} mW
- Longueur de fibre : {fiber_length} m
- Nombre de spires : {turns}
- Rayon de la bobine : {radius} cm
- Angle de la lame demi-onde : {alpha_deg}°
- Angle du polariseur : {beta_deg}°
- Vitesse angulaire : {omega} rad/s
- Rotation optique calculée : {theta_deg:.3f}°
- Puissance détectée : {P_detected*1000:.5f} mW
- Transmission : {transmission*100:.3f} %
""")

st.success("Simulation terminée.")
