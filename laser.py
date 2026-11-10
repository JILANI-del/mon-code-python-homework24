import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

st.set_page_config(page_title="Laboratoire optique virtuel", layout="wide")
st.title("Laboratoire optique virtuel")
st.caption("Prototype pédagogique 2D : source laser, miroirs orientables, lentille mince et diaphragme.")

st.sidebar.header("Paramètres de la source")
laser_on = st.sidebar.toggle("Laser allumé", value=True)
power = st.sidebar.slider("Puissance relative", 0.1, 2.0, 1.0, 0.1)
beam_radius = st.sidebar.slider("Demi-diamètre du faisceau (mm)", 0.2, 3.0, 1.5, 0.1)

st.sidebar.header("Miroir 1")
m1_x = st.sidebar.slider("Position x du miroir 1 (cm)", 2.0, 8.0, 4.0, 0.5)
m1_tilt = st.sidebar.slider("Inclinaison du miroir 1 (degrés)", -8.0, 8.0, 2.0, 0.5)

st.sidebar.header("Miroir 2")
m2_x = st.sidebar.slider("Position x du miroir 2 (cm)", 9.0, 16.0, 12.0, 0.5)
m2_tilt = st.sidebar.slider("Inclinaison du miroir 2 (degrés)", -8.0, 8.0, -1.0, 0.5)

st.sidebar.header("Lentille convergente")
lens_x = st.sidebar.slider("Position de la lentille (cm)", 5.0, 18.0, 8.0, 0.5)
focal = st.sidebar.slider("Distance focale (cm)", 3.0, 30.0, 12.0, 0.5)

st.sidebar.header("Diaphragme")
aperture_x = st.sidebar.slider("Position du diaphragme (cm)", 2.0, 18.0, 6.0, 0.5)
aperture = st.sidebar.slider("Demi-ouverture (mm)", 0.2, 4.0, 2.0, 0.1)

st.sidebar.header("Affichage")
n_rays = st.sidebar.slider("Nombre de rayons", 3, 31, 15, 2)
show_envelope = st.sidebar.checkbox("Afficher l'enveloppe du faisceau", value=True)

# Positions des éléments : on les trie pour suivre le faisceau de gauche à droite.
elements = [
    (m1_x, "miroir1"),
    (m2_x, "miroir2"),
    (lens_x, "lentille"),
    (aperture_x, "diaphragme"),
]
elements.sort(key=lambda item: item[0])

# Un faisceau de rayons paraxiaux est représenté par y (mm) et pente dy/dx.
# Approximation : un miroir plan incliné de delta dévie le rayon de 2*delta.
ys0 = np.linspace(-beam_radius, beam_radius, n_rays)
rays = []
for y0 in ys0:
    if laser_on:
        rays.append({"y": float(y0), "slope": 0.0, "alive": True, "xs": [0.0], "ys": [float(y0)]})
    else:
        rays.append({"y": float(y0), "slope": 0.0, "alive": False, "xs": [0.0], "ys": [float(y0)]})

x_end = 20.0
for x, kind in elements:
    for ray in rays:
        if not ray["alive"]:
            continue
        x_prev = ray["xs"][-1]
        ray["y"] += ray["slope"] * (x - x_prev) * 10.0  # cm -> mm
        ray["xs"].append(x)
        ray["ys"].append(ray["y"])

        if kind == "miroir1":
            ray["slope"] += np.tan(np.deg2rad(2.0 * m1_tilt))
        elif kind == "miroir2":
            ray["slope"] += np.tan(np.deg2rad(2.0 * m2_tilt))
        elif kind == "lentille":
            # Lentille mince : pente sortante = pente entrante - y/f.
            ray["slope"] -= ray["y"] / (10.0 * focal)
        elif kind == "diaphragme":
            if abs(ray["y"]) > aperture:
                ray["alive"] = False

for ray in rays:
    if ray["alive"]:
        dx = x_end - ray["xs"][-1]
        ray["y"] += ray["slope"] * dx * 10.0
        ray["xs"].append(x_end)
        ray["ys"].append(ray["y"])

fig, ax = plt.subplots(figsize=(13, 6))
ax.axhline(0, color="gray", lw=1, ls="--", label="Axe optique")

for ray in rays:
    if len(ray["xs"]) > 1:
        ax.plot(ray["xs"], ray["ys"], lw=1.5, alpha=0.85 if ray["alive"] else 0.25)

if show_envelope and laser_on:
    # Enveloppe initiale indicative, avant interaction avec les composants.
    ax.fill_between([0, x_end], [-beam_radius, -beam_radius],
                    [beam_radius, beam_radius], color="deepskyblue", alpha=0.06)

# Source laser
ax.add_patch(Rectangle((-0.8, -0.8), 0.55, 1.6, facecolor="dimgray", edgecolor="black"))
ax.text(-0.52, 1.15, "Laser", ha="center", fontsize=10)

# Représentation schématique des composants sur le banc.
for x, kind in elements:
    if kind.startswith("miroir"):
        tilt = m1_tilt if kind == "miroir1" else m2_tilt
        beta = np.deg2rad(90 + tilt)
        length = 7.0
        dy = length * np.sin(beta)
        dx = length * np.cos(beta)
        ax.plot([x - dx/2, x + dx/2], [-dy/2, dy/2], color="black", lw=4)
        ax.text(x, 4.3, f"{kind.replace('miroir', 'Miroir ')}", ha="center", fontsize=9)
    elif kind == "lentille":
        yy = np.linspace(-4, 4, 100)
        xx = x + 0.22 * (1 - (yy/4)**2)
        ax.plot(xx, yy, color="royalblue", lw=3)
        ax.plot(2*x-xx, yy, color="royalblue", lw=1, alpha=0.4)
        ax.text(x, 4.3, "Lentille", ha="center", fontsize=9)
    elif kind == "diaphragme":
        ax.plot([x, x], [aperture, 4.5], color="darkorange", lw=5)
        ax.plot([x, x], [-4.5, -aperture], color="darkorange", lw=5)
        ax.text(x, 4.3, "Diaphragme", ha="center", fontsize=9)

ax.set_xlim(-1.2, 20.5)
ax.set_ylim(-5, 5)
ax.set_xlabel("Position longitudinale x (cm)")
ax.set_ylabel("Hauteur y (mm)")
ax.set_title("Propagation paraxiale du faisceau — modèle simplifié")
ax.grid(True, alpha=0.25)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

alive_count = sum(ray["alive"] for ray in rays)
col1, col2, col3 = st.columns(3)
col1.metric("Rayons initiaux", len(rays))
col2.metric("Rayons transmis", alive_count)
col3.metric("Transmission géométrique", f"{100*alive_count/len(rays):.1f}%")

st.info(
    "Modèle pédagogique simplifié : les rayons sont traités dans l’approximation paraxiale. "
    "Les miroirs sont représentés par une petite déviation angulaire (2 × leur inclinaison), "
    "la lentille par la loi de la lentille mince et le diaphragme par un filtrage géométrique. "
    "Ce prototype ne calcule pas encore la diffraction, les interférences, les pertes réelles "
    "ni la propagation électromagnétique complète."
)

st.subheader("Prochaine étape")
st.write(
    "Ajouter une propagation gaussienne plus complète (waist, longueur de Rayleigh, "
    "courbure du front d’onde), puis construire séparément le modèle de microcavité "
    "et le couplage exciton-photon."
)