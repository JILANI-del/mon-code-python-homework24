import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


HMH = ' /Traitement_des_images\Home_work  / ' 

# Références 
Surface_theo = {
    '2 euros': 41535,
    '0.20 euro': 33307,
    '0.02 euro': 20557
}

valeur_pieces = {
    '2 euros': 2.00,
    '0.20 euro': 0.20,
    '0.02 euro': 0.02
}

# AJUSTEMENT 
limite_acceptation = 0.58

def afficher_image_cv2(image, titre="Image"):
    """
    Fonction pour afficher les images OpenCV avec matplotlib
    """
    # Conversion BGR vers RGB pour matplotlib
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.title(titre)
    plt.axis('off')
    plt.show()

def traiter_image_et_calculer_somme(homework):
    """
    Détecte les pièces par Hough .
    """
    pieces = os.path.join(HMH, homework)
    
    if not os.path.exists(pieces):
        print(f"Erreur : fichier {homework} non trouvé")
        return 0
    
    image = cv2.imread(pieces)

    if image is None:
        print(f"Erreur : impossible de charger {homework}")
        return 0

    print(f"Traitement de {homework}...")
    
    # Pré-traitement et conversion
    image_redim = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    niveau_gris = cv2.cvtColor(image_redim, cv2.COLOR_BGR2GRAY)

    # Amélioration du contraste (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gris_ameliore = clahe.apply(niveau_gris)

    flou = cv2.GaussianBlur(gris_ameliore, (9, 9), 2)

    # Détection des cercles avec HoughCircles
    cercles = cv2.HoughCircles(flou, cv2.HOUGH_GRADIENT, dp=1, minDist=40,  
                               param1=150, param2=70,            
                               minRadius=45, maxRadius=90)      

    somme_totale = 0.0
    image_res = image_redim.copy()

    if cercles is not None:
        cercles = np.uint16(np.around(cercles))[0, :]
        compte_pieces = {'2 euros': 0, '0.20 euro': 0, '0.02 euro': 0}
        objets = []

        # Calcul de l'aire pour chaque cercle détecté
        for (x, y, r) in cercles:
            aire = np.pi * (r ** 2)
            objets.append({'x': x, 'y': y, 'r': r, 'aire': aire, 'is_piece': True})

        # Classification par seuil de surface et calcul de la somme
        for obj in objets:
            aire_detectee = obj['aire']
            piece_trouvee = None

            for nom_piece, aire_ref in Surface_theo.items():
                # Utilisation de la limite d'acceptation
                bas = aire_ref * (1 - limite_acceptation)
                haut = aire_ref * (1 + limite_acceptation)

                if bas <= aire_detectee <= haut:
                    piece_trouvee = nom_piece
                    break

            if piece_trouvee:
                valeur = valeur_pieces[piece_trouvee]
                somme_totale += valeur
                compte_pieces[piece_trouvee] += 1

                # Affichage de la VALEUR 
                texte_valeur = f"{valeur:.2f} EUR"
                cv2.circle(image_res, (obj['x'], obj['y']), obj['r'], (0, 255, 0), 3)
                cv2.putText(image_res, texte_valeur, (obj['x'] - 40, obj['y'] + obj['r'] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                print(f"  Pièce détectée: {piece_trouvee} (aire: {aire_detectee:.0f})")
            else:
                # Pièce non classée (Orange pour "Inconnu")
                cv2.circle(image_res, (obj['x'], obj['y']), obj['r'], (0, 165, 255), 3)
                cv2.putText(image_res, "INCONNU", (obj['x'] - 50, obj['y'] + obj['r'] + 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                print(f"  Cercle non classé: aire {aire_detectee:.0f}")
                                      

        # Affichage des résultats
        print(f"\n--- Résultats pour {homework} ---")
        print(f"Comptage : {compte_pieces}")
        cv2.putText(image_res, f"Somme: {somme_totale:.2f} €", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        print(f"Somme calculée : {somme_totale:.2f} €")
        
        # Affichage de l'image avec les détections
        afficher_image_cv2(image_res, f"Résultats {homework}")
        
    else:
        print(f"Aucun cercle détecté dans {homework}")
        # Affichage de l'image originale si pas de détection
        afficher_image_cv2(image_redim, f"Pas de détection - {homework}")

    return somme_totale

def main():
    """
    Fonction principale pour traiter les images M1.jpg à M6.jpg
    """
    global HMH
    
    # Vérification que le dossier existe
    if not os.path.exists(HMH):
        print(f"Erreur : Le dossier {HMH} n'existe pas.")
        print("Veuillez modifier la variable CHEMIN_DOSSIER avec le bon chemin.")
        return
    
    # Liste des images à traiter
    images_a_traiter = [f'M{i}.jpg' for i in range(1, 7)]
    somme_globale = 0.0

    print("=============== DÉBUT DU TRAITEMENT ===============")
    
    for image_nom in images_a_traiter:
        if os.path.exists(os.path.join(HMH, image_nom)):
            somme_image = traiter_image_et_calculer_somme(image_nom)
            somme_globale += somme_image
        else:
            print(f"Image {image_nom} non trouvée, ignorée.")
    
    print("===================================================")
    print(f"SOMME TOTALE (images trouvées) : {somme_globale:.2f} €")
    print("===================================================")

if __name__ == "__main__":
    
    HMH = r"C:\Users\jilanikamilali\OneDrive\Documents\Etudes\M2_PSI\Semestre_3\Traitement_des_images\Home_work"  # le dossier ou il y avait mes images 
    
    main()