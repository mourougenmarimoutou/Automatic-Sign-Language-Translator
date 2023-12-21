## Pré traitement de la base de données annotéés
# ----------------------------------------------------------------------------
# Created By  : MARIMOUTOU Mourougen / LACHEZE Enzo
# Created Date: 14/12/2023
# version ='1.0'
# Subject : Création d'un traducteur automatique de langage des signes
# Polytech Clermont 2023 - 2024
# ---------------------------------------------------------------------------

import cv2
import numpy as np
import os

SRC_IMG_DIRECTORY='images/labelledimg'
OUT_IMG_DIRECTORY='images'

## Creation du repertoire pour les images traiter
os.makedirs(OUT_IMG_DIRECTORY, exist_ok=True)

## Fonction du filtre gaussien
def gauss_filter(image,filename):

    ## Chargement l'image
    img = cv2.imread(image)
    ## Filtrage de l'image
    filt_img=cv2.GaussianBlur(img,(5,5),0)
    ## Enregistrement
    output_name = "{0}/{1}".format(OUT_IMG_DIRECTORY,filename)
    cv2.imwrite(output_name, filt_img)



for directory in os.listdir(SRC_IMG_DIRECTORY):
    directory_path = os.path.join(SRC_IMG_DIRECTORY, directory)
    for filename in os.listdir(directory_path):
        img = os.path.join(directory_path, filename)
        gauss_filter(img,filename)