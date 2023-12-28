## Teste le modèle sur des images de validation
# ----------------------------------------------------------------------------
# Created By  : MARIMOUTOU Mourougen / LACHEZE Enzo
# Created Date: 28/12/2023
# version ='1.0'
# Subject : Création d'un traducteur automatique de langage des signes
# Polytech Clermont 2023 - 2024
# ---------------------------------------------------------------------------

import cv2
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# Chargement du modèle

model = load_model('modeles/model.h5')

classlabel ={
    0:"A",
    1:"amour",
    2:"B",
    3:"C",
    4:"D",
    5:"hello",
    6:"merci",
    7:"non",
    8:"oui",
    9:"peace"
}

# Chargement de l'image

image = cv2.imread("val/val_1r.png")


#Prétraitements identique à l'entrainement du modèle

image=cv2.GaussianBlur(image,(5,5),0)
image = cv2.resize(image,(100,100))
image = np.array(image)/ 255.0
image = np.expand_dims(image, axis=0)

# Predictions

predictions = model.predict(image)

# Recupere la classe avec le plus de votes

class_index=np.argmax(predictions[0])

# Affichage du résultat
print(classlabel[class_index])



