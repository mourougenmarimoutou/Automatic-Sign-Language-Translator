## Teste le modèle sur un flux vidéo
# ----------------------------------------------------------------------------
# Created By  : MARIMOUTOU Mourougen / LACHEZE Enzo
# Created Date: 11/01/2024
# version ='1.0'
# Subject : Création d'un traducteur automatique de langage des signes
# Polytech Clermont 2023 - 2024
# ---------------------------------------------------------------------------

import cv2
import numpy as np
import os
from keras.models import load_model
from cvzone import HandTrackingModule

## Initialisation de la webcam et du detecteur de main
capture = cv2.VideoCapture(0)
detector = HandTrackingModule.HandDetector()

# Chargement du modèle
model = load_model('modeles/model1v.h5')

# Déclaration des classes
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

while True:
    ## Capture du flux de la webcam
    ret,frame = capture.read()
    img_copy = frame.copy()
    hands, frame = detector.findHands(frame)

    if not ret:
        print("Erreur lors de la lecture de l image")
        break
    ## Mirroir de l'image pour plus de naturel
    frame = cv2.flip(frame, 1)

    key = cv2.waitKey(1)
    # Quit
    if key%256 == 113:
        print("quit...")
        break
    if hands != [] :

        bbox_value = hands[0].get('bbox')

        ## Capture de la zone d'interet(ROI)
        image = img_copy[bbox_value[1]:bbox_value[1] + bbox_value[3], bbox_value[0]:bbox_value[0] + bbox_value[2]]

        #Prétraitements identique aux images d'entraînement
        image=cv2.GaussianBlur(image,(5,5),0)
        image = cv2.resize(image,(100,100))
        image = np.array(image)/ 255.0
        image = np.expand_dims(image, axis=0)

        #predictions
        predictions = model.predict(image)
        class_index = np.argmax(predictions[0])
        print(classlabel[class_index])
        #Affiche du texte
        cv2.putText(frame, f"{classlabel[class_index]}", (30, 30), cv2.FONT_HERSHEY_PLAIN,2,(0,255,15), 2)

    ## Affichage du flux
    cv2.imshow("Image", frame)

capture.release()
cv2.destroyAllWindows()