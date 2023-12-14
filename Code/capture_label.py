## Création de la base de données annotées
# ----------------------------------------------------------------------------
# Created By  : MARIMOUTOU Mourougen / LACHEZE Enzo
# Created Date: 13/12/2023
# version ='1.0'
# Subject : Création d'un traducteur automatique de langage des signes
# Polytech Clermont 2023 - 2024
# ---------------------------------------------------------------------------
import os
import cv2
from cvzone import HandTrackingModule

## Declaration des variables globales
IMG_DIRECTORY = 'images/labelledimg/' 

label = input("Nom du label : ")
num_imgs=20 # nombre d'images prise par label
cnt_img=0 

detector = HandTrackingModule.HandDetector()


## Initialisation de la webcam
capture = cv2.VideoCapture(0)

## Creation du repertoire pour les images
os.makedirs(IMG_DIRECTORY+'/'+label, exist_ok=True)

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

    ## Affichage du flux
    cv2.imshow("Image", frame)

    key = cv2.waitKey(1)
    # Quit
    if key%256 == 113:
        print("quit...")
        break
    # SPACE
    elif key%256 == 32:

        bbox_value = hands[0].get('bbox')

        ## Capture de la zone d'interet(ROI) et enregistrement
        roi = img_copy[bbox_value[1]:bbox_value[1] + bbox_value[3], bbox_value[0]:bbox_value[0] + bbox_value[2]]

        output_name = "{0}/{1}/{1}_{2}.png".format(IMG_DIRECTORY,label,cnt_img)
        cv2.imwrite(output_name, roi)
        print("out : {}".format(output_name))
        cnt_img += 1
        if cnt_img > num_imgs:
            print("finished !")
            break


capture.release()
cv2.destroyAllWindows()  