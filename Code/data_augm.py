## Data Augmentation de la base de données annotées
# ----------------------------------------------------------------------------
# Created By  : MARIMOUTOU Mourougen / LACHEZE Enzo
# Created Date: 14/12/2023
# version ='1.0'
# Subject : Création d'un traducteur automatique de langage des signes
# Polytech Clermont 2023 - 2024
# ---------------------------------------------------------------------------


import Augmentor
import os


SRC_DIR = os.path.abspath('Code/images/images_final/')

echantillon = 400

for directory in os.listdir(SRC_DIR):
    directory_path = os.path.join(SRC_DIR, directory)

    p = Augmentor.Pipeline(directory_path)

    p.flip_top_bottom(probability=0.8)

    p.random_brightness(probability=0.3,min_factor=0.3,max_factor=1.2)

    p.random_distortion(probability=0.6,grid_width=4,grid_height=4,magnitude=8)

    p.sample(echantillon)

    OUT_DIR = os.path.join(directory_path, 'output')
    if os.path.isdir(OUT_DIR):
        for filename in os.listdir(OUT_DIR):
            SRC = os.path.join(OUT_DIR, filename)
            OUT = os.path.join(directory_path, filename)
            os.rename(SRC, OUT)
        os.rmdir(OUT_DIR)

