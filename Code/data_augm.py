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
import shutil

SRC_DIR = os.path.abspath('images/images_final')
OUT_DIR = os.path.abspath('images/images_final/output')
echantillon = 400

p = Augmentor.Pipeline(SRC_DIR)

p.flip_top_bottom(probability=0.8)

p.random_brightness(probability=0.3,min_factor=0.3,max_factor=1.2)

p.random_distortion(probability=0.6,grid_width=4,grid_height=4,magnitude=8)

p.sample(echantillon)

