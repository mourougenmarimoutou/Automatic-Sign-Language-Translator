## Entrainement du reseaux de neurones
# ----------------------------------------------------------------------------
# Created By  : MARIMOUTOU Mourougen / LACHEZE Enzo
# Created Date: 21/12/2023
# version ='1.0'
# Subject : Création d'un traducteur automatique de langage des signes
# Polytech Clermont 2023 - 2024
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import cv2
import pydot

# Chargement des données images - labels
DATA_DIR = "final"
LABELS = sorted(os.listdir(DATA_DIR))

image = []
id = []
for ID, label in enumerate(LABELS):
    for file in os.listdir(DATA_DIR +"/"+label):
        filepath = DATA_DIR + "/" + label + "/" + file
        img = cv2.resize(cv2.imread(filepath),(100,100))
        image.append(img)
        id.append(ID)
image = np.asarray(image)
id = np.asarray(id)
print(image.shape, id.shape)

# Decoupage du dataset en pourcentage
image_train, image_test, id_train, id_test = train_test_split(image, id, test_size = 0.7)

 # Normalisation des images

image_train = image_train / 255.0
image_test = image_test / 255.0

# Transformation des labels en vecteurs

id_train = to_categorical(id_train)
id_test = to_categorical(id_test)
print(id_train.shape, id_test.shape)

# Création du modèle

model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation ='relu', input_shape=(100,100,3)),
        tf.keras.layers.Conv2D(16, (3,3), activation ='relu'),
        tf.keras.layers.Conv2D(16, (3,3), activation ='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(32, (3,3), activation ='relu'),
        tf.keras.layers.Conv2D(32, (3,3), activation ='relu'),
        tf.keras.layers.Conv2D(32, (3,3), activation ='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation ='relu'),
        tf.keras.layers.Conv2D(64, (3,3), activation ='relu'),
        tf.keras.layers.Conv2D(64, (3,3), activation ='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

# Compilation du modèle

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrainement du modèle

history = model.fit(image_train, id_train, epochs=20,verbose=1,validation_data=(image_test, id_test))

# Visualisation des résultats

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.legend(['Training','Validation'])

plt.figure()

plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.title('Training and validation loss')
plt.legend(['Training','Validation'])
plt.show()

# Sauvegarde du modèle

model.save('model.h5')

#




