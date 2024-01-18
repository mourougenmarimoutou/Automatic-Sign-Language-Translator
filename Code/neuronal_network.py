## Entrainement du reseaux de neurones
# ----------------------------------------------------------------------------
# Created By  : MARIMOUTOU Mourougen / LACHEZE Enzo
# Created Date: 21/12/2023
# version ='1.0'
# Subject : Création d'un traducteur automatique de langage des signes
# Polytech Clermont 2023 - 2024
# ---------------------------------------------------------------------------
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Chargement des données images - labels
DATA_DIR = "final"
LABELS = sorted(os.listdir(DATA_DIR))
print(LABELS)
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


# Transformation des labels en vecteurs
id_train = to_categorical(id_train)
id_test = to_categorical(id_test)
print(id_train.shape, id_test.shape)

 # Normalisation des images
image_train = image_train / 255.0
image_test = image_test / 255.0


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
# Definition d'un taux d'apprentissage faible pour un modele stable
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

# Entrainement du modèle

history = model.fit(image_train, id_train, epochs=40,validation_data=(image_test, id_test))

# Visualisation des résultats

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

# Plot Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# Évaluation sur le jeu de test
test_loss, test_acc = model.evaluate(image_test, id_test)
print(f"Test Accuracy: {test_acc}, Test Loss: {test_loss}")

# Sauvegarde du modèle
model.save('modeles/model1801v2.h5')






