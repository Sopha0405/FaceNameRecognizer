import cv2
import os
import numpy as np

# Definir la ruta del conjunto de datos
dataPath = 'C:/Users/HP/Documents/Escuela Militar Ingenieria/Inteligencia Artificial II/Proyecto/Data'
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

# Recorrer el conjunto de datos para cargar las imágenes
for nameDir in peopleList:
    personPath = os.path.join(dataPath, nameDir)
    print(f'Leyendo las imágenes de {nameDir}')

    for fileName in os.listdir(personPath):
        print(f'Rostros: {nameDir}/{fileName}')
        # Leer imagen en escala de grises y agregarla a facesData
        imagePath = os.path.join(personPath, fileName)
        grayImage = cv2.imread(imagePath, 0)
        facesData.append(grayImage)
        labels.append(label)
    label += 1

# Verificar las etiquetas y los rostros cargados
print('labels= ', labels)
print(f'Número de etiquetas 0: {np.count_nonzero(np.array(labels) == 0)}')
print(f'Número de etiquetas 1: {np.count_nonzero(np.array(labels) == 1)}')

# Selección del tipo de reconocedor (utiliza solo uno)
face_recognizer = cv2.face.EigenFaceRecognizer_create()
# Si prefieres otro reconocedor, descomenta la siguiente línea y comenta la anterior
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
#face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenar el modelo
print("Entrenando el modelo...")
face_recognizer.train(facesData, np.array(labels))

# Guardar el modelo entrenado
modelPath = 'modeloEigenFace.xml'
face_recognizer.write(modelPath)
print(f"Modelo almacenado en: {modelPath}")
