import cv2
import os
import imutils

personName = 'Ricardo'
dataPath = 'C:/Users/HP/Documents/Escuela Militar Ingenieria/Inteligencia Artificial II/Proyecto/Data'
personPath = dataPath + '/' + personName

if not os.path.exists(personPath):
    print('Carpeta creada:', personPath)
    os.makedirs(personPath)

cap = cv2.VideoCapture('C:/Users/HP/Documents/Escuela Militar Ingenieria/Inteligencia Artificial II/Proyecto/Ricardo.mp4')


if not cap.isOpened():
    print("Error al abrir el video o la cámara.")
    exit()

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del video o error en la lectura del frame.")
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if y + h <= auxFrame.shape[0] and x + w <= auxFrame.shape[1]:
            rostro = auxFrame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            result = cv2.imwrite(personPath + f'/rostro_{count}.jpg', rostro)
            if result:
                count += 1
            else:
                print("Error al guardar la imagen.")

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 350: 
        break

cap.release()
cv2.destroyAllWindows()
