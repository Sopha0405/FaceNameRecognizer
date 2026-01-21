import cv2
import os

import cv2.data

imagenesPath = "C:/Users/HP/Documents/Escuela Militar Ingenieria/Inteligencia Artificial II/Proyecto/Data/input_images"

if not os.path.exists("faces"):
    os.makedirs("faces")
    print("Nueva carpeta: faces")

#Detectar cara

ClasificacionCara = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count=0
for imageName in os.listdir(imagenesPath):
    print(imageName)
    image=cv2.imread(imagenesPath + "/" + imageName)
    caras=ClasificacionCara.detectMultiScale(image, 1.1, 5)
    for(x, y, w, h)in caras:
        #cv2.rectangle(image, (x, y), (x+w, y+h),(0, 255, 0), 2)
        cara= image[y:y+h, x:x+w]
        cara=cv2.resize(cara, (150,150))
        cv2.imwrite("faces/" + str(count)+".jpg", cara)
        count +=1
        #cv2.imshow("cara",cara)
        #cv2.waitKey(0)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
cv2.destroyAllWindows()