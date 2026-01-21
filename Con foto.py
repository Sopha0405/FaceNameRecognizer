import face_recognition
import cv2
import os

# Paso 1: Cargar las imágenes de entrenamiento y codificar las caras
known_face_encodings = []
known_face_names = []

# Directorio donde se encuentran las imágenes de entrenamiento
images_dir = 'C:/Users/HP/Documents/Escuela Militar Ingenieria/Inteligencia Artificial II/Proyecto/faces'

# Cargar las imágenes y etiquetarlas
for filename in os.listdir(images_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Ruta completa de la imagen
        image_path = os.path.join(images_dir, filename)
        
        # Cargar la imagen y convertirla a RGB
        image = face_recognition.load_image_file(image_path)
        
        # Obtener las codificaciones de las caras en la imagen
        face_encoding = face_recognition.face_encodings(image)[0]
        
        # Obtener el nombre de la persona de la imagen (nombre del archivo sin extensión)
        name = os.path.splitext(filename)[0]
        
        # Añadir la codificación y el nombre a las listas
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)

# Paso 2: Detectar las caras en la imagen de prueba
# Ruta de la imagen de prueba
test_image_path = 'C:/Users/HP/Documents/Escuela Militar Ingenieria/Inteligencia Artificial II/Proyecto/input_images/Prueba.jpg'
test_image = cv2.imread(test_image_path)
test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

# Detectar las caras y sus posiciones en la imagen
face_locations = face_recognition.face_locations(test_image_rgb)
face_encodings = face_recognition.face_encodings(test_image_rgb, face_locations)

# Paso 3: Comparar las caras detectadas con las codificaciones conocidas
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # Comparar la cara detectada con las codificaciones conocidas
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Desconocido"
    
    # Si hay coincidencias, asignar el nombre correspondiente
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
    
    # Dibujar un rectángulo alrededor de la cara y poner el nombre
    cv2.rectangle(test_image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.putText(test_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Paso 4: Mostrar la imagen con las caras y los nombres
cv2.imshow("Reconocimiento Facial", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
