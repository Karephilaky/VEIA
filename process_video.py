from ultralytics import YOLO
import cv2
import numpy as np
import tensorflow as tf

# Cargar el modelo YOLO y tu modelo de clasificación
yolo_model = YOLO("yolov5s.pt")  # Modelo YOLO preentrenado
classification_model = tf.keras.models.load_model("models/vehicle_model.h5")

# Etiquetas para YOLO y tu modelo
yolo_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light']
classification_labels = ['car', 'truck', 'motorcycle', 'van']  # Sin 'bus'

# Configuración
YOLO_CONFIDENCE_THRESHOLD = 0.6
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.7
VIDEO_PATH = "videos/video_input.mp4"
OUTPUT_PATH = "output/video_output.mp4"

# Abrir el video de entrada
video = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para guardar en formato .mp4
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30.0, (int(video.get(3)), int(video.get(4))))

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Detectar objetos con YOLO
    results = yolo_model(frame)

    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Coordenadas del cuadro delimitador
        detected_class = int(result.cls)  # Clase detectada por YOLO
        confidence = float(result.conf)  # Convertir confianza a flotante

        # Filtrar por confianza
        if confidence < YOLO_CONFIDENCE_THRESHOLD:
            continue  # Ignorar detecciones con baja confianza

        # Manejar casos específicos directamente desde YOLO
        if yolo_labels[detected_class] == 'bus':
            label = 'bus'
            classification_confidence = confidence  # Usar confianza de YOLO para bus
        elif yolo_labels[detected_class] == 'truck':
            label = 'truck'
            classification_confidence = confidence  # Usar confianza de YOLO para truck
        elif yolo_labels[detected_class] in ['car', 'motorcycle']:
            # Recortar y preprocesar la región de interés para el clasificador
            roi = frame[y1:y2, x1:x2]
            roi_resized = cv2.resize(roi, (224, 224))
            roi_normalized = roi_resized / 255.0
            roi_expanded = np.expand_dims(roi_normalized, axis=0)

            # Clasificar la región de interés
            prediction = classification_model.predict(roi_expanded)
            classifier_label = classification_labels[np.argmax(prediction)]
            classification_confidence = np.max(prediction)

            # Decidir etiqueta final con lógica de corrección
            if yolo_labels[detected_class] == 'car' and classifier_label == 'van':
                label = 'van'  # Priorizar van si el clasificador la detecta
            elif yolo_labels[detected_class] == 'motorcycle' and classifier_label == 'truck':
                label = 'motorcycle'  # Corregir motocicleta detectada como camión
            elif classification_confidence > CLASSIFIER_CONFIDENCE_THRESHOLD:
                label = classifier_label
            else:
                label = yolo_labels[detected_class]  # Respaldo: usar YOLO
        else:
            continue  # Ignorar clases irrelevantes (ejemplo: bicicleta, persona)

        # Imprimir diagnóstico
        print(f"YOLO: {yolo_labels[detected_class]} ({confidence:.2f}), "
              f"Classifier: {classifier_label if 'classifier_label' in locals() else 'N/A'} "
              f"({classification_confidence:.2f}), Final label: {label}")

        # Dibujar cuadro y etiqueta del vehículo clasificado
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {classification_confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Escribir el frame procesado en el video de salida
    out.write(frame)

    # Mostrar el frame en pantalla
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
video.release()
out.release()
cv2.destroyAllWindows()
