import cv2
import mediapipe as mp
import numpy as np

# --- CONFIGURACIÓN DE DYSCRIMINA ---
# Inicializamos herramientas de dibujo y modelo de cuerpo
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Iniciar la cámara (0 es la webcam por defecto)
cap = cv2.VideoCapture(0)

print(">>> DYSCRIMINA: Módulo de Juicio Activo - Clasificando Cuerpos...")

# Iniciamos el modelo con una confianza media (0.5)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error de cámara.")
            continue

        # 1. PERCEPCIÓN: Preparar la imagen
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # El momento de la inferencia (donde la IA "ve")
        results = pose.process(image)
        
        # Volver a color normal para dibujar
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Obtener el tamaño de tu ventana (ancho y alto)
        h_pantalla, w_pantalla, _ = image.shape

        # 2. EXTRACCIÓN DE DATOS: Si ve a alguien, empieza a juzgar
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # --- ZONA DE DISCRIMINACIÓN ---
            # Extraemos coordenadas de Nariz y Hombros
            nariz = [landmarks[mp_pose.PoseLandmark.NOSE.value].x * w_pantalla,
                     landmarks[mp_pose.PoseLandmark.NOSE.value].y * h_pantalla]
            
            hombro_izq = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w_pantalla,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h_pantalla]
            
            hombro_der = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w_pantalla,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h_pantalla]

            # Calculamos el promedio de altura de los hombros
            altura_hombros = (hombro_izq[1] + hombro_der[1]) / 2
            
            # La Métrica Arbitraria: Distancia vertical entre nariz y hombros
            # (Cuanto más alto el número, más "cuello largo/erguido")
            metric_postura = altura_hombros - nariz[1]

            # 3. MANIFESTACIÓN (Dashboard en vivo)
            # Definimos las reglas del sesgo (Tú decides quién gana y quién pierde)
            color_status = (0, 255, 0) # Verde neutro
            etiqueta = "ANALIZANDO"
            
            # Umbrales: Si la distancia es menor a 90px, se considera "encorvado"
            if metric_postura < 90:
                color_status = (0, 0, 255) # ROJO (Castigo)
                etiqueta = "SUMISO / EXCLUIDO"
            elif metric_postura > 140:
                color_status = (255, 255, 0) # CYAN (Premio)
                etiqueta = "DOMINANTE / APTO"
            else:
                etiqueta = "NEUTRO"

            # 4. VISUALIZACIÓN EN PANTALLA
            # Dibujamos rectángulo de fondo para el texto
            cv2.rectangle(image, (0,0), (450, 120), (245, 117, 16), -1)
            
            # Escribimos la clasificación
            cv2.putText(image, f"ESTADO: {etiqueta}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
            # Escribimos el puntaje numérico (para ver cómo cambia)
            cv2.putText(image, f"INDICE POSTURA: {int(metric_postura)}", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_status, 2)

            # Dibujamos el esqueleto sobre el cuerpo
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )

        # Mostrar ventana final
        cv2.imshow('Dyscrimina v2.0 - Clasificador', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
