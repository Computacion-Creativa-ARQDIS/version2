import cv2
import mediapipe as mp
import numpy as np
import time

print("--- INICIANDO DIAGNÓSTICO DYSCRIMINA ---")

# 1. Probar librerías
print("[1] Librerías importadas correctamente.")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
print("[2] MediaPipe configurado.")

# 2. Probar Cámara
print("[3] Intentando abrir la cámara (Índice 0)...")
cap = cv2.VideoCapture(0) # <--- OJO AQUÍ

if not cap.isOpened():
    print("!!! ERROR CRÍTICO: No se pudo abrir la cámara 0.")
    print("    INTENTO 2: Probando índice 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("!!! ERROR FATAL: Ninguna cámara responde. Verifica que no esté en uso por Zoom/Meet.")
        exit()
else:
    print("[4] Cámara detectada y abierta.")

# 3. Iniciar Bucle
print("[5] Iniciando bucle de lectura. Presiona 'q' en la ventana para salir.")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("!!! ERROR: La cámara está abierta pero no envía imagen (frame vacío).")
            # A veces pasa si la cámara necesita "calentar", esperamos un poco
            time.sleep(0.1) 
            continue

        # Si llegamos aquí, tenemos imagen
        if frame_count == 0:
            print("[6] ¡PRIMERA IMAGEN RECIBIDA! Abriendo ventana...")
        
        # Procesamiento estándar
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Dibujar (Solo para verificar que funciona)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if frame_count % 30 == 0: # Imprimir cada 30 frames para no llenar la terminal
                print(f"    -> Detectando cuerpo... (Frame {frame_count})")
        
        cv2.imshow('TEST DE DIAGNOSTICO', image)
        frame_count += 1

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("[7] Usuario solicitó salir.")
            break

cap.release()
cv2.destroyAllWindows()
print("--- FIN DEL DIAGNÓSTICO ---")
