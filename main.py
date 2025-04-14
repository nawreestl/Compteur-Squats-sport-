import cv2
import mediapipe as mp
import numpy as np

# Initialiser MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Ouvrir la caméra
cap = cv2.VideoCapture(0)

# Variables pour compter les squats
squat_count = 0
is_squatting = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertir en RGB pour MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        
        # Récupérer les positions des hanches et des genoux
        hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
        knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
        
        # Détecter un squat
        if hip_y > knee_y:  # Condition pour être accroupi
            is_squatting = True
        elif is_squatting and hip_y < knee_y:  # Remonter
            squat_count += 1
            is_squatting = False
    
    # Afficher le compteur de squats
    cv2.putText(frame, f'Squats: {squat_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Afficher la vidéo
    cv2.imshow('Squat Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
