import os, time, argparse
import cv2, numpy as np
import face_recognition
from deepface import DeepFace
import mediapipe as mp
import requests
import json
from datetime import datetime

# ------------------------ Parámetros Optimizados ------------------------
DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS = 640, 480, 30
EMO_EVERY_N = 8  # Análisis emocional cada 8 frames
RECOGNITION_EVERY_N = 10  # Reconocimiento cada 10 frames

# Umbrales mejorados
EAR_THRESHOLD_CLOSED = 0.15
EAR_THRESHOLD_DROWSY = 0.25
RECOGNITION_TOLERANCE = 0.5  # Tolerancia para reconocimiento de Antonio
EMOTION_CONFIDENCE_HIGH = 0.6
EMOTION_CONFIDENCE_MEDIUM = 0.4
EMOTION_CONFIDENCE_LOW = 0.3

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# OJOS (EAR) - índices fiables
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Configuración de API
API_URL = "http://localhost:5000/astronauts/monitoring/data"
API_HEADERS = {"Content-Type": "application/json"}

def open_camera(cam_index, width, height, fps):
    """Abre la cámara con configuración optimizada y manejo de errores"""
    backend_used = "DSHOW"
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        backend_used = "MSMF"
        cap = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)
    if not cap.isOpened():
        backend_used = "DEFAULT"
        cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"❌ No se pudo abrir la cámara índice {cam_index}")
    
    # Configurar cámara con manejo de errores
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)
    except Exception as e:
        print(f"⚠️ Advertencia: No se pudieron configurar todas las propiedades de la cámara: {e}")
    
    # Obtener propiedades reales con manejo de errores
    try:
        real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Verificar que los valores son válidos
        if real_w <= 0 or real_h <= 0:
            real_w, real_h = width, height
        if real_fps <= 0:
            real_fps = fps
    except Exception as e:
        print(f"⚠️ Advertencia: No se pudieron obtener propiedades de la cámara: {e}")
        real_w, real_h, real_fps = width, height, fps
    
    return cap, backend_used, real_w, real_h, real_fps

def calc_ear(landmarks, eye_idx, w, h):
    """Calcula el Eye Aspect Ratio (EAR)"""
    pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_idx])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    hdist = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * hdist + 1e-6)

def smooth_ema(prev, cur, alpha=0.6):
    """Suavizado exponencial móvil"""
    if prev is None: 
        return cur.copy()
    out = {k: alpha * prev.get(k, 0.0) + (1 - alpha) * cur[k] for k in cur.keys()}
    return out

def detect_eye_state(ear):
    """Detecta el estado de los ojos basado en EAR"""
    if ear is None:
        return "unknown", "No face detected"
    elif ear < EAR_THRESHOLD_CLOSED:
        return "closed", "Eyes closed/Fainting"
    elif ear < EAR_THRESHOLD_DROWSY:
        return "drowsy", "Drowsy"
    else:
        return "open", "Eyes open"

def analyze_emotional_state(emotions, confidence):
    """Analiza el estado emocional de Antonio"""
    if not emotions or len(emotions) == 0:
        return "unknown", "No emotional data"
    
    if confidence < EMOTION_CONFIDENCE_LOW:
        return "unknown", "Very low confidence"
    
    # Obtener la emoción dominante
    dominant_emotion = max(emotions, key=emotions.get)
    dominant_value = emotions[dominant_emotion]
    
    # Lógica específica para Antonio
    if dominant_emotion in ["happy", "neutral"] and dominant_value > 0.4:
        return "OPTIMO", "Antonio is happy/cheerful"
    elif dominant_emotion in ["sad", "angry", "disgust"] and dominant_value > 0.5:
        return "CRITICO", "Antonio is tired/sad"
    elif dominant_emotion in ["surprise", "fear"] and dominant_value > 0.5:
        return "ESTRESADO", "Antonio is stressed/anxious"
    else:
        return "OPTIMO", "Antonio is normal"

def load_known_faces():
    """Carga automáticamente todas las caras conocidas desde la carpeta known_faces"""
    known_encodings = []
    known_names = []
    
    # Buscar todas las fotos en known_faces
    known_faces_dir = "known_faces"
    if not os.path.exists(known_faces_dir):
        print(f"❌ Folder {known_faces_dir} not found")
        return [], []
    
    # Obtener todos los archivos de imagen
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(known_faces_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if not image_files:
        print(f"❌ No photos found in {known_faces_dir}")
        print("💡 Add photos with names like: person1.jpg, antonio.png, etc.")
        return [], []
    
    print(f"🔍 Found {len(image_files)} photos in {known_faces_dir}")
    
    for image_file in image_files:
        try:
            # Obtener nombre sin extensión
            name = os.path.splitext(image_file)[0]
            image_path = os.path.join(known_faces_dir, image_file)
            
            # Cargar imagen
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(name)
                print(f"✅ {name} loaded from {image_file}")
            else:
                print(f"❌ No face found in {image_file}")
                
        except Exception as e:
            print(f"❌ Error loading {image_file}: {e}")
    
    if known_encodings:
        print(f"✅ Total: {len(known_encodings)} faces loaded successfully")
    else:
        print("❌ Could not load known faces")
    
    return known_encodings, known_names

def recognize_faces(frame, known_encodings, known_names):
    """Reconoce caras conocidas en el frame"""
    if not known_encodings:
        return "Unknown", 0.0, [], []
    
    # Reducir tamaño para mejor rendimiento
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Encontrar ubicaciones y encodings de caras
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    if not face_encodings:
        return "Unknown", 0.0, [], []
    
    # Comparar con todas las caras conocidas
    matches = face_recognition.compare_faces(known_encodings, face_encodings[0], tolerance=RECOGNITION_TOLERANCE)
    face_distances = face_recognition.face_distance(known_encodings, face_encodings[0])
    
    # Encontrar la mejor coincidencia
    best_match_index = np.argmin(face_distances)
    
    if matches[best_match_index]:
        confidence = 1.0 - face_distances[best_match_index]
        return known_names[best_match_index], confidence, face_locations, face_encodings
    else:
        return "Unknown", 0.0, face_locations, face_encodings

def calculate_facial_indicators(ear, eye_state, emotional_state, emo_conf, consecutive_closed_frames):
    """Calcula indicadores faciales avanzados"""
    # Apertura ocular (0-100)
    if ear is not None:
        eye_opening = min(100, max(0, (ear / 0.4) * 100))  # Normalizar EAR a porcentaje
    else:
        eye_opening = 0
    
    # Expresiones de tensión basadas en emociones negativas
    tension_expressions = 0
    if emotional_state == "CRITICO":
        tension_expressions = 80 + (emo_conf * 20)
    elif emotional_state == "ESTRESADO":
        tension_expressions = 50 + (emo_conf * 30)
    else:
        tension_expressions = 20 + (emo_conf * 10)
    
    # Palidez (basada en estado de ojos cerrados)
    pallor = 0
    if consecutive_closed_frames > 10:
        pallor = 90
    elif consecutive_closed_frames > 5:
        pallor = 60
    elif eye_state == "drowsy":
        pallor = 30
    else:
        pallor = 10
    
    # Focalización (basada en estabilidad del estado)
    focus = 100 - (consecutive_closed_frames * 5)
    focus = max(0, min(100, focus))
    
    # Concentración (basada en estabilidad emocional)
    concentration = 100 - (tension_expressions * 0.5)
    concentration = max(0, min(100, concentration))
    
    return {
        "eyeOpening": int(eye_opening),
        "tensionExpressions": int(tension_expressions),
        "pallor": int(pallor),
        "focus": int(focus),
        "concentration": int(concentration)
    }

def send_astronaut_monitoring_data(astronaut_data):
    """Envía datos completos de monitoreo de astronautas cada 10 segundos"""
    try:
        # Estructura de datos completa para el backend NestJS
        data = {
            # Identificación básica
            "astronautId": astronaut_data.get("astronautId", "unknown"),
            "astronautName": astronaut_data.get("astronautName", "Desconocido"),
            "codename": astronaut_data.get("codename", "UNK"),
            "timestamp": datetime.now().isoformat() + "Z",
            
            # Reconocimiento facial
            "faceDetected": astronaut_data.get("faceDetected", False),
            "faceConfidence": astronaut_data.get("faceConfidence", 0),
            "recognizedName": astronaut_data.get("recognizedName", ""),
            
            # Indicadores faciales (valores 0-100)
            "eyeOpening": astronaut_data.get("eyeOpening", 0),
            "tensionExpressions": astronaut_data.get("tensionExpressions", 0),
            "pallor": astronaut_data.get("pallor", 0),
            "focus": astronaut_data.get("focus", 0),
            "concentration": astronaut_data.get("concentration", 0),
            "ear": astronaut_data.get("ear", 0),
            "eyeState": astronaut_data.get("eyeState", "unknown"),
            
            # Datos emocionales
            "dominantEmotion": astronaut_data.get("dominantEmotion", "neutral"),
            "emotionConfidence": astronaut_data.get("emotionConfidence", 0),
            "emotionalState": astronaut_data.get("emotionalState", "OPTIMO"),
            "emotionBreakdown": astronaut_data.get("emotionBreakdown", {}),
            
            # Estado general
            "overallState": astronaut_data.get("overallState", "OPTIMO"),
            "stateDescription": astronaut_data.get("stateDescription", ""),
            "alertLevel": astronaut_data.get("alertLevel", "NORMAL"),
            "consecutiveClosedFrames": astronaut_data.get("consecutiveClosedFrames", 0),
            "stabilityFrames": astronaut_data.get("stabilityFrames", 0),
            
            # Análisis por minuto
            "sentimentCounts": astronaut_data.get("sentimentCounts", {}),
            "dominantSentiment": astronaut_data.get("dominantSentiment", "OPTIMO"),
            "sentimentPercentage": astronaut_data.get("sentimentPercentage", 0),
            "totalFrames": astronaut_data.get("totalFrames", 0),
            
            # Alertas
            "activeAlerts": astronaut_data.get("activeAlerts", []),
            "alertHistory": astronaut_data.get("alertHistory", []),
            "recommendedActions": astronaut_data.get("recommendedActions", [])
        }
        
        # Enviar datos a la API
        print(f"🔄 ENVIANDO DATOS A LA BASE DE DATOS...")
        print(f"   - Astronauta: {data['astronautName']}")
        print(f"   - ID Original: {data['astronautId']}")
        print(f"   - Sentimiento: {data['dominantSentiment']}")
        print(f"   - Estado: {data['overallState']}")
        print(f"   - URL: {API_URL}")
        print(f"   - El backend mapeará '{data['astronautId']}' al ID completo")
        
        response = requests.post(API_URL, headers=API_HEADERS, json=data, timeout=5)
        
        if response.status_code in [200, 201]:
            print(f"✅ ¡GUARDADO EXITOSO EN LA BASE DE DATOS!")
            print(f"   - Astronauta: {data['astronautName']}")
            print(f"   - ID Enviado: {data['astronautId']}")
            print(f"   - ID Mapeado por Backend: (verificar en logs del servidor)")
            print(f"   - Sentimiento Dominante: {data['dominantSentiment']}")
            print(f"   - Porcentaje: {data['sentimentPercentage']:.1f}%")
            print(f"   - Conteos: {data['sentimentCounts']}")
            print(f"   - Estado General: {data['overallState']}")
            print(f"   - Nivel de Alerta: {data['alertLevel']}")
            print(f"   - Timestamp: {data['timestamp']}")
            print(f"   - Status Code: {response.status_code}")
            return True
        else:
            print(f"❌ ¡ERROR AL GUARDAR EN LA BASE DE DATOS!")
            print(f"   - Status Code: {response.status_code}")
            print(f"   - Error: {response.text}")
            print(f"   - URL: {API_URL}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ ¡NO SE PUDO CONECTAR A LA BASE DE DATOS!")
        print(f"   - URL: { }")
        print(f"   - Verifica que el servidor esté ejecutándose")
        print(f"   - Verifica que el puerto 5000 esté disponible")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ ¡TIMEOUT AL GUARDAR EN LA BASE DE DATOS!")
        print(f"   - El servidor tardó más de 5 segundos en responder")
        print(f"   - URL: {API_URL}")
        return False
    except Exception as e:
        print(f"❌ ¡ERROR INESPERADO AL GUARDAR!")
        print(f"   - Error: {e}")
        print(f"   - URL: {API_URL}")
        return False

def main():
    parser = argparse.ArgumentParser("Antonio Recognition with Emotions")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--no_mirror", action="store_true")
    args = parser.parse_args()

    # Abrir cámara con manejo de errores
    try:
        cap, backend, real_w, real_h, real_fps = open_camera(args.cam, args.width, args.height, args.fps)
        print(f"📹 Camera {args.cam} via {backend} -> {real_w}x{real_h}@{real_fps:.0f}")
    except RuntimeError as e:
        print(f"❌ {e}")
        print("💡 Possible solutions:")
        print("   - Check that the camera is connected")
        print("   - Try with --cam 1, --cam 2, etc.")
        print("   - Close other applications using the camera")
        print("   - Restart the program")
        return
    except Exception as e:
        print(f"❌ Unexpected error opening camera: {e}")
        return

    # Cargar caras conocidas
    print("🔄 Loading known faces...")
    known_encodings, known_names = load_known_faces()
    
    if not known_encodings:
        print("❌ Could not load known faces. Exiting...")
        return
    
    print(f"✅ {len(known_names)} faces loaded: {', '.join(known_names)}")

    # Variables de estado
    frame_idx = 0
    emo_ema = None
    dominant_emo = "neutral"
    emo_conf = 0.5
    prev_t = time.time()
    fps_meas = 0.0
    
    # Variables para reconocimiento
    current_name = "Unknown"
    current_confidence = 0.0
    last_recognized_name = "Unknown"
    person_detected = False
    
    # Variables para máscara facial
    show_face_mask = False
    face_mask_overlay = None
    
    # Variables para monitoreo
    consecutive_closed_frames = 0
    last_eye_state = "open"
    last_emotional_state = "neutral"
    
    # Variables para análisis avanzado
    emotion_history = []
    ear_history = []
    stable_state_frames = 0
    last_stable_state = "neutral"
    
    # Variables para análisis de sentimientos cada 10 segundos
    sentiment_tracker = {
        "OPTIMO": 0,
        "ESTRESADO": 0, 
        "CRITICO": 0
    }
    minute_start_time = time.time()
    current_minute_sentiment = "N/A"
    sentiment_analysis_active = True
    api_sending_active = True
    
    print("🎮 Controls:")
    print("  'q' - Exit")
    print("  'm' - Toggle face mask")
    print("  's' - Toggle sentiment analysis")
    print("  'a' - Toggle API sending")
    print("  'l' - Toggle landmarks")
    print("  'r' - Reset sentiment analysis")
    print("")
    print("📊 LOGS ACTIVADOS:")
    print("  - Cada 20 segundos: Análisis completo de sentimientos")
    print("  - Cada 1 segundo: Estado actual del frame")
    print("  - Al guardar: ¡MENSAJES CLAROS DE ÉXITO/ERROR!")
    print("  - Presiona 's' para activar/desactivar análisis")
    print("  - Presiona 'a' para activar/desactivar envío a API")
    print("")
    print("🔍 MENSAJES QUE VERÁS:")
    print("  ✅ ¡GUARDADO EXITOSO EN LA BASE DE DATOS!")
    print("  ❌ ¡ERROR AL GUARDAR EN LA BASE DE DATOS!")
    print("  🔄 ENVIANDO DATOS A LA BASE DE DATOS...")
    print("  💥 ¡FALLO AL GUARDAR EN LA BD!")
    print("")
    print("🗺️ MAPEO DE IDs (SOLUCIONADO):")
    print("  Python Envía    →  Backend Mapea")
    print("  'antonio'       →  'montejo_soto_arturo_antonio'")
    print("  'bautista'      →  'bautista_machuca_luis_carlos'")
    print("  'castro'        →  'castro_garcia_jose_heiner'")
    print("  'gamonal'       →  'gamonal_chauca_jose_roger'")
    print("  'lopez'         →  'lopez_campoverde_miguel_angel'")
    print("  'miranda'       →  'miranda_saldana_rodolfo_junior'")
    print("")

    # Variables para manejo de errores de cámara
    camera_error_count = 0
    max_camera_errors = 10
    last_frame_time = time.time()
    
    while True:
        try:
            ok, frame = cap.read()
            if not ok:
                camera_error_count += 1
                print(f"⚠️ Camera error #{camera_error_count}: Could not read frame")
                
                if camera_error_count >= max_camera_errors:
                    print("❌ Too many camera errors. Closing...")
                    break
                
                # Crear frame de error
                frame = np.zeros((real_h, real_w, 3), dtype=np.uint8)
                cv2.putText(frame, "CAMERA DISCONNECTED", (50, real_h//2 - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                cv2.putText(frame, f"Errors: {camera_error_count}/{max_camera_errors}", (50, real_h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Press 'q' to exit", (50, real_h//2 + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            else:
                # Resetear contador de errores si el frame es válido
                camera_error_count = 0
                last_frame_time = time.time()
                
        except Exception as e:
            camera_error_count += 1
            print(f"❌ Critical camera error: {e}")
            if camera_error_count >= max_camera_errors:
                print("❌ Too many critical errors. Closing...")
                break
            
            # Crear frame de error crítico
            frame = np.zeros((real_h, real_w, 3), dtype=np.uint8)
            cv2.putText(frame, "CRITICAL CAMERA ERROR", (50, real_h//2 - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(frame, f"Error: {str(e)[:50]}...", (50, real_h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to exit", (50, real_h//2 + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

        now = time.time()
        dt = max(1e-3, now - prev_t)
        prev_t = now
        fps_meas = 0.9 * fps_meas + 0.1 * (1.0 / dt)
        
        if not args.no_mirror: 
            frame = cv2.flip(frame, 1)
        
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---------- Face Mesh + EAR ----------
        ear = None
        results = face_mesh.process(rgb)
        face_found = results.multi_face_landmarks is not None

        if face_found:
            lm = results.multi_face_landmarks[0].landmark
            
            # EAR
            ear_l = calc_ear(lm, LEFT_EYE, w, h)
            ear_r = calc_ear(lm, RIGHT_EYE, w, h)
            ear = (ear_l + ear_r) / 2.0
            
            # Actualizar historial EAR
            ear_history.append(ear)
            if len(ear_history) > 20:
                ear_history.pop(0)

        # ---------- Reconocimiento de Caras ----------
        if frame_idx % RECOGNITION_EVERY_N == 0:
            try:
                name, confidence, face_locations, face_encodings = recognize_faces(frame, known_encodings, known_names)
                
                if name != "Unknown" and confidence > 0.5:
                    current_name = name
                    current_confidence = confidence
                    last_recognized_name = name
                    person_detected = True
                else:
                    current_name = "Unknown"
                    current_confidence = 0.0
                    person_detected = False
            except Exception as e:
                print(f"⚠️ Error in facial recognition: {e}")
                current_name = "Unknown"
                current_confidence = 0.0
                person_detected = False

        if frame_idx % EMO_EVERY_N == 0:
            try:
                # Análisis emocional optimizado - SIEMPRE
                r = DeepFace.analyze(frame, actions=["emotion"], detector_backend="opencv", enforce_detection=False)
                rr = r[0] if isinstance(r, list) else r
                emo = {k.lower(): float(v) / 100.0 for k, v in rr["emotion"].items()}

                # Procesamiento
                label = max(emo, key=emo.get)
                p = emo[label]
                
                if p < 0.45:
                    label = "neutral"
                    p = emo.get("neutral", p)

                # EMA suavizado
                emo_ema = smooth_ema(emo_ema, emo, alpha=0.6)
                dominant_emo = label
                emo_conf = float(p)
                
                # Historial de emociones
                emotion_history.append(emo)
                if len(emotion_history) > 15:
                    emotion_history.pop(0)
                
            except Exception as e:
                print(f"⚠️ Error in emotional analysis: {e}")
                # Mantener valores por defecto en caso de error
                if emo_ema is None:
                    emo_ema = {"neutral": 1.0}
                dominant_emo = "neutral"
                emo_conf = 0.5

        frame_idx += 1

        # ---------- Análisis de Sentimientos cada 10 segundos ----------
        current_time = time.time()
        elapsed_seconds = (current_time - minute_start_time)
        
        # Cada 20 segundos, calcular promedio y resetear
        if elapsed_seconds >= 20.0 and sentiment_analysis_active:
            # Calcular el sentimiento dominante del minuto
            total_states = sum(sentiment_tracker.values())
            if total_states > 0:
                # Encontrar el estado con más ocurrencias
                dominant_sentiment = max(sentiment_tracker, key=sentiment_tracker.get)
                dominant_count = sentiment_tracker[dominant_sentiment]
                percentage = (dominant_count / total_states) * 100
                
                # Añadir nombre de la persona si está detectada
                if person_detected and last_recognized_name != "Unknown":
                    current_minute_sentiment = f"{last_recognized_name} - {dominant_sentiment} ({dominant_count}/{total_states} - {percentage:.1f}%)"
                    print(f"📊 Sentiment of {last_recognized_name} (10s): {dominant_sentiment} ({dominant_count}/{total_states} - {percentage:.1f}%)")
                    print(f"🔍 DETAILED SENTIMENT BREAKDOWN:")
                    print(f"   - OPTIMO: {sentiment_tracker['OPTIMO']} frames")
                    print(f"   - ESTRESADO: {sentiment_tracker['ESTRESADO']} frames") 
                    print(f"   - CRITICO: {sentiment_tracker['CRITICO']} frames")
                    print(f"   - WINNER: {dominant_sentiment} with {percentage:.1f}%")
                    
                    # Enviar datos completos de monitoreo a la API si está activado
                    if api_sending_active:
                        # Calcular indicadores faciales
                        facial_indicators = calculate_facial_indicators(ear, eye_state, emotional_state, emo_conf, consecutive_closed_frames)
                        
                        # Determinar alertas activas
                        active_alerts = []
                        recommended_actions = []
                        
                        if estado_general == "CRITICO":
                            active_alerts.append("ESTADO CRÍTICO DETECTADO")
                            recommended_actions.append("NOTIFICAR_MÉDICO")
                            recommended_actions.append("REVISAR_VITALES")
                        elif estado_general == "ESTRESADO":
                            active_alerts.append("ESTRÉS ALTO DETECTADO")
                            recommended_actions.append("RECOMENDAR_DESCANSO")
                        elif consecutive_closed_frames > 10:
                            active_alerts.append("POSIBLE DESMAYO")
                            recommended_actions.append("VERIFICAR_CONSCIENCIA")
                        
                        # Preparar datos completos de monitoreo
                        astronaut_data = {
                            "astronautId": last_recognized_name.lower().replace(" ", "_"),
                            "astronautName": last_recognized_name,
                            "codename": last_recognized_name[:2].upper(),
                            "faceDetected": person_detected,
                            "faceConfidence": current_confidence,
                            "recognizedName": last_recognized_name,
                            "ear": ear if ear else 0,
                            "eyeState": eye_state,
                            "dominantEmotion": dominant_emo,
                            "emotionConfidence": emo_conf,
                            "emotionalState": emotional_state,
                            "emotionBreakdown": emo_ema or {},
                            "overallState": estado_general,
                            "stateDescription": estado_desc,
                            "alertLevel": "CRITICAL" if estado_general == "CRITICO" else "WARNING" if estado_general == "ESTRESADO" else "NORMAL",
                            "consecutiveClosedFrames": consecutive_closed_frames,
                            "stabilityFrames": stable_state_frames,
                            "sentimentCounts": sentiment_tracker,
                            "dominantSentiment": dominant_sentiment,
                            "sentimentPercentage": percentage,
                            "totalFrames": total_states,
                            "activeAlerts": active_alerts,
                            "alertHistory": [],
                            "recommendedActions": recommended_actions,
                            # Indicadores faciales
                            **facial_indicators
                        }
                        
                        # Intentar guardar en la base de datos
                        success = send_astronaut_monitoring_data(astronaut_data)
                        if success:
                            print(f"🎉 ¡DATOS GUARDADOS CORRECTAMENTE EN LA BD!")
                            print(f"   📊 Mapeo: '{last_recognized_name}' → ID completo en BD")
                            print(f"   🔍 Verifica en Postman con el ID mapeado")
                        else:
                            print(f"💥 ¡FALLO AL GUARDAR EN LA BD!")
                            print(f"   ⚠️ Revisa que el servidor NestJS esté ejecutándose")
                    else:
                        print(f"📤 API sending disabled - Data: {last_recognized_name} - {dominant_sentiment}")
                        print(f"⚠️ Los datos NO se están guardando en la base de datos")
                else:
                    current_minute_sentiment = f"{dominant_sentiment} ({dominant_count}/{total_states} - {percentage:.1f}%)"
                    print(f"📊 10-second sentiment: {current_minute_sentiment}")
            else:
                current_minute_sentiment = "No data"
                print("📊 No recognized persons to analyze")
            
            # Resetear para los siguientes 10 segundos
            sentiment_tracker = {"OPTIMO": 0, "ESTRESADO": 0, "CRITICO": 0}
            minute_start_time = current_time

        # ---------- Evaluación de Estados ----------
        # Detectar estado de ojos
        eye_state, eye_desc = detect_eye_state(ear)
        
        # Detectar estado emocional (SIEMPRE, sin importar si Antonio está detectado)
        emotional_state, emotional_desc = analyze_emotional_state(emo_ema or {}, emo_conf)
        
        # Análisis de estabilidad del estado
        current_state = f"{eye_state}_{emotional_state}"
        if current_state == last_stable_state:
            stable_state_frames += 1
        else:
            stable_state_frames = 0
            last_stable_state = current_state
        
        # Conteo de frames con ojos cerrados
        if eye_state == "closed":
            consecutive_closed_frames += 1
        else:
            consecutive_closed_frames = 0
        
        # ---------- Máscara Facial ----------
        if show_face_mask and face_found and results.multi_face_landmarks:
            # Crear máscara facial usando MediaPipe
            face_landmarks = results.multi_face_landmarks[0]
            
            # Obtener puntos de contorno facial
            face_oval = mp_face_mesh.FACEMESH_FACE_OVAL
            face_points = []
            
            for connection in face_oval:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_point = face_landmarks.landmark[start_idx]
                end_point = face_landmarks.landmark[end_idx]
                
                start_x = int(start_point.x * w)
                start_y = int(start_point.y * h)
                end_x = int(end_point.x * w)
                end_y = int(end_point.y * h)
                
                face_points.extend([(start_x, start_y), (end_x, end_y)])
            
            # Crear máscara
            mask = np.zeros((h, w), dtype=np.uint8)
            if face_points:
                cv2.fillPoly(mask, [np.array(face_points)], 255)
                
                # Aplicar máscara con transparencia
                overlay = frame.copy()
                overlay[mask > 0] = [0, 255, 0]  # Verde para la máscara
                cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

        # Lógica de estados - SIEMPRE muestra emociones, con o sin persona detectada
        if not person_detected:
            # Sin Antonio detectado, pero muestra emociones de quien esté en cámara
            if eye_state == "closed" and consecutive_closed_frames > 10:
                estado_general = "CRITICO"
                estado_desc = "FAINTING - Eyes closed"
                color = (0, 0, 255)  # Rojo
            elif eye_state == "drowsy" and consecutive_closed_frames > 5:
                estado_general = "CRITICO"
                estado_desc = "TIRED - Drowsy"
                color = (0, 0, 255)  # Rojo
            elif emotional_state == "CRITICO":
                estado_general = "CRITICO"
                estado_desc = "SAD/ANGRY"
                color = (0, 0, 255)  # Rojo
            elif emotional_state == "ESTRESADO":
                estado_general = "ESTRESADO"
                estado_desc = "STRESSED/ANXIOUS"
                color = (0, 255, 255)  # Amarillo
            elif emotional_state == "OPTIMO":
                estado_general = "OPTIMO"
                estado_desc = "HAPPY/CHEERFUL"
                color = (0, 255, 0)  # Verde
            else:
                estado_general = "OPTIMO"
                estado_desc = "Normal state"
                color = (0, 255, 0)  # Verde
        else:
            # Persona conocida detectada - mismo análisis pero con su nombre
            if eye_state == "closed" and consecutive_closed_frames > 10:
                estado_general = "CRITICO"
                estado_desc = f"{last_recognized_name} - FAINTING"
                color = (0, 0, 255)  # Rojo
            elif eye_state == "drowsy" and consecutive_closed_frames > 5:
                estado_general = "CRITICO"
                estado_desc = f"{last_recognized_name} - TIRED"
                color = (0, 0, 255)  # Rojo
            elif emotional_state == "CRITICO":
                estado_general = "CRITICO"
                estado_desc = f"{last_recognized_name} - SAD/ANGRY"
                color = (0, 0, 255)  # Rojo
            elif emotional_state == "ESTRESADO":
                estado_general = "ESTRESADO"
                estado_desc = f"{last_recognized_name} - STRESSED"
                color = (0, 255, 255)  # Amarillo
            elif emotional_state == "OPTIMO":
                estado_general = "OPTIMO"
                estado_desc = f"{last_recognized_name} - HAPPY"
                color = (0, 255, 0)  # Verde
            else:
                estado_general = "OPTIMO"
                estado_desc = f"{last_recognized_name} - NORMAL"
                color = (0, 255, 0)  # Verde
        
        # ---------- Registrar Estado en Tracker de Sentimientos ----------
        # SOLO registrar si la persona está reconocida (no desconocidos)
        if sentiment_analysis_active and person_detected and last_recognized_name != "Unknown":
            try:
                sentiment_tracker[estado_general] += 1
                # Log cada frame para debugging
                if frame_idx % 30 == 0:  # Cada 30 frames (1 segundo aprox)
                    print(f"🎯 FRAME {frame_idx}: {last_recognized_name} → {estado_general} (Total: {sentiment_tracker})")
            except KeyError:
                # Si el estado no está en el tracker, añadirlo
                sentiment_tracker[estado_general] = 1
                print(f"🆕 NEW STATE ADDED: {estado_general}")
            except Exception as e:
                print(f"⚠️ Error in sentiment tracker: {e}")

        # ---------- Interfaz Mejorada ----------
        # Fondo semi-transparente para mejor legibilidad
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 300), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Información principal
        cv2.putText(frame, f"Antonio Recognition - {real_w}x{real_h}@{real_fps:.0f} FPS~{fps_meas:.1f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Nombre y confianza
        if person_detected:
            name_color = (0, 255, 0) if current_confidence > 0.7 else (0, 255, 255)
            cv2.putText(frame, f"👤 {last_recognized_name}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, name_color, 3)
            cv2.putText(frame, f"Confidence: {current_confidence:.2f}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, name_color, 2)
        else:
            cv2.putText(frame, "👤 Unknown", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (128, 128, 128), 3)
        
        # ESTADO PRINCIPAL - GRANDE Y CLARO
        # Traducir estados a inglés para mostrar
        estado_english = {
            "OPTIMO": "OPTIMAL",
            "ESTRESADO": "STRESSED", 
            "CRITICO": "CRITICAL"
        }.get(estado_general, estado_general)
        
        cv2.putText(frame, f"STATE: {estado_english}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 4)
        cv2.putText(frame, f"{estado_desc}", (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Información técnica
        cv2.putText(frame, f"Emotion: {dominant_emo} ({emo_conf:.2f})", (20, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.putText(frame, f"Eyes: {eye_desc}", (20, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        if ear is not None:
            cv2.putText(frame, f"EAR: {ear:.3f}", (20, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Información de reconocimiento
        if person_detected:
            cv2.putText(frame, f"Recognized from: known_faces/{last_recognized_name}.jpg", (20, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(frame, f"Confidence: {current_confidence:.2f}", (20, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.putText(frame, f"Comparing with: {', '.join(known_names)}", (20, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            cv2.putText(frame, f"Does not match any known face", (20, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        # Panel de información lateral
        right_x = w - 320
        cv2.rectangle(frame, (right_x-10, 10), (w-10, 200), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.8, frame, 0.2, 0, frame)
        
        cv2.putText(frame, "=== STATES ===", (right_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "OPTIMAL: Green", (right_x, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "STRESSED: Yellow", (right_x, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, "CRITICAL: Red", (right_x, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "NO_DETECTED: Gray", (right_x, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        cv2.putText(frame, "=== CONTROLS ===", (right_x, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "m - Face mask", (right_x, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "s - Sentiment analysis", (right_x, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "a - API sending", (right_x, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "r - Reset analysis", (right_x, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "q - Exit", (right_x, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Estado de la máscara
        mask_status = "ON" if show_face_mask else "OFF"
        mask_color = (0, 255, 0) if show_face_mask else (128, 128, 128)
        cv2.putText(frame, f"Mask: {mask_status}", (right_x, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mask_color, 1)
        
        # Estado del análisis de sentimientos
        sentiment_status = "ON" if sentiment_analysis_active else "OFF"
        sentiment_status_color = (0, 255, 0) if sentiment_analysis_active else (128, 128, 128)
        cv2.putText(frame, f"Analysis: {sentiment_status}", (right_x, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sentiment_status_color, 1)
        
        # Estado del envío a API
        api_status = "ON" if api_sending_active else "OFF"
        api_status_color = (0, 255, 0) if api_sending_active else (128, 128, 128)
        cv2.putText(frame, f"API: {api_status}", (right_x, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.5, api_status_color, 1)
        
        # ---------- Análisis de Sentimientos cada 10 segundos ----------
        # Panel de sentimientos
        sentiment_y = 260
        cv2.rectangle(frame, (right_x-10, sentiment_y-10), (w-10, sentiment_y+80), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.8, frame, 0.2, 0, frame)
        
        cv2.putText(frame, "=== SENTIMENTS (10s) ===", (right_x, sentiment_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Mostrar sentimiento con nombre si está disponible
        if person_detected and last_recognized_name != "Unknown":
            cv2.putText(frame, f"{last_recognized_name}: {current_minute_sentiment}", (right_x, sentiment_y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.putText(frame, f"Waiting for known person...", (right_x, sentiment_y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Contadores en tiempo real - SOLO para personas reconocidas
        total_current = sum(sentiment_tracker.values())
        if total_current > 0:
            # Mostrar nombre de la persona si está detectada
            if person_detected and last_recognized_name != "Unknown":
                cv2.putText(frame, f"Analyzing: {last_recognized_name}", (right_x, sentiment_y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(frame, f"OPTIMAL: {sentiment_tracker['OPTIMO']}", (right_x, sentiment_y+65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(frame, f"STRESSED: {sentiment_tracker['ESTRESADO']}", (right_x, sentiment_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.putText(frame, f"CRITICAL: {sentiment_tracker['CRITICO']}", (right_x, sentiment_y+95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            else:
                cv2.putText(frame, f"OPTIMAL: {sentiment_tracker['OPTIMO']}", (right_x, sentiment_y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(frame, f"STRESSED: {sentiment_tracker['ESTRESADO']}", (right_x, sentiment_y+65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.putText(frame, f"CRITICAL: {sentiment_tracker['CRITICO']}", (right_x, sentiment_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        else:
            if person_detected and last_recognized_name != "Unknown":
                cv2.putText(frame, f"Analyzing: {last_recognized_name}", (right_x, sentiment_y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(frame, "Collecting data...", (right_x, sentiment_y+65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            else:
                cv2.putText(frame, "Waiting for known person...", (right_x, sentiment_y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                cv2.putText(frame, "Unknown persons NOT analyzed", (right_x, sentiment_y+65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Mostrar frame con manejo de errores
        try:
            cv2.imshow("Antonio Recognition with Emotions", frame)
        except Exception as e:
            print(f"⚠️ Error showing frame: {e}")
        
        # Controles de teclado
        try:
            key = cv2.waitKey(1) & 0xFF
        except Exception as e:
            print(f"⚠️ Error in keyboard controls: {e}")
            key = 0
        if key == ord('q'): 
            break
        elif key == ord('m'):  # Toggle máscara facial
            show_face_mask = not show_face_mask
            print(f"🎭 Face mask: {'ACTIVATED' if show_face_mask else 'DEACTIVATED'}")
        elif key == ord('s'):  # Toggle análisis de sentimientos
            sentiment_analysis_active = not sentiment_analysis_active
            print(f"📊 Sentiment analysis: {'ACTIVATED' if sentiment_analysis_active else 'DEACTIVATED'}")
        elif key == ord('a'):  # Toggle envío a API
            api_sending_active = not api_sending_active
            print(f"📤 API sending: {'ACTIVATED' if api_sending_active else 'DEACTIVATED'}")
        elif key == ord('r'):  # Reset análisis de sentimientos
            sentiment_tracker = {"OPTIMO": 0, "ESTRESADO": 0, "CRITICO": 0}
            minute_start_time = time.time()
            current_minute_sentiment = "N/A"
            print("🔄 Sentiment analysis reset")
        elif key == ord('l'):  # Toggle landmarks (implementar si necesario)
            pass

    # Limpieza con manejo de errores
    try:
        cap.release()
        print("📹 Camera released")
    except Exception as e:
        print(f"⚠️ Error releasing camera: {e}")
    
    try:
        face_mesh.close()
        print("🔍 Face mesh closed")
    except Exception as e:
        print(f"⚠️ Error closing face mesh: {e}")
    
    try:
        cv2.destroyAllWindows()
        print("🖥️ Windows closed")
    except Exception as e:
        print(f"⚠️ Error closing windows: {e}")
    
    print("✅ Program finished correctly")

if __name__ == "__main__":
    main()
