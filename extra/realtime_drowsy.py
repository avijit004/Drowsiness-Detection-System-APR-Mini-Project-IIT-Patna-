import os, time, math, threading, sys
import numpy as np
import cv2

# ---- Optional TF model (blended) -------------------------------------------
MODEL_PATH = "drowsiness_mobilenetv2.h5"
IMG_SIZE   = 145   # change to 224 if you trained at 224
USE_MODEL  = os.path.exists(MODEL_PATH)

if USE_MODEL:
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[INFO] Loaded model: {MODEL_PATH}")
        MODEL_LABELS = ["yawn", "no_yawn", "Closed", "Open"]  # adjust if your order differs
    except Exception as e:
        print(f"[WARN] Could not load model ({e}). Continuing without it.")
        USE_MODEL = False

# ---- MediaPipe Face Mesh ----------------------------------------------------
import mediapipe as mp
mp_drawing   = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Iris-refined landmarks help eye geometry
FACE_MESH = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---- Alarm player (non-blocking) -------------------------------------------
# ---- Alarm player (macOS-safe) -------------------------------------------
import subprocess
import threading
import time

class AlarmPlayer:
    def __init__(self, wav_path="alarm.wav"):
        self.wav_path = wav_path
        self._running = False
        self._thread = None

    def _loop(self):
        # Keep playing while running = True
        while self._running:
            if os.path.exists(self.wav_path):
                subprocess.call(["afplay", self.wav_path])
            else:
                print("[WARN] alarm.wav missing!")
                time.sleep(0.8)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False


ALARM = AlarmPlayer("alarm.wav")


# ---- Geometry helpers (FaceMesh indices) -----------------------------------
# Eye landmarks (6 points) for EAR
# Left eye:   [33, 160, 158, 133, 153, 144]
# Right eye:  [362, 385, 387, 263, 373, 380]
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth corners / lips for MAR
# Horizontal: corners 61, 291   Vertical: 13 (upper), 14 (lower)
MOUTH_H  = (61, 291)
MOUTH_V  = (13, 14)

def _p2xy(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h], dtype=np.float32)

def _euclid(a, b):
    return float(np.linalg.norm(a - b))

def ear_from_landmarks(landmarks, w, h, eye_idx):
    p = [_p2xy(landmarks[i], w, h) for i in eye_idx]
    # Format: [p1, p2, p3, p4, p5, p6] as per standard EAR
    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    return ( _euclid(p[1], p[5]) + _euclid(p[2], p[4]) ) / (2.0 * _euclid(p[0], p[3]) + 1e-6)

def mar_from_landmarks(landmarks, w, h):
    pL = _p2xy(landmarks[MOUTH_H[0]], w, h)
    pR = _p2xy(landmarks[MOUTH_H[1]], w, h)
    pU = _p2xy(landmarks[MOUTH_V[0]], w, h)
    pD = _p2xy(landmarks[MOUTH_V[1]], w, h)
    horiz = _euclid(pL, pR) + 1e-6
    vert  = _euclid(pU, pD)
    return vert / horiz

# ---- Thresholds / logic -----------------------------------------------------
EAR_THRESH = 0.18      # lower => closed
EAR_FRAMES = 10      # consecutive frames for eye-closure alarm

MAR_THRESH = 0.45      # higher => yawning
YAWN_FRAMES = 4      # consecutive frames for yawn alarm

PREDICT_EVERY = 5      # run CNN every N frames (if model is available)

# ---- Video loop -------------------------------------------------------------
def main():
    # macOS backend: AVFoundation is usually most stable on M-series
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERR] Cannot open camera.")
        sys.exit(1)

    eye_counter = 0
    yawn_counter = 0
    frame_i = 0
    fps_t = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = FACE_MESH.process(rgb)

        ear = None
        mar = None
        eye_closed = False
        yawning = False
        cnn_eye_closed = None
        cnn_yawn = None

        if res.multi_face_landmarks:
            face = res.multi_face_landmarks[0].landmark

            ear_left  = ear_from_landmarks(face, w, h, LEFT_EYE)
            ear_right = ear_from_landmarks(face, w, h, RIGHT_EYE)
            ear = (ear_left + ear_right) / 2.0

            mar = mar_from_landmarks(face, w, h)

            eye_closed = ear < EAR_THRESH
            yawning    = mar > MAR_THRESH

            # Optional: use your CNN every N frames, blend decisions
            if USE_MODEL and (frame_i % PREDICT_EVERY == 0):
                # Crop a face ROI from landmarks (tight bbox)
                xs = [lm.x for lm in face]
                ys = [lm.y for lm in face]
                x1 = max(0, int(min(xs) * w))
                y1 = max(0, int(min(ys) * h))
                x2 = min(w-1, int(max(xs) * w))
                y2 = min(h-1, int(max(ys) * h))
                # Slightly pad
                pad = int(0.05 * max(x2-x1, y2-y1))
                x1 = max(0, x1-pad); y1=max(0, y1-pad)
                x2 = min(w-1, x2+pad); y2=min(h-1, y2+pad)

                roi = frame[y1:y2, x1:x2]
                if roi.size != 0:
                    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                    arr = roi.astype("float32") / 255.0
                    arr = np.expand_dims(arr, 0)
                    prob = model.predict(arr, verbose=0)[0]  # 4-class
                    # Map to interpretable
                    out = dict(zip(MODEL_LABELS, prob))
                    # Soft decisions
                    cnn_eye_closed = out.get("Closed", 0.0) > 0.60 and out.get("Open", 0.0) < 0.4
                    cnn_yawn       = out.get("yawn", 0.0)   > 0.60

                    # Blend with heuristics (OR is simple/robust)
                    eye_closed = eye_closed or cnn_eye_closed
                    yawning    = yawning or cnn_yawn

                    # Draw model probs
                    yy = 30
                    for k, v in out.items():
                        cv2.putText(frame, f"{k}: {v:.2f}", (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                        yy += 18

        # Update counters
        if eye_closed:
            eye_counter += 1
        else:
            eye_counter = 0

        if yawning:
            yawn_counter += 1
        else:
            yawn_counter = 0

        # Alarm conditions
        alarm = False
        reason = []
        if eye_counter >= EAR_FRAMES:
            alarm = True
            reason.append("EYES CLOSED")
        if yawn_counter >= YAWN_FRAMES:
            alarm = True
            reason.append("YAWN")

        if alarm:
            ALARM.start()

            cv2.rectangle(frame, (0,0), (w, 60), (0,0,255), -1)
            txt = "DROWSY! " + " + ".join(reason)
            cv2.putText(frame, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        else:
            ALARM.stop()
        # HUD
        fps_now = time.time()
        if fps_now - fps_t >= 0.5:
            fps = 1.0 / max(1e-6, (fps_now - fps_t))
            fps_t = fps_now
        cv2.putText(frame, f"FPS: {fps:.1f}", (w-120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if ear is not None:
            cv2.putText(frame, f"EAR:{ear:.3f}  thr:{EAR_THRESH}", (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        if mar is not None:
            cv2.putText(frame, f"MAR:{mar:.3f}  thr:{MAR_THRESH}", (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("Drowsiness (q to quit)", frame)
        frame_i += 1

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
