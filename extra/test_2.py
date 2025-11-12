import os, time, threading, sys
import numpy as np
import cv2
import mediapipe as mp

# ==================  OPTIONAL CNN MODEL  ==================
MODEL_PATH = "drowsiness_mobilenetv2.h5"
IMG_SIZE = 145
USE_MODEL = os.path.exists(MODEL_PATH)

if USE_MODEL:
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        MODEL_LABELS = ["yawn", "no_yawn", "Closed", "Open"]
        print("[INFO] CNN Model Loaded Successfully ✅")
    except:
        USE_MODEL = False
        print("[WARN] Model load failed. Continuing without CNN.")

# ==================  MEDIAPIPE FACE MESH  ==================
mp_face_mesh = mp.solutions.face_mesh
FACE_MESH = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==================  ALARM (MAC M1/M2 SAFE)  ==================
class AlarmPlayer:
    def __init__(self, wav_path="alarm.wav"):
        self.wav_path = wav_path
        self._running = False
        self._thread = None

    def _loop(self):
        while self._running:
            if os.path.exists(self.wav_path):
                os.system(f"afplay '{self.wav_path}'")
            else:
                print("[WARN] 'alarm.wav' not found. Alarm silent.")
                time.sleep(1)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

ALARM = AlarmPlayer("alarm.wav")

# ==================  FACIAL LANDMARK INDEXES  ==================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_H = (61, 291)
MOUTH_V = (13, 14)

# ==================  GEOMETRY HELPERS  ==================
def _p2xy(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

def _dist(a, b):
    return float(np.linalg.norm(a - b))

def ear(lm, w, h, idx):
    p = [_p2xy(lm[i], w, h) for i in idx]
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / (2 * _dist(p[0], p[3]) + 1e-6)

def mar(lm, w, h):
    pL = _p2xy(lm[MOUTH_H[0]], w, h)
    pR = _p2xy(lm[MOUTH_H[1]], w, h)
    pU = _p2xy(lm[MOUTH_V[0]], w, h)
    pD = _p2xy(lm[MOUTH_V[1]], w, h)
    return _dist(pU, pD) / (_dist(pL, pR) + 1e-6)

# ==================  THRESHOLDS & SETTINGS  ==================
EAR_THRESH = 0.19                 # Eyes considered closed below this
MAR_THRESH = 0.60                 # Mouth considered yawning above this
YAWN_FRAMES = 4                   # Yawn must persist this many frames
EYE_CLOSED_SECONDS_LIMIT = 3.0    # Drowsy if eyes closed this long (seconds)

# ==================  MAIN LOOP  ==================
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    # Timers/Counters
    eye_closed_start = None
    yawn_counter = 0

    fps_t = time.time()
    fps = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        # Run FaceMesh
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = FACE_MESH.process(rgb)
        rgb.flags.writeable = True

        # Defaults
        eye_closed = False
        yawning = False
        do_alarm = False
        reasons = []

        # ================== FACE/ EYES VISIBILITY CHECK ==================
        if not res.multi_face_landmarks:
            # No face (and hence eyes) visible -> considered drowsy
            do_alarm = True
            reasons.append("FACE / EYES NOT VISIBLE")
        else:
            lm = res.multi_face_landmarks[0].landmark

            # Compute EAR & MAR
            e = (ear(lm, w, h, LEFT_EYE) + ear(lm, w, h, RIGHT_EYE)) / 2.0
            m = mar(lm, w, h)

            eye_closed = e < EAR_THRESH
            yawning   = m > MAR_THRESH

            # HUD: show metrics
            cv2.putText(frame, f"EAR:{e:.3f} thr:{EAR_THRESH}", (10, h-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(frame, f"MAR:{m:.3f} thr:{MAR_THRESH}", (10, h-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            # ================== YAWN PERSISTENCE (frames-based) ==================
            if yawning:
                yawn_counter += 1
            else:
                yawn_counter = 0

            if yawn_counter >= YAWN_FRAMES:
                do_alarm = True
                reasons.append("YAWNING")

            # ================== CONTINUOUS EYE CLOSURE (time-based) ==================
            if eye_closed:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                else:
                    if (time.time() - eye_closed_start) > EYE_CLOSED_SECONDS_LIMIT:
                        do_alarm = True
                        reasons.append(f"EYES CLOSED > {int(EYE_CLOSED_SECONDS_LIMIT)}s")
            else:
                eye_closed_start = None  # reset when eyes open

        # ================== ALARM CONTROL + UI ==================
        if do_alarm:
            ALARM.start()
            # Banner
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 255), -1)
            msg = "DROWSY! " + " + ".join(reasons)
            cv2.putText(frame, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
        else:
            ALARM.stop()

        # FPS
        now = time.time()
        if now - fps_t >= 0.5:
            fps = 1.0 / (now - fps_t)
            fps_t = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Drowsiness (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    ALARM.stop()

if __name__ == "__main__":
    if not os.path.exists("alarm.wav"):
        print("[WARN] 'alarm.wav' file not found. Alarm will be silent.")
    main()
