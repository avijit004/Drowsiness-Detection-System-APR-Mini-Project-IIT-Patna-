import os, time, sys
import numpy as np
import cv2
import mediapipe as mp
import subprocess

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

# ==================  SINGLE-BEEP (NO OVERLAP)  ==================
def beep():
    if os.path.exists("alarm.wav"):
        subprocess.Popen(["afplay", "alarm.wav"])  # plays once, no loop
    else:
        print("\a")

# ==================  MEDIAPIPE SETUP  ==================
mp_face_mesh = mp.solutions.face_mesh
FACE_MESH = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_H = (61, 291)
MOUTH_V = (13, 14)

def _p2xy(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

def _dist(a,b):
    return float(np.linalg.norm(a-b))

def ear(lm, w, h, idx):
    p = [_p2xy(lm[i],w,h) for i in idx]
    return (_dist(p[1],p[5]) + _dist(p[2],p[4])) / (2*_dist(p[0],p[3]) + 1e-6)

def mar(lm,w,h):
    pL=_p2xy(lm[MOUTH_H[0]],w,h)
    pR=_p2xy(lm[MOUTH_H[1]],w,h)
    pU=_p2xy(lm[MOUTH_V[0]],w,h)
    pD=_p2xy(lm[MOUTH_V[1]],w,h)
    return _dist(pU,pD)/(_dist(pL,pR)+1e-6)

# ==================  THRESHOLDS  ==================
EAR_THRESH = 0.19
MAR_THRESH = 0.45
YAWN_FRAMES = 4
EYE_CLOSED_SECONDS_LIMIT = 3.0       # >3s eyes closed → drowsy
BEEP_COOLDOWN = 5.0                  # seconds (NEW)

def main():
    cap = cv2.VideoCapture(0)

    eye_closed_start = None
    yawn_counter = 0
    last_beep_time = 0   # NEW

    fps_t = time.time()
    fps = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = FACE_MESH.process(rgb)

        eye_closed = False
        yawning = False
        drowsy = False
        reasons = []

        # ---------------- FACE NOT DETECTED = DROWSY ----------------
        if not res.multi_face_landmarks:
            drowsy = True
            reasons.append("FACE / EYES NOT VISIBLE")

        else:
            lm = res.multi_face_landmarks[0].landmark
            e = (ear(lm,w,h,LEFT_EYE)+ear(lm,w,h,RIGHT_EYE))/2
            m = mar(lm,w,h)

            eye_closed = e < EAR_THRESH
            yawning = m > MAR_THRESH

            cv2.putText(frame,f"EAR:{e:.3f}",(10,h-40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            cv2.putText(frame,f"MAR:{m:.3f}",(10,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

            # Yawning detection (must persist)
            if yawning:
                yawn_counter += 1
            else:
                yawn_counter = 0
            if yawn_counter >= YAWN_FRAMES:
                drowsy = True
                reasons.append("YAWNING")

            # Eyes closed time-based detection
            if eye_closed:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                elif time.time() - eye_closed_start >= EYE_CLOSED_SECONDS_LIMIT:
                    drowsy = True
                    reasons.append(f"EYES CLOSED > {int(EYE_CLOSED_SECONDS_LIMIT)}s")
            else:
                eye_closed_start = None

        # ---------------- DROWSINESS ALERT + COOLDOWN ----------------
        if drowsy:
            if time.time() - last_beep_time >= BEEP_COOLDOWN:
                beep()
                last_beep_time = time.time()

            cv2.rectangle(frame,(0,0),(w,60),(0,0,255),-1)
            cv2.putText(frame,"DROWSY! " + " + ".join(reasons),(10,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        # FPS display
        now=time.time()
        if now-fps_t>=0.5:
            fps=1/(now-fps_t)
            fps_t=now
        cv2.putText(frame,f"FPS:{fps:.1f}",(w-120,25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        cv2.imshow("Drowsiness Detection (q to quit)",frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
