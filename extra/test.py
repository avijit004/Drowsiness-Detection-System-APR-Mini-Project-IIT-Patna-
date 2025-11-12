import os
import time
import subprocess
import numpy as np
import cv2
import mediapipe as mp

# ================== SETTINGS ==================
EAR_THRESH = 0.19
MAR_THRESH = 0.25

EYE_FRAMES_LIMIT = 7    # frames to count one blink event
YAWN_FRAMES_LIMIT = 4   # frames to count one yawn event

BLINK_EVENT_THRESHOLD = 7   # after 7 blink events → beep
YAWN_EVENT_THRESHOLD = 4    # after 4 yawn events → beep

EYE_CLOSED_SECONDS_LIMIT = 6   # NEW: continuous eyes closed alarm
NO_FACE_SECONDS_LIMIT = 2      # NEW: face missing alarm

ALARM_WAV = "alarm.wav"

# ================== MEDIAPIPE ==================
mp_face_mesh = mp.solutions.face_mesh
FACE_MESH = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ================== LANDMARK INDEXES ==================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_H = (61, 291)
MOUTH_V = (13, 14)

# ================== HELPERS ==================
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

def play_beep():
    if os.path.exists(ALARM_WAV):
        subprocess.Popen(["afplay", ALARM_WAV])  # plays once
    else:
        print("\a")  # fallback beep

# ================== MAIN ==================
def main():
    cap = cv2.VideoCapture(0)

    # Blink/Yawn event counters
    eye_frames = 0
    yawn_frames = 0
    in_eye_event = False
    in_yawn_event = False
    blink_total = 0
    yawn_total = 0

    # NEW continuous timers
    eye_closed_start = None
    no_face_start = None

    while True:
        ok, frame = cap.read()
        if not ok: break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = FACE_MESH.process(rgb)

        eye_closed = False
        yawning = False

        # ================== FACE DETECTED ==================
        if res.multi_face_landmarks:
            no_face_start = None  # reset missing-face timer
            lm = res.multi_face_landmarks[0].landmark

            e = (ear(lm, w, h, LEFT_EYE) + ear(lm, w, h, RIGHT_EYE)) / 2
            m = mar(lm, w, h)

            eye_closed = e < EAR_THRESH
            yawning = m > MAR_THRESH

        # ================== NEW: FACE NOT DETECTED ALARM ==================
        else:
            if no_face_start is None:
                no_face_start = time.time()
            else:
                if time.time() - no_face_start > NO_FACE_SECONDS_LIMIT:
                    play_beep()
                    no_face_start = None

            cv2.imshow("Drowsiness Counter (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # ================== BLINK EVENT DETECTION ==================
        if eye_closed:
            eye_frames += 1
            if eye_frames >= EYE_FRAMES_LIMIT and not in_eye_event:
                blink_total += 1
                in_eye_event = True
                eye_frames = 0
        else:
            eye_frames = 0
            in_eye_event = False

        # ================== YAWN EVENT DETECTION ==================
        if yawning:
            yawn_frames += 1
            if yawn_frames >= YAWN_FRAMES_LIMIT and not in_yawn_event:
                yawn_total += 1
                in_yawn_event = True
                yawn_frames = 0
        else:
            yawn_frames = 0
            in_yawn_event = False

        # ================== NEW: ALARM FOR CONTINUOUS EYE CLOSURE (>6 sec) ==================
        if eye_closed:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            else:
                if time.time() - eye_closed_start > EYE_CLOSED_SECONDS_LIMIT:
                    play_beep()
                    eye_closed_start = None
        else:
            eye_closed_start = None

        # ================== EVENT THRESHOLD ALARMS ==================
        if blink_total >= BLINK_EVENT_THRESHOLD:
            play_beep()
            blink_total = 0

        if yawn_total >= YAWN_EVENT_THRESHOLD:
            play_beep()
            yawn_total = 0

        # ================== DISPLAY UI ==================
        cv2.putText(frame, f"Blinks: {blink_total}/{BLINK_EVENT_THRESHOLD}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Yawns:  {yawn_total}/{YAWN_EVENT_THRESHOLD}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Drowsiness Counter (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
