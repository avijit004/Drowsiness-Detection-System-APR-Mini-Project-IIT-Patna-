import streamlit as st
import cv2
import time
import numpy as np
import mediapipe as mp
import subprocess
import os

MODEL_PATH = "drowsiness_mobilenetv2.h5"
USE_MODEL = os.path.exists(MODEL_PATH)

if USE_MODEL:
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    LABELS = ["yawn", "no_yawn", "Closed", "Open"]

# ================== MEDIA PIPE ==================
mp_face_mesh = mp.solutions.face_mesh
FACE_MESH = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
MOUTH_H = (61,291)
MOUTH_V = (13,14)

def _p2xy(lm, w, h): return np.array([lm.x*w, lm.y*h],dtype=np.float32)
def _dist(a,b): return float(np.linalg.norm(a-b))
def ear(lm,w,h,idx):
    p=[_p2xy(lm[i],w,h) for i in idx]
    return (_dist(p[1],p[5])+_dist(p[2],p[4]))/(2*_dist(p[0],p[3])+1e-6)
def mar(lm,w,h):
    pL=_p2xy(lm[MOUTH_H[0]],w,h); pR=_p2xy(lm[MOUTH_H[1]],w,h)
    pU=_p2xy(lm[MOUTH_V[0]],w,h); pD=_p2xy(lm[MOUTH_V[1]],w,h)
    return _dist(pU,pD)/(_dist(pL,pR)+1e-6)

# ================== SOUND ALERT ==================
def beep():
    if os.path.exists("alarm.wav"):
        subprocess.Popen(["afplay","alarm.wav"])
    else:
        print("\a")

# ================== DROWSINESS PARAMETERS ==================
EAR_THRESH = 0.19
MAR_THRESH = 0.45
YAWN_FRAMES = 4
EYE_CLOSED_LIMIT = 3
COOLDOWN = 5

# ================== UI LAYOUT ==================
st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")

st.markdown("<h1 style='text-align:center;'>🚗 Real-Time Driver Drowsiness Monitoring</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:grey;'>Stay safe on the road. This system monitors your alertness.</p>", unsafe_allow_html=True)
st.write("---")

col1, col2 = st.columns([2,1])

start = col2.button("▶ Start Detection")
stop = col2.button("⏹ Stop")

status_box = st.empty()
video_frame = col1.empty()
metrics = col2.empty()

# ================== MAIN PROCESS ==================
def main():
    cap = cv2.VideoCapture(0)
    eye_closed_start=None
    yawn_counter=0
    last_beep=0

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame=cv2.flip(frame,1)
        h,w=frame.shape[:2]
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res=FACE_MESH.process(rgb)

        drowsy=False
        reasons=[]
        ear_val=0; mar_val=0

        if not res.multi_face_landmarks:
            drowsy=True
            reasons.append("Face/Eyes Not Visible")
        else:
            lm=res.multi_face_landmarks[0].landmark
            ear_val=(ear(lm,w,h,LEFT_EYE)+ear(lm,w,h,RIGHT_EYE))/2
            mar_val=mar(lm,w,h)

            eye_closed=ear_val<EAR_THRESH
            yawning=mar_val>MAR_THRESH

            if yawning:
                yawn_counter+=1
            else:
                yawn_counter=0
            if yawn_counter>=YAWN_FRAMES:
                drowsy=True; reasons.append("Yawning")

            if eye_closed:
                if eye_closed_start is None:
                    eye_closed_start=time.time()
                elif time.time()-eye_closed_start>=EYE_CLOSED_LIMIT:
                    drowsy=True; reasons.append(f"Eyes Closed > {EYE_CLOSED_LIMIT}s")
            else:
                eye_closed_start=None

        if drowsy:
            now=time.time()
            if now-last_beep>=COOLDOWN:
                beep()
                last_beep=now
            cv2.rectangle(frame,(0,0),(w,60),(0,0,255),-1)
            cv2.putText(frame,"DROWSY: "+" + ".join(reasons),(10,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        status_box.markdown(
            f"<h2 style='text-align:center;color:{'red' if drowsy else 'green'};'>"
            f"{'⚠ DROWSY' if drowsy else '✅ ALERT & SAFE'}</h2>", unsafe_allow_html=True)

        metrics.write(f"""
        **EAR:** {ear_val:.3f} (threshold {EAR_THRESH})  
        **MAR:** {mar_val:.3f} (threshold {MAR_THRESH})  
        **Yawn Counter:** {yawn_counter}/{YAWN_FRAMES}  
        """)

        video_frame.image(frame,channels="BGR")

        if stop: break
    cap.release()

if start:
    main()
