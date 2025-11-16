# Driver Drowsiness Detection System

A real-time drowsiness detection system that monitors drivers using facial features and alerts them when signs of fatigue are detected, helping prevent road accidents caused by drowsy driving.

## Problem Statement

Road accidents claim approximately three lives every minute, with driver fatigue being a major contributing factor. This system aims to reduce accidents by detecting drowsiness in real-time and alerting drivers to take necessary breaks.

## Features

- **Real-time Monitoring**: Captures and analyzes driver's facial features continuously
- **Eye Closure Detection**: Tracks eye openness, blink rate, and closure duration
- **Eye Aspect Ratio (EAR)**: Measures eye openness to detect drowsiness patterns
- **Alert System**: Triggers audio alarm when drowsiness is detected
- **Facial Landmark Detection**: Uses computer vision to track key facial features

## Technology Stack

- Python
- OpenCV (Computer Vision)
- Keras/TensorFlow (Deep Learning)
- dlib (Facial Landmark Detection)

## Project Structure

```
├── main.py                          # Main application file
├── stream.py                        # Video stream handling
├── driver-drowsiness-using-keras.ipynb  # Model training notebook
├── drowsiness_new6.model            # Trained drowsiness detection model
├── drowsiness_new6.h5               # Model weights
├── drowsiness_mobilenetv2.h5        # MobileNetV2 model variant
├── alarm.wav                        # Alert sound file
├── haarcascade_*.xml                # Face and eye detection cascades
├── requirements.txt                 # Python dependencies
└── Group_21_Report.pdf              # Project report
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main application:
```bash
python main.py
```

The system will activate your webcam and begin monitoring for drowsiness. An alarm will sound if drowsiness is detected.

## How It Works

1. Captures video feed from the camera
2. Detects face and facial landmarks using Haar Cascades
3. Extracts eye regions and calculates Eye Aspect Ratio (EAR)
4. Analyzes patterns using the trained deep learning model
5. Triggers alarm if drowsiness is detected for a sustained period
