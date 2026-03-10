# GestureVision — Hand Gesture Recognition

Real-time hand gesture recognition using MediaPipe + OpenCV with a sleek dark UI.

## Gestures Supported
| Gesture | Emoji | Meaning |
|---|---|---|
| Thumbs Up | 👍 | Good / Approve |
| Peace | ✌️ | Hello / Victory |
| Fist | ✊ | Stop / Power |
| Open Hand | 🖐 | Pause / Wave |
| Pointing Up | 👆 | Select / Attention |
| OK Sign | 👌 | Perfect / Confirm |
| Rock On | 🤘 | Awesome / Energy |
| Pinch | 🤌 | Precise / Chef's Kiss |

## Setup

### 1. Clone / download the project
```bash
cd hand-gesture-recognition
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

## Tech Stack
- **MediaPipe** — Hand landmark detection (21 points per hand)
- **OpenCV** — Webcam capture & frame processing
- **Streamlit** — Web UI
- **NumPy** — Distance calculations for gesture logic

## Project Structure
```
hand-gesture-recognition/
├── app.py              # Streamlit UI
├── gesture_engine.py   # MediaPipe + gesture classification
├── requirements.txt
├── .streamlit/
│   └── config.toml     # Dark theme config
└── README.md
```

## Author
Nithin Chowdary — BCA Final Year, Kakatiya University
