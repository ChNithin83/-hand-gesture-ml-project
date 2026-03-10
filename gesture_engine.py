import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import screeninfo

# ── Safety: pyautogui won't throw on edge of screen ──────────
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# ── Screen size ───────────────────────────────────────────────
try:
    screen = screeninfo.get_monitors()[0]
    SCREEN_W, SCREEN_H = screen.width, screen.height
except:
    SCREEN_W, SCREEN_H = 1920, 1080

# ── Gesture Definitions ───────────────────────────────────────
GESTURES = {
    "THUMBS_UP":    {"emoji": "👍", "name": "Thumbs Up",    "meaning": "Good / Approve"},
    "THUMBS_DOWN":  {"emoji": "👎", "name": "Thumbs Down",  "meaning": "Bad / Reject"},
    "PEACE":        {"emoji": "✌️",  "name": "Peace",        "meaning": "Hello / Victory"},
    "FIST":         {"emoji": "✊", "name": "Fist",         "meaning": "Stop / Power"},
    "OPEN_HAND":    {"emoji": "🖐", "name": "Open Hand",    "meaning": "Pause / Wave"},
    "POINTING":     {"emoji": "👆", "name": "Move Cursor",  "meaning": "Mouse Control ON"},
    "OK_SIGN":      {"emoji": "👌", "name": "OK Sign",      "meaning": "Left Click"},
    "ROCK":         {"emoji": "🤘", "name": "Rock On",      "meaning": "Awesome / Energy"},
    "PINCH":        {"emoji": "🤌", "name": "Pinch",        "meaning": "Right Click"},
    "NO_GESTURE":   {"emoji": "—",  "name": "No Gesture",   "meaning": "Nothing detected"},
}


class GestureEngine:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )

        # Drawing style - custom colors
        self.landmark_style = self.mp_draw.DrawingSpec(
            color=(167, 139, 250), thickness=2, circle_radius=3  # violet
        )
        self.connection_style = self.mp_draw.DrawingSpec(
            color=(56, 189, 248), thickness=2  # cyan
        )

        # Mouse control state
        self.prev_x, self.prev_y = 0, 0
        self.smoothening = 5
        self.click_cooldown = 0

    def _move_mouse(self, index_tip, frame_w, frame_h):
        """Move mouse cursor based on index finger position"""
        # Map finger position to screen coordinates
        x = np.interp(index_tip.x, [0.1, 0.9], [SCREEN_W, 0])  # flipped (mirror)
        y = np.interp(index_tip.y, [0.1, 0.9], [0, SCREEN_H])

        # Smoothen movement
        smooth_x = self.prev_x + (x - self.prev_x) / self.smoothening
        smooth_y = self.prev_y + (y - self.prev_y) / self.smoothening

        pyautogui.moveTo(smooth_x, smooth_y)
        self.prev_x, self.prev_y = smooth_x, smooth_y

    def _fingers_up(self, landmarks):
        """Returns list of booleans [thumb, index, middle, ring, pinky]"""
        tips = [4, 8, 12, 16, 20]
        fingers = []

        # Thumb: compare x (left/right hand)
        if landmarks[tips[0]].x < landmarks[tips[0] - 1].x:
            fingers.append(True)
        else:
            fingers.append(False)

        # Other fingers: compare y (up/down)
        for i in range(1, 5):
            if landmarks[tips[i]].y < landmarks[tips[i] - 2].y:
                fingers.append(True)
            else:
                fingers.append(False)

        return fingers

    def _classify(self, landmarks) -> tuple[str, int]:
        """Returns (gesture_key, confidence%)"""
        f = self._fingers_up(landmarks)
        # f = [thumb, index, middle, ring, pinky]

        # Distances for special gestures
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
        index_tip = np.array([landmarks[8].x, landmarks[8].y])
        pinch_dist = np.linalg.norm(thumb_tip - index_tip)

        if f == [False, False, False, False, False]:
            return "FIST", 95

        if f == [False, True, True, False, False]:
            return "PEACE", 93

        if f == [True, False, False, False, False]:
            return "THUMBS_UP", 92

        if f == [False, False, False, False, False] and landmarks[4].y > landmarks[3].y:
            return "THUMBS_DOWN", 90

        if f == [True, True, True, True, True]:
            return "OPEN_HAND", 96

        if f == [False, True, False, False, False]:
            return "POINTING", 94

        if f == [True, False, False, False, True]:
            return "ROCK", 91

        # OK sign: thumb + index close, others up
        if pinch_dist < 0.05 and f[2] and f[3] and f[4]:
            return "OK_SIGN", 89

        # Pinch: thumb + index close
        if pinch_dist < 0.06:
            return "PINCH", 87

        return "NO_GESTURE", 60

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, dict | None]:
        """Process frame, draw landmarks, return annotated frame + gesture info"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        gesture_info = None

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_lm,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.landmark_style,
                    self.connection_style
                )

                # Classify
                gesture_key, confidence = self._classify(hand_lm.landmark)

                if gesture_key != "NO_GESTURE":
                    g = GESTURES[gesture_key].copy()
                    g["confidence"] = confidence
                    gesture_info = g

                    # Mouse control actions
                    if gesture_key == "POINTING":
                        # Move cursor with index finger
                        self._move_mouse(hand_lm.landmark[8], frame.shape[1], frame.shape[0])
                        # Draw cursor dot on frame
                        h, w = frame.shape[:2]
                        cx = int(hand_lm.landmark[8].x * w)
                        cy = int(hand_lm.landmark[8].y * h)
                        cv2.circle(frame, (cx, cy), 10, (56, 189, 248), -1)
                        cv2.circle(frame, (cx, cy), 14, (167, 139, 250), 2)

                    elif gesture_key == "OK_SIGN" and self.click_cooldown == 0:
                        # Left click
                        pyautogui.click()
                        self.click_cooldown = 20  # frames cooldown

                    elif gesture_key == "PINCH" and self.click_cooldown == 0:
                        # Right click
                        pyautogui.rightClick()
                        self.click_cooldown = 20

                    # Cooldown countdown
                    if self.click_cooldown > 0:
                        self.click_cooldown -= 1

                    # Draw label on frame
                    cv2.rectangle(frame, (10, 10), (320, 60), (15, 15, 26), -1)
                    cv2.rectangle(frame, (10, 10), (320, 60), (167, 139, 250), 1)
                    cv2.putText(frame, g['name'], (20, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (167, 139, 250), 2)

        return frame, gesture_info
