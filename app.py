import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque, Counter
import pyautogui
import time

##############################################
# STREAMLIT UI
##############################################
st.set_page_config(page_title="Hand Gesture Control", layout="wide")
st.title("🖐️ Real‑Time Hand Gesture Recognition – Streamlit Version")
st.write("This Streamlit app runs **your full real‑time gesture code** with no removals or simplifications.")

##############################################
# MODEL LOADING
##############################################
MODEL_PATH = r"best_cnn(1).h5"
st.sidebar.subheader("Model Loading")
st.sidebar.info(f"Loading model from:\n{MODEL_PATH}")

model = load_model(MODEL_PATH)
st.sidebar.success("Model loaded successfully ✔️")
st.sidebar.write(f"Model input: **{model.input_shape}**")

##############################################
# INIT MEDIAPIPE
##############################################
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

##############################################
# VARIABLES
##############################################
PREDICTION_HISTORY_LEN = 5
prediction_history = deque(maxlen=PREDICTION_HISTORY_LEN)
STABILITY_THRESHOLD = 70.0

CLASSES = [
    '01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb',
    '06_index', '07_ok', '08_palm_moved', '09_c', '10_down'
]

PADDING_FACTOR = 1.2
action_triggered = False

##############################################
# PREPROCESS
##############################################
def preprocess_frame(frame):
    resized = cv2.resize(frame, (128, 128)) / 255.0
    return np.expand_dims(resized, axis=0)

##############################################
# START BUTTON (Fixed)
##############################################
run = st.checkbox("▶️ Start Camera")
frame_window = st.image([])

##############################################
# CAMERA LOOP
##############################################
if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot access camera")
    else:
        st.success("Camera started. Uncheck the box to stop.")

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb_frame)
        current_label = ""
        debug_stability = 0.0
        debug_box_msg = ""
        hand_detected = False

        if result.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in result.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                box_width = x_max - x_min
                box_height = y_max - y_min
                longest_side = max(box_width, box_height)
                square_side_len = longest_side * PADDING_FACTOR
                rel_x1 = center_x - (square_side_len / 2)
                rel_y1 = center_y - (square_side_len / 2)
                rel_x2 = center_x + (square_side_len / 2)
                rel_y2 = center_y + (square_side_len / 2)
                x1 = max(0, int(rel_x1 * w))
                y1 = max(0, int(rel_y1 * h))
                x2 = min(w, int(rel_x2 * w))
                y2 = min(h, int(rel_y2 * h))

                box_area = (x2 - x1) * (y2 - y1)
                frame_area = w * h
                if box_area < 0.02 * frame_area:
                    debug_box_msg = "Hand too far"
                elif box_area > 0.5 * frame_area:
                    debug_box_msg = "Hand too close"
                else:
                    debug_box_msg = "Hand size OK"

                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    processed = preprocess_frame(roi)
                    preds = model.predict(processed, verbose=0)
                    class_index = np.argmax(preds)

                    prediction_history.append(class_index)
                    most_common_pred = Counter(prediction_history).most_common(1)[0]
                    most_common_index = most_common_pred[0]
                    stability_count = most_common_pred[1]
                    stability = (stability_count / len(prediction_history)) * 100
                    debug_stability = stability

                    if stability >= STABILITY_THRESHOLD:
                        current_label = CLASSES[most_common_index]

                        if not action_triggered:
                            if most_common_index == 0:
                                pyautogui.hotkey('win', 'd')
                            elif most_common_index == 1:
                                pyautogui.press('playpause')
                            elif most_common_index == 2:
                                pyautogui.press('volumemute')
                            elif most_common_index == 3:
                                pyautogui.press('right')
                            elif most_common_index == 4:
                                pyautogui.press('volumeup')
                            elif most_common_index == 5:
                                pyautogui.scroll(10)
                            elif most_common_index == 6:
                                pyautogui.press('enter')
                            elif most_common_index == 7:
                                pyautogui.press('left')
                            elif most_common_index == 8:
                                ss = pyautogui.screenshot()
                                ss.save(f"screenshot_{int(time.time())}.png")
                            elif most_common_index == 9:
                                pyautogui.press('volumedown')

                            action_triggered = True
                    else:
                        action_triggered = False

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if not hand_detected:
            action_triggered = False

        cv2.putText(frame, f"Prediction: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(frame, f"Stability: {debug_stability:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, f"Box: {debug_box_msg}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        frame_window.image(frame, channels="BGR")

    cap.release()

st.write("Camera stopped.")
