import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque, Counter
import pyautogui
import time

# ---------------------------
# Model Setup
# ---------------------------
MODEL_PATH = r"C:\Users\rodyd\hand-gesture-recognition-system\mobile_net(7epchs).keras"
print(f"Loading model from {MODEL_PATH} ...")
model = load_model(MODEL_PATH)
print("Model loaded successfully")
print(f"Model Input Shape: {model.input_shape}")

# ---------------------------
# Initialize MediaPipe Hands
# ---------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
# ---------------------------
# Initialize Camera and Variables
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access camera")
    exit()

PREDICTION_HISTORY_LEN = 5
prediction_history = deque(maxlen=PREDICTION_HISTORY_LEN)
STABILITY_THRESHOLD = 70.0

CLASSES = [
    '01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb',
    '06_index', '07_ok', '08_palm_moved', '09_c', '10_down'
]


PADDING_FACTOR = 1.2

# --- 2. ADD "COOLDOWN" FLAG ---
action_triggered = False

print("Press 'q' to quit\n")

# ---------------------------
# Preprocessing Function
# ---------------------------
def preprocess_frame(frame):
    """
    Resize the frame to 128x128, normalize pixel values,
    and add a batch dimension for model prediction.
    """
    resized = cv2.resize(frame, (128, 128)) / 255.0
    return np.expand_dims(resized, axis=0)  # Shape: (1,128,128,3)

# ---------------------------
# Real-Time Camera Loop
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)
    current_label = ""

    debug_stability = 0.0
    debug_box_msg = ""

    hand_detected = False # To check if we should reset our trigger

    if result.multi_hand_landmarks:
        hand_detected = True # We see a hand
        for hand_landmarks in result.multi_hand_landmarks:

            # --- Bounding box logic ---
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

            # --- Bounding box validation ---
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

                # ----------------- INTUITIVE GESTURE-TO-ACTION LOGIC -----------------
                if stability >= STABILITY_THRESHOLD:
                    current_label = CLASSES[most_common_index]
                    
                    # Check if a stable gesture has been detected
                    if not action_triggered:
                        
                        # --- Gesture 01_palm (Index 0): Show Desktop ---
                        if most_common_index == 0:
                            print("ACTION: Show Desktop/Minimize All 🖥️")
                            pyautogui.hotkey('win', 'd') 
                            action_triggered = True
                            
                        # --- Gesture 02_l (Index 1): Play/Pause ---
                        elif most_common_index == 1:
                            print("ACTION: Play/Pause ⏯️")
                            pyautogui.press('playpause')
                            action_triggered = True
                            
                        # --- Gesture 03_fist (Index 2): Mute/Unmute ---
                        elif most_common_index == 2:
                            print("ACTION: Mute/Unmute 🔇")
                            pyautogui.press('volumemute')
                            action_triggered = True
                            
                        # --- Gesture 04_fist_moved (Index 3): Next Track/Right Arrow ---
                        elif most_common_index == 3:
                            print("ACTION: Next Track ➡️")
                            pyautogui.press('right')
                            action_triggered = True
                            
                        # --- Gesture 05_thumb (Index 4): Volume Up ---
                        elif most_common_index == 4:
                            print("ACTION: Volume Up! 🔊")
                            pyautogui.press('volumeup')
                            action_triggered = True
                            
                        # --- Gesture 06_index (Index 5): Scroll Up (Page Up) ---
                        elif most_common_index == 5:
                            print("ACTION: Scroll Up ⬆️")
                            pyautogui.scroll(10)
                            action_triggered = True
                            
                        # --- Gesture 07_ok (Index 6): Enter/Select ---
                        elif most_common_index == 6:
                            print("ACTION: Enter/Select ⏎")
                            pyautogui.press('enter')
                            action_triggered = True
                            
                        # --- Gesture 08_palm_moved (Index 7): Previous Track/Left Arrow ---
                        elif most_common_index == 7:
                            print("ACTION: Previous Track ⬅️")
                            pyautogui.press('left')
                            action_triggered = True
                            
                        # --- Gesture 09_c (Index 8): Take Screenshot ---
                        elif most_common_index == 8:
                            print("ACTION: Taking Screenshot! 📸")
                            ss = pyautogui.screenshot()
                            ss_filename = f"screenshot_{int(time.time())}.png"
                            ss.save(ss_filename)
                            print(f"Saved as {ss_filename}")
                            action_triggered = True
                            
                        # --- Gesture 10_down (Index 9): Volume Down ---
                        elif most_common_index == 9:
                            print("ACTION: Volume Down 🔉")
                            pyautogui.press('volumedown') 
                            action_triggered = True
                            
                    # If action was already triggered, the flag prevents re-triggering.
                    
                else:
                    # If prediction is unstable, reset the flag
                    action_triggered = False
                # ----------------- END INTUITIVE GESTURE-TO-ACTION LOGIC -----------------


            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # If NO hand is detected, reset the trigger
    if not hand_detected:
        action_triggered = False

    # --- DEBUG TEXT ---
    cv2.putText(frame, f"Prediction: {current_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, f"Stability: {debug_stability:.1f}%", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Box: {debug_box_msg}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera stopped")
