import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque, Counter

# ---------------------------
# Model Setup
# ---------------------------
MODEL_PATH = r"C:\Users\Enter Computer\Downloads\mobile_net(7epchs).keras"  # Adjust your model path here
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
    max_num_hands=1,                  # Detect only one hand
    min_detection_confidence=0.7,     # Minimum detection confidence
    min_tracking_confidence=0.7       # Minimum tracking confidence
)

# ---------------------------
# Initialize Camera and Variables
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access camera")
    exit()

PREDICTION_HISTORY_LEN = 10  # Number of recent predictions to stabilize results
prediction_history = deque(maxlen=PREDICTION_HISTORY_LEN)
label = "..."

CLASSES = [
    '01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb',
    '06_index', '07_ok', '08_palm_moved', '09_c', '10_down'
]

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

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract the bounding box around the detected hand
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x1, y1 = int(min(x_coords) * w), int(min(y_coords) * h)
            x2, y2 = int(max(x_coords) * w), int(max(y_coords) * h)

            # Crop the hand region from the frame
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                processed = preprocess_frame(roi)
                preds = model.predict(processed, verbose=0)
                class_index = np.argmax(preds)
                confidence = np.max(preds)

                # Stability: Use majority vote from recent predictions
                prediction_history.append(class_index)
                most_common_index = Counter(prediction_history).most_common(1)[0][0]
                label = f"{CLASSES[most_common_index]} ({confidence*100:.1f}%)"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Draw landmarks and connections on the hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera stopped")
