import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque, Counter


MODEL_PATH = r"C:\Users\rodyd\hand-gesture-recognition-system\mobile_net(7epchs).keras"

# ROI Coordinates (Top-left, Bottom-right)
ROI_X1, ROI_Y1 = 100, 100
ROI_X2, ROI_Y2 = 400, 400

# Prediction stability
PREDICTION_HISTORY_LEN = 10  # Number of frames to average for smoothing
FRAME_SKIP = 3           # Predict every 3 frames

# Class labels
CLASSES = [
    '01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb',
    '06_index', '07_ok', '08_palm_moved', '09_c', '10_down'
]

#  1️ PREPROCESSING FUNCTION (COLOR MODEL)
def preprocess_frame_color(frame):

    # Resize the COLOR frame to match model input
    resized = cv2.resize(frame, (128, 128))
    # Normalize
    resized = resized / 255.0
    # Add batch dimension
    return np.expand_dims(resized, axis=0)  # Shape: (1, 128, 128, 3)
    
# PREPROCESSING FUNCTION (GREYSCALE MODEL)
def preprocess_frame_gray(frame):
 
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize
    gray = cv2.resize(gray, (128, 128))
    # Normalize
    gray = gray / 255.0
    # Add channel dimension
    img = np.expand_dims(gray, axis=-1)   
    # Add batch dimension
    return np.expand_dims(img, axis=0)  

#  LOAD MODEL 
print(f"Loading model from {MODEL_PATH} ...")
model = load_model(MODEL_PATH)
print("Model loaded successfully")
print(f"Model Input Shape: {model.input_shape}")


# -INITIALIZE CAMERA AND VARIABLES 
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access camera")
    exit()

print("Press 'q' to quit\n")

# 2️ Stability variables
prediction_history = deque(maxlen=PREDICTION_HISTORY_LEN)
label = "..."
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # Extract ROI
    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

    # 3️ Performance: Only predict every N frames
    frame_count += 1
    if frame_count % FRAME_SKIP == 0:
        
        #  PREPROCESS AND PREDICT 
        
        processed_roi = preprocess_frame_color(roi) 
      

        preds = model.predict(processed_roi, verbose=0)
        class_index = np.argmax(preds)
        confidence = np.max(preds)

        #  Stability: Add to history
        prediction_history.append(class_index)
        
        # Get the most common prediction from history
        if prediction_history:
            most_common_index = Counter(prediction_history).most_common(1)[0][0]
            label = f"{CLASSES[most_common_index]} ({confidence*100:.1f}%)"

    # DRAWING 
    # Draw ROI rectangle
    cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 0), 2)
    
    # Display the prediction
    cv2.putText(frame, label, (ROI_X1, ROI_Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera stopped")
