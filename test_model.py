import cv2
import math
import pickle
import mediapipe as mp

from config import (
    CAMERA_INDEX,
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE
)


MODEL_FILE = "gesture_model.pkl"


# ----------------------------------
# NORMALIZATION
# ----------------------------------

def normalize_landmarks(hand_landmarks):
    """
    Converts MediaPipe's 21 landmarks into the same
    63 normalized values used during model training.
    """

    landmarks = []

    for landmark in hand_landmarks.landmark:
        landmarks.append([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    # Wrist becomes the origin
    wrist_x = landmarks[0][0]
    wrist_y = landmarks[0][1]
    wrist_z = landmarks[0][2]

    for landmark in landmarks:
        landmark[0] -= wrist_x
        landmark[1] -= wrist_y
        landmark[2] -= wrist_z

    # Wrist -> middle MCP distance gives us hand scale
    middle_mcp = landmarks[9]

    hand_size = math.sqrt(
        middle_mcp[0] ** 2
        + middle_mcp[1] ** 2
        + middle_mcp[2] ** 2
    )

    if hand_size == 0:
        hand_size = 1

    # Normalize hand size
    for landmark in landmarks:
        landmark[0] /= hand_size
        landmark[1] /= hand_size
        landmark[2] /= hand_size

    # Convert 21 × 3 back into 63 values
    normalized = []

    for landmark in landmarks:
        normalized.extend(landmark)

    return normalized


# ----------------------------------
# LOAD TRAINED MODEL
# ----------------------------------

with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)

print("Gesture model loaded.")


# ----------------------------------
# MEDIAPIPE
# ----------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=DETECTION_CONFIDENCE,
    min_tracking_confidence=TRACKING_CONFIDENCE
)


# ----------------------------------
# CAMERA
# ----------------------------------

cap = cv2.VideoCapture(CAMERA_INDEX)


while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)

    prediction = "NO HAND"
    confidence = 0.0


    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        # Normalize live landmarks
        features = normalize_landmarks(hand)

        # Random Forest expects a list of samples.
        # We have one sample containing 63 features.
        prediction = model.predict([features])[0]

        # Get prediction probabilities
        probabilities = model.predict_proba([features])[0]

        confidence = max(probabilities) * 100


    cv2.putText(
        frame,
        f"Gesture: {prediction}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.1f}%",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    cv2.imshow(
        "ML Gesture Test",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
hands.close()
cv2.destroyAllWindows()

print("Test closed.")