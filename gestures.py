import math
import pickle


MODEL_FILE = "gesture_model.pkl"


# Load the trained model once when this file is imported
with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)


def normalize_landmarks(hand_landmarks):
    """
    Converts MediaPipe's 21 landmarks into the same
    63 normalized values used during training.
    """

    landmarks = []

    for landmark in hand_landmarks.landmark:
        landmarks.append([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    # Make the wrist the origin
    wrist_x = landmarks[0][0]
    wrist_y = landmarks[0][1]
    wrist_z = landmarks[0][2]

    for landmark in landmarks:
        landmark[0] -= wrist_x
        landmark[1] -= wrist_y
        landmark[2] -= wrist_z

    # Use wrist -> middle MCP distance as hand scale
    middle_mcp = landmarks[9]

    hand_size = math.sqrt(
        middle_mcp[0] ** 2
        + middle_mcp[1] ** 2
        + middle_mcp[2] ** 2
    )

    if hand_size == 0:
        hand_size = 1

    # Normalize scale
    for landmark in landmarks:
        landmark[0] /= hand_size
        landmark[1] /= hand_size
        landmark[2] /= hand_size

    # Flatten 21 x 3 into 63 values
    normalized = []

    for landmark in landmarks:
        normalized.extend(landmark)

    return normalized


def predict_gesture(hand_landmarks):
    """
    Predicts the gesture using the trained ML model.

    Returns:
        gesture: FIST or OPEN
        confidence: prediction confidence percentage
    """

    features = normalize_landmarks(hand_landmarks)

    gesture = model.predict([features])[0]

    probabilities = model.predict_proba([features])[0]

    confidence = max(probabilities) * 100

    return gesture, confidence


def is_fist(hand_landmarks):
    """
    Keeps compatibility with main.py.

    Returns True if ML predicts FIST.
    """

    gesture, _ = predict_gesture(hand_landmarks)

    return gesture == "FIST"