import cv2
import csv
import mediapipe as mp

from config import (
    CAMERA_INDEX,
    MAX_HANDS,
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE
)


# ----------------------------------
# SETTINGS
# ----------------------------------

DATASET_FILE = "gesture_data_raw.csv"


# ----------------------------------
# MEDIAPIPE SETUP
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


print("Gesture Data Collector")
print("----------------------")
print("Press F to save a FIST")
print("Press O to save an OPEN hand")
print("Press Q to quit")


while True:

    success, frame = cap.read()

    if not success:
        print("Could not access webcam.")
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    # Convert BGR -> RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    results = hands.process(rgb_frame)


    current_hand = None


    if results.multi_hand_landmarks:

        current_hand = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            current_hand,
            mp_hands.HAND_CONNECTIONS
        )


    cv2.putText(
        frame,
        "F = FIST | O = OPEN | Q = QUIT",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    cv2.imshow(
        "Gesture Data Collector",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    # ----------------------------------
    # SAVE FIST
    # ----------------------------------

    if key == ord("f") and current_hand is not None:

        row = []

        for landmark in current_hand.landmark:

            row.append(landmark.x)
            row.append(landmark.y)
            row.append(landmark.z)

        row.append("FIST")


        with open(
            DATASET_FILE,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)
            writer.writerow(row)


        print("Saved: FIST")


    # ----------------------------------
    # SAVE OPEN
    # ----------------------------------

    elif key == ord("o") and current_hand is not None:

        row = []

        for landmark in current_hand.landmark:

            row.append(landmark.x)
            row.append(landmark.y)
            row.append(landmark.z)

        row.append("OPEN")


        with open(
            DATASET_FILE,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)
            writer.writerow(row)


        print("Saved: OPEN")


    # ----------------------------------
    # QUIT
    # ----------------------------------

    elif key == ord("q"):
        break


cap.release()
hands.close()
cv2.destroyAllWindows()

print("Data collector closed.")