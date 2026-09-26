import cv2
import mediapipe as mp

from config import (
    CAMERA_INDEX,
    MAX_HANDS,
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE
)

from gestures import predict_gesture
from steering import calculate_steering
from controller import GameController
from smoothing import GestureSmoother


def main():

    # -----------------------------
    # SETUP
    # -----------------------------

    cap = cv2.VideoCapture(CAMERA_INDEX)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE
    )

    controller = GameController()
    smoother1 = GestureSmoother(window_size=5)
    smoother2 = GestureSmoother(window_size=5)

    print("Gesture Drive Controller started.")
    print("Launch Beach Buggy now.")
    print("Press Q in the camera window to quit.")

    try:

        # -----------------------------
        # MAIN LOOP
        # -----------------------------

        while True:

            success, frame = cap.read()

            if not success:
                print("Could not read webcam.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb_frame)

            detected_hands = []

            if results.multi_hand_landmarks:

                for hand_landmarks in results.multi_hand_landmarks:

                    detected_hands.append(hand_landmarks)

                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            # Automatic Xbox START button
            controller.handle_auto_start()

            # -----------------------------
            # DEFAULT VALUES
            # -----------------------------

            action = "NEUTRAL"
            direction = "STRAIGHT"
            angle = 0.0
            steering_value = 0.0

            gesture1 = "N/A"
            gesture2 = "N/A"

            confidence1 = 0.0
            confidence2 = 0.0

            # -----------------------------
            # TWO HANDS DETECTED
            # -----------------------------

            if len(detected_hands) == 2:

                hand1 = detected_hands[0]
                hand2 = detected_hands[1]

                raw_gesture1, confidence1 = predict_gesture(hand1)
                raw_gesture2, confidence2 = predict_gesture(hand2)

                gesture1 = smoother1.update(raw_gesture1)
                gesture2 = smoother2.update(raw_gesture2)

                fist1 = gesture1 == "FIST"
                fist2 = gesture2 == "FIST"

                # Calculate steering
                angle, steering_value, direction = calculate_steering(
                    hand1,
                    hand2
                )

                controller.steer(steering_value)

                # Both fists = accelerate
                if fist1 and fist2:

                    controller.accelerate()
                    action = "ACCELERATE"

                # Both open = brake/reverse
                elif not fist1 and not fist2:

                    controller.brake()
                    action = "BRAKE"

                # Mixed = coast
                else:

                    controller.coast()
                    action = "COAST"

            # -----------------------------
            # LESS THAN TWO HANDS
            # -----------------------------

            else:

                controller.steer(0.0)
                controller.coast()

            # Send controller state
            controller.update()

            # -----------------------------
            # DISPLAY
            # -----------------------------

            cv2.putText(
                frame,
                f"Action: {action}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Steering: {direction}",
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Angle: {angle:.1f} deg",
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Input: {steering_value * 100:.0f}%",
                (30, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Hand 1: {gesture1} ({confidence1:.0f}%)",
                (30, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Hand 2: {gesture2} ({confidence2:.0f}%)",
                (30, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Gesture Drive Controller",
                frame
            )

            # Q quits
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        # -----------------------------
        # CLEAN SHUTDOWN
        # -----------------------------

        controller.reset()

        cap.release()
        hands.close()

        cv2.destroyAllWindows()

        print("Controller stopped safely.")


if __name__ == "__main__":
    main()