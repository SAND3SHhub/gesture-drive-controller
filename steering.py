import math

from config import DEAD_ZONE, MAX_ANGLE


def calculate_steering(hand1, hand2):
    """
    Calculates steering from the angle between two wrists.

    Returns:
        angle: actual hand angle in degrees
        steering_value: Xbox joystick value from -1.0 to +1.0
        direction: LEFT, RIGHT, or STRAIGHT
    """

    # Landmark 0 is the wrist
    x1 = hand1.landmark[0].x
    y1 = hand1.landmark[0].y

    x2 = hand2.landmark[0].x
    y2 = hand2.landmark[0].y


    # Difference between wrist positions
    dx = x2 - x1
    dy = y2 - y1


    # Calculate angle of the invisible line
    # connecting the two wrists
    angle = math.degrees(
        math.atan2(dy, dx)
    )


    # Normalize the angle
    if angle > 90:
        angle -= 180

    elif angle < -90:
        angle += 180


    # Dead zone
    if abs(angle) < DEAD_ZONE:

        steering_value = 0.0
        direction = "STRAIGHT"

    else:

        # Convert hand angle to Xbox joystick range
        steering_value = angle / MAX_ANGLE


        # Limit value between -1 and +1
        steering_value = max(
            -1.0,
            min(1.0, steering_value)
        )


        if steering_value < 0:
            direction = "LEFT"

        else:
            direction = "RIGHT"


    return angle, steering_value, direction