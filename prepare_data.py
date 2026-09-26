import csv
import math


INPUT_FILE = "gesture_data_raw.csv"
OUTPUT_FILE = "gesture_data_normalized.csv"


def normalize_landmarks(values):
    """
    Takes 63 raw landmark values:
    x0, y0, z0, ... x20, y20, z20

    Returns 63 normalized values.
    """

    # ----------------------------------
    # STEP 1: Rebuild the 21 landmarks
    # ----------------------------------

    landmarks = []

    for i in range(0, 63, 3):

        x = float(values[i])
        y = float(values[i + 1])
        z = float(values[i + 2])

        landmarks.append([x, y, z])


    # ----------------------------------
    # STEP 2: Make wrist the origin
    # ----------------------------------

    wrist_x = landmarks[0][0]
    wrist_y = landmarks[0][1]
    wrist_z = landmarks[0][2]


    for landmark in landmarks:

        landmark[0] -= wrist_x
        landmark[1] -= wrist_y
        landmark[2] -= wrist_z


    # ----------------------------------
    # STEP 3: Calculate hand size
    # ----------------------------------

    # Landmark 9 = middle finger MCP joint
    middle_mcp = landmarks[9]

    hand_size = math.sqrt(
        middle_mcp[0] ** 2
        + middle_mcp[1] ** 2
        + middle_mcp[2] ** 2
    )


    # Prevent division by zero
    if hand_size == 0:
        hand_size = 1


    # ----------------------------------
    # STEP 4: Normalize scale
    # ----------------------------------

    for landmark in landmarks:

        landmark[0] /= hand_size
        landmark[1] /= hand_size
        landmark[2] /= hand_size


    # ----------------------------------
    # STEP 5: Flatten back to 63 values
    # ----------------------------------

    normalized = []

    for landmark in landmarks:

        normalized.extend(landmark)


    return normalized


# ----------------------------------
# PROCESS DATASET
# ----------------------------------

sample_count = 0


with open(INPUT_FILE, "r", newline="") as input_file, \
     open(OUTPUT_FILE, "w", newline="") as output_file:

    reader = csv.reader(input_file)
    writer = csv.writer(output_file)

    for row in reader:

        # Last column is FIST or OPEN
        features = row[:-1]
        label = row[-1]

        normalized_features = normalize_landmarks(features)

        normalized_features.append(label)

        writer.writerow(normalized_features)

        sample_count += 1


print(f"Finished normalizing {sample_count} samples.")
print(f"Saved to: {OUTPUT_FILE}")