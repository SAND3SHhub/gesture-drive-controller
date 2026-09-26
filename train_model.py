import csv
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


DATASET_FILE = "gesture_data_normalized.csv"
MODEL_FILE = "gesture_model.pkl"


# ----------------------------------
# LOAD DATASET
# ----------------------------------

X = []
y = []

with open(DATASET_FILE, "r", newline="") as file:

    reader = csv.reader(file)

    for row in reader:

        # First 63 columns = landmark features
        features = [float(value) for value in row[:-1]]

        # Last column = FIST or OPEN
        label = row[-1]

        X.append(features)
        y.append(label)


print(f"Loaded {len(X)} samples.")


# ----------------------------------
# SPLIT DATA
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ----------------------------------
# CREATE MODEL
# ----------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ----------------------------------
# TRAIN MODEL
# ----------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training complete.")


# ----------------------------------
# TEST MODEL
# ----------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n------------------------------")
print("MODEL RESULTS")
print("------------------------------")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)


# ----------------------------------
# SAVE MODEL
# ----------------------------------

with open(MODEL_FILE, "wb") as file:

    pickle.dump(
        model,
        file
    )


print(f"Model saved to: {MODEL_FILE}")