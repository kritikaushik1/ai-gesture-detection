
import os
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "static_landmarks.csv")
MODEL_PATH = os.path.join(BASE_DIR, "static_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")


def load_dataset():
    df = pd.read_csv(CSV_PATH)

    df.columns = df.columns.str.strip()

    return df


def append_new_samples(new_csv_path):
    old_df = load_dataset()

    new_df = pd.read_csv(new_csv_path)

    new_df.columns = new_df.columns.str.strip()

    if list(old_df.columns) != list(new_df.columns):
        raise ValueError(
            "New CSV columns do not match training dataset"
        )

    combined_df = pd.concat(
        [old_df, new_df],
        ignore_index=True
    )

    combined_df = combined_df.dropna()
    combined_df = combined_df.drop_duplicates()

    combined_df.to_csv(
        CSV_PATH,
        index=False
    )

    print("New samples appended successfully")
    print("Total Samples:", len(combined_df))

    return combined_df


def train_model(df):
    X = df.drop(
        "gesture_label",
        axis=1
    )

    y = df["gesture_label"]

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        learning_rate=0.05,
        max_depth=5,
        n_estimators=200,
        random_state=42,
        eval_metric="mlogloss"
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("Retraining completed")
    print("Accuracy:", round(accuracy, 4))
    print("Gesture Classes:", label_encoder.classes_)

    joblib.dump(
        model,
        MODEL_PATH
    )

    joblib.dump(
        label_encoder,
        ENCODER_PATH
    )

    print("Updated model saved")
    print("Updated label encoder saved")


def retrain(new_csv_path=None):
    if new_csv_path is not None:
        df = append_new_samples(
            new_csv_path
        )
    else:
        df = load_dataset()

    train_model(df)


if __name__ == "__main__":
    retrain()
