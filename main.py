import pandas as pd
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# 2. download metadata
metadata_path = "UrbanSound8K/metadata/UrbanSound8K.csv"
audio_path = "UrbanSound8K/audio/"

df = pd.read_csv(metadata_path)

print("Original classes from dataset:")
print(df['class'].unique())



# 3. Create our classes classters
mapping = {
    "engine_idling": "transport",
    "car_horn": "transport",

    "children_playing": "human",
    "street_music": "human",

    "siren": "alert",

    "drilling": "building_noise",
    "jackhammer": "building_noise"
}

df["new_class"] = df["class"].map(mapping)

df["new_class"] = df["new_class"].fillna("others")

print("\nnew classes:")
print(df["new_class"].value_counts())


# 4. Function extract MFCC with padding till 5 seccond
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=5)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Error with file {file_path}: {e}")
        return None


X = []
y = []

for index, row in df.iterrows():
    file_name = row["slice_file_name"]
    fold = row["fold"]

    file_path = os.path.join(audio_path, f"fold{fold}", file_name)

    features = extract_features(file_path)

    if features is not None:
        X.append(features)
        y.append(row["new_class"])

X = np.array(X)
y = np.array(y)

print("\ndata volume:", X.shape)


# 6. normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)


# 7. split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 9. Predict
y_pred = model.predict(X_test)



# 10. Accuracy of model CNN
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))



# 11. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

labels = np.unique(y)
plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)

plt.xlabel("Predicted")
plt.ylabel("True")

for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()




# =========================
# 10. Predict custom
# =========================
def extract_features(file_path, max_duration=5):
    try:
        audio, sr = librosa.load(file_path, duration=max_duration)
        if len(audio) < max_duration * sr:
            pad_width = max_duration * sr - len(audio)
            audio = np.pad(audio, (0, pad_width), mode='constant')
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Error with file {file_path}: {e}")
        return None


def predict_audio_class(file_path, model, scaler):
    features = extract_features(file_path, max_duration=5)
    if features is None:
        return None
    features = scaler.transform([features])
    predicted_class = model.predict(features)[0]
    return predicted_class


user_audio_folder = "UrbanSound8K/Test/user_audio/"

for file_name in os.listdir(user_audio_folder):
    if file_name.endswith(".wav"):
        file_path = os.path.join(user_audio_folder, file_name)
        predicted_class = predict_audio_class(file_path, model, scaler)
        print(f"File {file_name} → class: {predicted_class}")
