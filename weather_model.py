# weather_model.py

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report
import joblib
from sklearn.utils import resample

# Load data from CSV file
data = pd.read_csv("historical_weather_three_moods.csv")

# Upsample minority classes to balance 'pleasant', 'neutral', and 'bad'
from collections import Counter
max_count = data["weather_mood"].value_counts().max()
resampled_data = []

for mood in data["weather_mood"].unique():
    subset = data[data["weather_mood"] == mood]
    if len(subset) < max_count:
        upsampled = resample(subset, replace=True, n_samples=max_count, random_state=42)
        resampled_data.append(upsampled)
    else:
        resampled_data.append(subset)

data = pd.concat(resampled_data)
print("✅ After upsampling:\n", data['weather_mood'].value_counts())

# Numeric features
num_features = data[["temp", "humidity", "wind_speed"]]

# Categorical feature (OneHotEncode weather_main)
cat_features = data[["weather_main"]]
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cat_encoded = ohe.fit_transform(cat_features)
cat_encoded_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(["weather_main"]))

# Combine features
X = pd.concat([num_features.reset_index(drop=True), cat_encoded_df.reset_index(drop=True)], axis=1)

# Target variable (LabelEncode weather_mood)
le = LabelEncoder()
y = le.fit_transform(data["weather_mood"].values)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Initialize GradientBoostingClassifier with tuned parameters for balanced accuracy
model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.85,
    min_samples_split=10,
    min_samples_leaf=3,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print(f"Train Accuracy: {model.score(X_train, y_train):.4f}")

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Evaluate on test set
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# Save model and encoders
joblib.dump(model, "weather_mood_model_v3.pkl")
joblib.dump(ohe, "weather_main_encoder.pkl")
joblib.dump(le, "weather_mood_encoder.pkl")
print("✅ Model, OneHotEncoder, and LabelEncoder saved.")
