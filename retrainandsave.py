import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from flaml import AutoML

# ✅ Load your dataset (adjust this path as needed)
df = pd.read_csv("app/data/Video_Games_Sales_as_at_22_Dec_2016.csv")
 # <-- Replace with correct path

# ✅ Preprocessing
df = df.dropna()
target_column = "Genre"  # <-- Replace with your actual target column
X = df.drop(columns=[target_column])
y = df[target_column]

# ✅ Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ✅ Clean feature names
X.columns = X.columns.str.replace(r"[^\w]", "_", regex=True)

# ✅ Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ✅ Train AutoML
automl = AutoML()
automl.fit(
    X_train=X_train,
    y_train=y_train,
    task="classification",
    time_budget=60,
    model_history=True  # ✅ Enables model ranking tracking
)

# ✅ Save model and label encoder
os.makedirs("models", exist_ok=True)

with open("models/best_flamlmodel.pkl", "wb") as f:
    pickle.dump(automl, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Model and encoder saved successfully with current environment.")
