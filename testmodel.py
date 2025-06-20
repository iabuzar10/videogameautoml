import pickle

model_path = "app/models/best_flamlmodel.pkl"
encoder_path = "app/models/label_encoder.pkl"

# Check model
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)

# Check encoder
try:
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    print("✅ Encoder loaded successfully!")
except Exception as e:
    print("❌ Error loading encoder:", e)
