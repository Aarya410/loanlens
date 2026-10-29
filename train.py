"""Train the model and write models/model.joblib + models/metrics.json."""
from app.model import train

if __name__ == "__main__":
    m = train()
    print("Model trained:", m)
