import os
from typing import List, Tuple
import joblib
from sklearn.pipeline import Pipeline

def load_model():
    # Load the model
    model_file = os.path.join(os.path.dirname(__file__),
    "newgroups_model.joblib")
    loaded_model: Tuple[Pipeline, List[str]] = joblib.load(model_file)

    model, targets = loaded_model

    return model, targets
    

if __name__ == "__main__":

    model, targets = load_model()

    # Run a prediction
    p = model.predict(["Computer cpu memory ram"])
    print(targets[p[0]])