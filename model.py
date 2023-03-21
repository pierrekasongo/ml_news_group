import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import os
from typing import List, Tuple
import joblib

# Load some categories of newsgroups dataset
categories = [
    "soc.religion.christian",
    "talk.religion.misc",
    "comp.sys.mac.hardware",
    "sci.crypt",
]

def train_model():
        
    newsgroups_training = fetch_20newsgroups(
        subset = "train", categories = categories, random_state=0
    )

    newsgroups_testing = fetch_20newsgroups(
        subset = "test", categories=categories, random_state=0
    )

    print(newsgroups_training.data[0])

    # Make the pipeline
    model = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB(),
    )

    # Train the model
    model.fit(newsgroups_training.data, newsgroups_training.target)

    # Run prediction with the testing set
    predicted_targets = model.predict(newsgroups_testing.data)

    # Compute the accuracy
    accuracy = accuracy_score(newsgroups_testing.target, predicted_targets)

    print("Accuracy", accuracy)

    # Serialize the model and the target names
    model_file = "newgroups_model.joblib"
    model_targets_tuple = (model, newsgroups_training.target_names)
    joblib.dump(model_targets_tuple, model_file)

if __name__ == "__main__":
    train_model()



