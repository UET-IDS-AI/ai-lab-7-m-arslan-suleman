"""
AIstats_lab.py

Student starter file for:
1. Naive Bayes spam classification
2. K-Nearest Neighbors on Iris
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def accuracy_score(y_true, y_pred):
    """
    Compute classification accuracy.
    """
    return float(np.mean(y_true == y_pred))


# =========================
# Q1 Naive Bayes
# =========================

def naive_bayes_mle_spam():
    """
    Implement Naive Bayes spam classification using simple MLE.
    """

    texts = [
        "win money now",
        "limited offer win cash",
        "cheap meds available",
        "win big prize now",
        "exclusive offer buy now",
        "cheap pills buy cheap meds",
        "win lottery claim prize",
        "urgent offer win money",
        "free cash bonus now",
        "buy meds online cheap",
        "meeting schedule tomorrow",
        "project discussion meeting",
        "please review the report",
        "team meeting agenda today",
        "project deadline discussion",
        "review the project document",
        "schedule a meeting tomorrow",
        "please send the report",
        "discussion on project update",
        "team sync meeting notes"
    ]

    labels = np.array([
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0
    ])

    test_email = "win cash prize now"

    # Tokenize
    tokenized = [text.split() for text in texts]

    # Vocabulary
    vocab = set(word for doc in tokenized for word in doc)

    # Priors
    priors = {
        0: np.mean(labels == 0),
        1: np.mean(labels == 1)
    }

    # Word counts
    word_counts = {0: {}, 1: {}}
    total_words = {0: 0, 1: 0}

    for c in [0, 1]:
        word_counts[c] = {word: 0 for word in vocab}

    for doc, label in zip(tokenized, labels):
        for word in doc:
            word_counts[label][word] += 1
            total_words[label] += 1

    # Word probabilities (MLE)
    word_probs = {0: {}, 1: {}}
    for c in [0, 1]:
        for word in vocab:
            word_probs[c][word] = word_counts[c][word] / total_words[c]

    # Prediction
    test_words = test_email.split()
    scores = {}

    for c in [0, 1]:
        score = np.log(priors[c])
        for word in test_words:
            prob = word_probs[c].get(word, 0)
            if prob > 0:
                score += np.log(prob)
            else:
                score += -1e9  # simulate log(0)
        scores[c] = score

    prediction = max(scores, key=scores.get)

    return priors, word_probs, prediction


# =========================
# Q2 KNN
# =========================

def knn_iris(k=3, test_size=0.2, seed=0):
    """
    Implement KNN from scratch on the Iris dataset.
    """

    # Load dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Euclidean distance
    def euclidean(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # Prediction function
    def predict(x):
        distances = [euclidean(x, x_train) for x_train in X_train]
        k_indices = np.argsort(distances)[:k]
        k_labels = y_train[k_indices]
        values, counts = np.unique(k_labels, return_counts=True)
        return values[np.argmax(counts)]

    # Predictions
    train_preds = np.array([predict(x) for x in X_train])
    test_preds = np.array([predict(x) for x in X_test])

    # Accuracy
    train_accuracy = accuracy_score(y_train, train_preds)
    test_accuracy = accuracy_score(y_test, test_preds)

    return train_accuracy, test_accuracy, test_preds
