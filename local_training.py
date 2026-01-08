# framework/local_training.py
import numpy as np
from sklearn.linear_model import LogisticRegression

def local_train(X, y, model=None):
    """Simulate local training on client data."""
    if model is None:
        model = LogisticRegression(max_iter=100)
    model.fit(X, y)
    gradients = model.coef_.flatten()
    return gradients, model
