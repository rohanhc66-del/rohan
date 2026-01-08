# framework/differential_privacy.py
import numpy as np

def clip_gradients(gradients, C):
    norm = np.linalg.norm(gradients)
    return gradients if norm <= C else gradients * (C / norm)

def add_noise(gradients, sigma, C):
    """Add Gaussian noise for (ε, δ)-Differential Privacy."""
    noise = np.random.normal(0, sigma * C, size=gradients.shape)
    return gradients + noise

def apply_differential_privacy(gradients, C=1.0, sigma=0.5):
    clipped = clip_gradients(gradients, C)
    noisy = add_noise(clipped, sigma, C)
    return noisy
