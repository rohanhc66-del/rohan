# main.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ---------- Framework Modules ----------
# Make sure these are inside 'framework/' folder
# 1. local_training.py
def local_train(X, y, model=None):
    from sklearn.linear_model import LogisticRegression
    if model is None:
        model = LogisticRegression(max_iter=100)
    model.fit(X, y)
    gradients = model.coef_.flatten()
    return gradients, model

# 2. differential_privacy.py
def clip_gradients(gradients, C):
    norm = np.linalg.norm(gradients)
    return gradients if norm <= C else gradients * (C / norm)

def add_noise(gradients, sigma, C):
    noise = np.random.normal(0, sigma * C, size=gradients.shape)
    return gradients + noise

def apply_differential_privacy(gradients, C=1.0, sigma=0.5):
    clipped = clip_gradients(gradients, C)
    noisy = add_noise(clipped, sigma, C)
    return noisy

# 3. homo_encryption.py
import tenseal as ts

def create_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2**40
    return context

def encrypt_gradients(context, gradients):
    return ts.ckks_vector(context, gradients)

def decrypt_vector(context, encrypted_vector):
    return encrypted_vector.decrypt()

# 4. server_aggregation.py
def homomorphic_aggregate(encrypted_gradients):
    agg = encrypted_gradients[0]
    for enc_grad in encrypted_gradients[1:]:
        agg += enc_grad
    N = len(encrypted_gradients)
    agg *= (1.0 / N)  # Multiply by reciprocal instead of dividing
    return agg

# ---------- Main Execution ----------
# 1️⃣ Load UCI Adult Dataset
data = pd.read_csv("dataset/adult.csv")

# Preprocess categorical columns
for col in data.select_dtypes(include='object'):
    data[col] = LabelEncoder().fit_transform(data[col])

# Split features and labels
X = data.drop('income', axis=1)
y = data['income']

# Standardize features
X = StandardScaler().fit_transform(X)

# Split data among clients
clients = 5
split_size = len(X) // clients
client_data = [(X[i*split_size:(i+1)*split_size], y[i*split_size:(i+1)*split_size]) for i in range(clients)]

# 2️⃣ Local training + Differential Privacy
local_gradients = []
for i, (Xi, yi) in enumerate(client_data):
    gradients, _ = local_train(Xi, yi)
    dp_grad = apply_differential_privacy(gradients)
    local_gradients.append(dp_grad)
    print(f"Client {i+1}: Gradient (DP applied, first 5 elements): {dp_grad[:5]}")

# 3️⃣ Encrypt gradients using FHE
context = create_context()
encrypted_gradients = [encrypt_gradients(context, grad) for grad in local_gradients]
print("\nAll gradients encrypted successfully!")

# 4️⃣ Server-side homomorphic aggregation
agg_encrypted = homomorphic_aggregate(encrypted_gradients)
print("Server aggregation completed on encrypted data.")

# 5️⃣ Trusted decryption
agg_decrypted = decrypt_vector(context, agg_encrypted)
print("\nDecrypted aggregated gradient (first 5 elements):", agg_decrypted[:5])

# 6️⃣ Update model weights (simple gradient descent step)
eta = 0.1
model_weights = np.zeros_like(agg_decrypted)
new_weights = model_weights - eta * np.array(agg_decrypted)
print("\nUpdated model weights (first 5 elements):", new_weights[:5])
