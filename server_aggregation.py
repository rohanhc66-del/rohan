# framework/server_aggregation.py
def homomorphic_aggregate(encrypted_gradients):
    """Aggregate encrypted gradients homomorphically."""
    agg = encrypted_gradients[0]
    for enc_grad in encrypted_gradients[1:]:
        agg += enc_grad
    agg /= len(encrypted_gradients)
    return agg
