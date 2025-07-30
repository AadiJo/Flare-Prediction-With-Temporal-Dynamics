import numpy as np

def compute_HED_mask(Bp, Bt, Br, threshold=1e4):
    """Compute binary HED mask using lowered threshold."""
    B_total2 = Bp**2 + Bt**2 + Br**2
    B_potential2 = Br**2
    rho_free = (B_total2 - B_potential2) / (8 * np.pi)
    return rho_free >= threshold  # boolean mask

def apply_HED_mask(seq_norm, seq_orig, threshold=1e4):
    """Apply binary HED mask uniformly to all 4 channels."""
    Bp, Bt, Br = seq_orig[..., 0], seq_orig[..., 1], seq_orig[..., 2]
    masked_seq = np.zeros_like(seq_norm, dtype=np.float32)
    
    for t in range(seq_norm.shape[0]):
        mask = compute_HED_mask(Bp[t], Bt[t], Br[t], threshold=threshold)
        masked_seq[t] = seq_norm[t] * mask[..., np.newaxis]  # broadcast across 4 channels

    return masked_seq

def convert_to_HED_masked(input_file='all_data.npz',
                          output_file='processed_HED_data.npz',
                          threshold=1e4):
    print(f"Loading {input_file}...")
    data = np.load(input_file, allow_pickle=True)
    X_norm = data['X']
    X_orig = data['X_original']

    assert X_norm.shape == X_orig.shape, "Mismatch between X and X_original shapes."

    print("Applying binary HED masking...")
    X_masked = np.array([
        apply_HED_mask(X_norm[i], X_orig[i], threshold=threshold)
        for i in range(len(X_norm))
    ], dtype=np.float32)

    print(f"Saving masked dataset to {output_file}...")
    # Save all original keys except 'X', replace with X_masked
    save_dict = {key: data[key] for key in data.files if key != 'X'}
    save_dict['X'] = X_masked
    np.savez_compressed(output_file, **save_dict)
    print("Done.")

if __name__ == "__main__":
    convert_to_HED_masked()
