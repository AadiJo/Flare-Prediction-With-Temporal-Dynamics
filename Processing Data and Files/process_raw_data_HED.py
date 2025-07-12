import numpy as np

def compute_HED_mask(Bp, Bt, Br, threshold=2e4):
    """Compute HED region mask using Gauss-scale magnetic fields."""
    B_total = np.sqrt(Bp**2 + Bt**2 + Br**2)
    B_potential = np.abs(Br)  # Br-only approximation
    rho_free = (B_total**2 - B_potential**2) / (8 * np.pi)  # in erg/cm^3
    return rho_free >= threshold

def compute_HED_masked_normalized(seq_norm, seq_orig):
    """Apply HED mask (based on Gauss-scale) to normalized data."""
    Bp, Bt, Br = seq_orig[..., 0], seq_orig[..., 1], seq_orig[..., 2]
    mask_seq = np.array([compute_HED_mask(Bp[t], Bt[t], Br[t]) for t in range(seq_orig.shape[0])])

    # Apply mask to normalized channels (all 4)
    return seq_norm * mask_seq[..., np.newaxis]

def convert_to_HED_masked(input_file='processed_solar_data.npz', output_file='processed_HED_data.npz'):
    print(f"Loading {input_file}...")
    data = np.load(input_file, allow_pickle=True)
    X_norm = data['X']               # normalized input (0–1)
    X_orig = data['X_Original']      # Gauss-scale magnetic fields
    y = data['y']
    metadata = data['metadata']
    
    assert X_norm.shape == X_orig.shape, "X and X_Original shapes do not match!"

    print("Computing HED-masked sequences...")
    X_HED = np.array([
        compute_HED_masked_normalized(X_norm[i], X_orig[i])
        for i in range(len(X_norm))
    ], dtype=np.float32)

    print(f"Saving HED-masked dataset to {output_file}...")
    np.savez_compressed(output_file, X=X_HED, y=y, metadata=metadata)
    print("Done")

if __name__ == "__main__":
    convert_to_HED_masked()
