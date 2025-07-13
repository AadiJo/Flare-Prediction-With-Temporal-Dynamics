import numpy as np
import matplotlib.pyplot as plt
import random
import os

def debug_sample_visualization(
    file_original="processed_solar_data.npz",
    file_boosted="processed_HED_data.npz",
    flare_only=None,  # None = mix, True = flare only, False = quiet only
    num_samples=4,
    save_dir="./plots"  # Directory to save plots
):
    print("Loading datasets...")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    data_orig = np.load(file_original, allow_pickle=True)
    data_boost = np.load(file_boosted, allow_pickle=True)

    X_orig = data_orig['X']
    y = data_orig['y']
    metadata = data_orig['metadata']
    X_boost = data_boost['X']

    flare_indices = [i for i in range(len(y)) if y[i] == 1]
    quiet_indices = [i for i in range(len(y)) if y[i] == 0]

    if flare_only is True:
        selected = random.sample(flare_indices, k=min(num_samples, len(flare_indices)))
    elif flare_only is False:
        selected = random.sample(quiet_indices, k=min(num_samples, len(quiet_indices)))
    else:
        half = num_samples // 2
        selected = (
            random.sample(flare_indices, min(half, len(flare_indices))) +
            random.sample(quiet_indices, min(num_samples - half, len(quiet_indices)))
        )
        random.shuffle(selected)

    for idx in selected:
        sample_norm = X_orig[idx]
        sample_boost = X_boost[idx]
        label = y[idx]
        meta = metadata[idx]

        label_str = "FLARE" if label == 1 else "QUIET"
        case_name = meta['case']
        timestamp = meta['timestamp']
        phase = meta['solar_cycle_phase']

        print(f"\nSample #{idx} | Label: {label_str}")
        print(f"   Case: {case_name}, Timestamp: {timestamp}, Phase: {phase}")

        timestep = 2
        channel_names = ["Bp", "Bt", "Br", "Continuum"]

        for ch in range(4):
            before = sample_norm[timestep, :, :, ch]
            after = sample_boost[timestep, :, :, ch]

            plt.figure(figsize=(10, 4))
            plt.suptitle(f"{label_str} | Case {case_name} | Timestep {timestep} | Channel: {channel_names[ch]}")

            plt.subplot(1, 2, 1)
            plt.title("Original Normalized")
            plt.imshow(before, cmap='inferno', vmin=0, vmax=1)
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.title("After Energy Boost")
            plt.imshow(after, cmap='inferno', vmin=0, vmax=1)
            plt.colorbar()

            plt.tight_layout()
            
            # Save the plot instead of showing it
            filename = f"sample_{idx}_{label_str}_timestep_{timestep}_channel_{channel_names[ch]}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"   Saved: {filename}")
            plt.close()  # Close the figure to free memory

    print(f"\nAll visualizations saved to: {save_dir}")


if __name__ == "__main__":
    debug_sample_visualization(flare_only=None, num_samples=2)
