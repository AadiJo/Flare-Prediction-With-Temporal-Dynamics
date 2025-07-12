# Script to export saliency maps for the 10 most recent data entries using the CNNVisualizationTool
import numpy as np
from visualize_cnn_attention import CNNVisualizationTool, load_sample_data
import os

# Load models and data
viz_tool = CNNVisualizationTool()
X, y = load_sample_data()

if X is None or y is None:
    print("Failed to load data.")
    exit(1)

# Get the 10 most recent entries (assuming last 10 in X)
recent_indices = list(range(len(X) - 10, len(X)))

# Output directory for saliency maps
output_dir = "saliency_exports"
os.makedirs(output_dir, exist_ok=True)

for idx in recent_indices:
    for channel_name in viz_tool.channel_names:
        model = viz_tool.models[channel_name]
        channel_idx = viz_tool.channel_names.index(channel_name)
        channel_data = X[idx][:, :, :, channel_idx:channel_idx+1]
        channel_data_batch = np.expand_dims(channel_data, axis=0)
        channel_data_tensor = channel_data_batch
        # Compute saliency map
        saliency = viz_tool.create_saliency_map(model, channel_data_tensor)
        # Save each timestep as an image (if 3D)
        if saliency is not None:
            if len(saliency.shape) == 3:
                for t in range(saliency.shape[0]):
                    out_path = os.path.join(output_dir, f"saliency_{channel_name}_sample_{idx}_t{t}.png")
                    import matplotlib.pyplot as plt
                    plt.imsave(out_path, saliency[t], cmap='hot')
            else:
                out_path = os.path.join(output_dir, f"saliency_{channel_name}_sample_{idx}.png")
                import matplotlib.pyplot as plt
                plt.imsave(out_path, saliency, cmap='hot')
        print(f"Exported saliency for sample {idx}, channel {channel_name}")
print("Done exporting saliency maps.")
