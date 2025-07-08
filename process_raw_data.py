import os
import re
import numpy as np
from datetime import datetime, timedelta
from astropy.io import fits
from skimage.transform import resize
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# Constants for data processing
JSOC_DIR = 'sharp_cnn_lstm_data'
OUTPUT_FILE = 'processed_solar_data.npz'

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
SEQUENCE_LENGTH = 6  # Changed to match your 6 timesteps

rising = 0
falling = 0

# Instead of hardcoding segments, we'll detect available ones
# These are preferred if available
PREFERRED_SEGMENTS = ["Bp", "Bt", "Br", "continuum"]


def normalize_image(image_data):
    """Normalize image to [0, 1] range with proper NaN handling"""
    # Replace infinities and handle NaNs
    image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Get min and max for normalization
    min_val = np.min(image_data)
    max_val = np.max(image_data)

    # Normalize to [0, 1]
    if max_val > min_val:
        normalized = (image_data - min_val) / (max_val - min_val)
    else:
        # Handle constant value case
        normalized = np.zeros_like(image_data)

    return normalized


def detect_available_segments(data_dir):
    """
    Analyze the dataset to determine which segments are consistently available.
    Returns a list of segment names that appear in most cases.
    """
    print("Detecting available segments in the dataset...")
    base_path = Path(data_dir)

    # Count occurrences of each segment
    segment_counter = Counter()
    total_timestep_folders = 0

    # Check what segments are available
    for case_folder in base_path.iterdir():
        if not (case_folder.is_dir() and
                (case_folder.name.startswith('flare_case_') or case_folder.name.startswith('quiet_case_'))):
            continue

        for timestep_folder in case_folder.iterdir():
            if not (timestep_folder.is_dir() and timestep_folder.name.startswith('timestep_')):
                continue

            total_timestep_folders += 1

            # Find all fits files
            fits_files = list(timestep_folder.glob("*.fits"))

            # Extract segment names
            for fits_file in fits_files:
                # Try to extract segment name with more flexible pattern
                for segment in PREFERRED_SEGMENTS:
                    if segment in fits_file.name:
                        segment_counter[segment] += 1
                        break
                else:
                    # If none of the preferred segments match, try to extract any segment
                    match = re.search(r'\.([^.]+)\.fits$', fits_file.name)
                    if match:
                        segment = match.group(1)
                        segment_counter[segment] += 1

    # Choose segments that appear in at least 50% of timestep folders
    threshold = max(1, total_timestep_folders * 0.5)  # At least present in 50% of folders
    common_segments = [segment for segment, count in segment_counter.items() if count >= threshold]

    # If we have at least one common segment, use it
    if common_segments:
        print(f"Found {len(common_segments)} commonly available segments: {common_segments}")
        return sorted(common_segments)  # Sort for consistency

    # If no common segments, use whatever is most common
    if segment_counter:
        most_common = segment_counter.most_common(1)[0][0]
        print(f"No consistently available segments found. Using most common: {most_common}")
        return [most_common]

    print("No segments found in the dataset!")
    return []


def process_reorganized_data(data_dir="sharp_cnn_lstm_data"):
    """
    Process the reorganized data from case folders with timestep directories.
    This works with data organized by reorganize_files.py
    """
    # First detect what segments are available in the dataset
    available_segments = detect_available_segments(data_dir)

    if not available_segments:
        print("No consistent segments found in the dataset. Cannot proceed.")
        return []

    CHANNELS = len(available_segments)
    print(f"Processing data using {CHANNELS} segments: {available_segments}")

    processed_data = []
    base_path = Path(data_dir)

    # Find all case folders
    case_folders = []
    for folder in base_path.iterdir():
        if folder.is_dir() and (folder.name.startswith('flare_case_') or folder.name.startswith('quiet_case_')):
            case_folders.append(folder)

    print(f"Found {len(case_folders)} case folders")

    for case_folder in tqdm(case_folders, desc="Processing case folders"):
        # Determine label based on folder name
        label = 1 if case_folder.name.startswith('flare_case_') else 0

        # Get all timestep folders
        timestep_folders = sorted([
            f for f in case_folder.iterdir()
            if f.is_dir() and f.name.startswith('timestep_')
        ])

        if len(timestep_folders) != 6:  # We expect exactly 6 timesteps
            print(f"Warning: {case_folder.name} has {len(timestep_folders)} timesteps, expected 6. Skipping.")
            continue

        # Process each timestep
        timestep_data = []
        timestamps = []

        for timestep_folder in timestep_folders:
            # Find all FITS files for the segments we're using
            segment_files = {}
            for segment in available_segments:
                # Find any file containing this segment name
                candidates = list(timestep_folder.glob(f"*{segment}*.fits"))
                if candidates:
                    segment_files[segment] = candidates[0]  # Use the first matching file
                else:
                    break  # Missing a required segment

            # Skip this timestep if we're missing any segments
            if len(segment_files) != len(available_segments):
                print(f"  Warning: Missing segments in {timestep_folder}. Found {len(segment_files)}/{len(available_segments)}.")

                # Debug info to understand what segments we're finding
                found_segments = list(segment_files.keys())
                print(f"  Found segments: {found_segments}")

                # Show what files are actually in this directory
                all_fits_files = list(timestep_folder.glob("*.fits"))
                if all_fits_files:
                    print(f"  Available files: {[f.name for f in all_fits_files[:5]]}" +
                          (f" and {len(all_fits_files)-5} more" if len(all_fits_files) > 5 else ""))
                else:
                    print(f"  No FITS files found in {timestep_folder}")

                continue

            # Extract timestamp from one of the files
            sample_file = list(segment_files.values())[0]
            timestamp_match = re.search(r'\.(\d{8}_\d{6})_TAI\.', sample_file.name)
            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                timestamps.append(timestamp)
            else:
                print(f"  Warning: Could not extract timestamp from {sample_file.name}")
                continue

            # Process all segments
            channels_data = []
            for segment in available_segments:
                fits_file = segment_files[segment]

                try:
                    with fits.open(fits_file) as hdul:
                        # Load data
                        image_data = hdul[1].data

                        # Handle NaNs and infinities
                        image_data = np.nan_to_num(image_data)

                        # Resize using scikit-image
                        image_resized = resize(image_data, (IMAGE_HEIGHT, IMAGE_WIDTH),
                                              anti_aliasing=True, preserve_range=True)

                        # Normalize to [0,1]
                        image_normalized = normalize_image(image_resized)

                        # Add to channel data
                        channels_data.append(image_normalized)
                except Exception as e:
                    print(f"  Error processing {fits_file}: {e}")
                    channels_data = []
                    break

            # If all segments were processed successfully
            if len(channels_data) == CHANNELS:
                stacked_image = np.stack(channels_data, axis=-1)
                timestep_data.append((timestamp, stacked_image))

        # Process this case only if we have data for all timesteps
        if len(timestep_data) == len(timestep_folders):
            # Sort timesteps chronologically
            timestep_data.sort(key=lambda x: x[0])

            # Create sequence for this case
            sequence_images = [item[1] for item in timestep_data]
            sequence_array = np.stack(sequence_images)

            # Determine solar cycle phase based on the first timestamp
            first_timestamp = timestep_data[0][0]
            solar_cycle_phase = get_solar_cycle_phase(first_timestamp)

            # Add the processed sequence to our dataset
            processed_data.append({
                'case': case_folder.name,
                'sequence': sequence_array,
                'label': label,
                'solar_cycle_phase': solar_cycle_phase,
                'timestamp': first_timestamp
            })
            print(f"  ✓ Processed {case_folder.name}: {sequence_array.shape}, label={label}, solar_cycle={solar_cycle_phase}")
        else:
            print(f"  ✗ Skipped {case_folder.name}: only {len(timestep_data)}/{len(timestep_folders)} valid timesteps")

    print(f"Successfully processed {len(processed_data)} sequences")
    return processed_data


def create_final_dataset(processed_data):
    """
    Convert processed data into the final format needed for the CNN-LSTM model.
    """
    if not processed_data:
        print("No processed data to convert!")
        return None, None, None

    X = []
    y = []
    metadata = []

    for item in processed_data:
        X.append(item['sequence'])
        y.append(item['label'])
        metadata.append({
            'case': item['case'],
            'solar_cycle_phase': item['solar_cycle_phase'],
            'timestamp': item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        })

    X_array = np.array(X)
    y_array = np.array(y)

    print(f"Final dataset shapes: X={X_array.shape}, y={y_array.shape}")

    # Verify that the shapes match what the model expects
    if len(X_array.shape) != 5:
        print(f"Warning: Expected 5D array for X, got shape {X_array.shape}")

    # Print class distribution
    unique_values, counts = np.unique(y_array, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_values, counts))}")

    # Print solar cycle phase distribution
    solar_phases = [item['solar_cycle_phase'] for item in metadata]
    from collections import Counter
    phase_counts = Counter(solar_phases)
    print(f"Solar cycle phase distribution: {dict(phase_counts)}")

    return X_array, y_array, metadata


def get_solar_cycle_phase(timestamp):
    """
    Determine if the given timestamp is in the rising or falling phase of the solar cycle.
    Based on historical solar cycle data.

    Solar Cycle minima and maxima (approximate dates):
    - Cycle 23: Min ~1996.4, Max ~2000.3, Min ~2008.9
    - Cycle 24: Min ~2008.9, Max ~2014.3, Min ~2019.8
    - Cycle 25: Min ~2019.8, Max ~2025.7 (predicted), Min ~2030 (predicted)

    Args:
        timestamp (datetime): The timestamp to check

    Returns:
        str: 'rising' or 'falling'
    """
    year = timestamp.year + (timestamp.month - 1) / 12.0  # Convert to decimal year

    # Define solar cycle periods with their minima and maxima
    solar_cycles = [
        {'cycle': 23, 'min_start': 1996.4, 'max': 2000.3, 'min_end': 2008.9},
        {'cycle': 24, 'min_start': 2008.9, 'max': 2014.3, 'min_end': 2019.8},
        {'cycle': 25, 'min_start': 2019.8, 'max': 2025.7, 'min_end': 2030.0},  # Predicted
    ]

    # Find which solar cycle the timestamp belongs to
    for cycle in solar_cycles:
        if cycle['min_start'] <= year <= cycle['min_end']:
            # Check if we're in rising or falling phase
            if year <= cycle['max']:
                return 'rising'
            else:
                return 'falling'

    # If outside defined cycles, make a reasonable assumption
    # Most data in the dataset appears to be from 2011, which is cycle 24 falling phase
    if year < 1996.4:
        return 'falling'  # Assume falling for very old data
    elif year > 2030.0:
        return 'rising'   # Assume rising for future data
    else:
        # Fallback: if we can't determine, assume falling
        return 'falling'


if __name__ == "__main__":
    # Process the reorganized data structure
    processed_data = process_reorganized_data(JSOC_DIR)

    if not processed_data:
        print("No data was processed successfully. Exiting.")
    else:
        # Create the final dataset
        X_final, y_final, metadata = create_final_dataset(processed_data)

        if X_final is not None and X_final.size > 0:
            print(f"\nFinal dataset shapes: X={X_final.shape}, y={y_final.shape}")
            print(f"Metadata entries: {len(metadata)}")
            np.savez_compressed(OUTPUT_FILE, X=X_final, y=y_final, metadata=metadata)
            print(f"Successfully saved processed data with metadata to '{OUTPUT_FILE}'")

            # Verify the output format matches what the model expects
            # The shape should be [n_samples, n_timesteps, height, width, channels]
            print(f"Output format: [n_samples={X_final.shape[0]}, "
                  f"timesteps={X_final.shape[1]}, "
                  f"height={X_final.shape[2]}, "
                  f"width={X_final.shape[3]}, "
                  f"channels={X_final.shape[4]}]")

            # Show sample metadata entries
            if metadata:
                print(f"\nSample metadata entries:")
                for i, meta in enumerate(metadata[:3]):  # Show first 3 entries
                    print(f"  {i+1}. Case: {meta['case']}, "
                          f"Solar cycle: {meta['solar_cycle_phase']}, "
                          f"Timestamp: {meta['timestamp']}")
                if len(metadata) > 3:
                    print(f"  ... and {len(metadata) - 3} more entries")

            # Make sure we have enough samples to train a model
            if X_final.shape[0] < 10:
                print("⚠️ Warning: Very small dataset! Consider adding more data.")
        else:
            print("Could not generate final dataset from processed images.")