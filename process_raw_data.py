import os
import re
import numpy as np
from datetime import datetime, timedelta
from astropy.io import fits
from skimage.transform import resize
import tensorflow as tf
from tqdm import tqdm


EVENTS_DIR = 'data/2024_events'
JSOC_DIR = 'data/jsoc'
OUTPUT_FILE = 'processed_solar_data.npz'

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
SEQUENCE_LENGTH = 12
HOURS_BEFORE_FLARE = 12
FITS_SEGMENTS = ["field", "azimuth", "inclination", "disambig"]
CHANNELS = len(FITS_SEGMENTS)


def parse_flare_events(events_dir):
    """Parses event files to find M- and X-class flare peak times."""
    flare_times = []
    print(f"Parsing flare event files from: {events_dir}")
    if not os.path.isdir(events_dir):
        raise FileNotFoundError(f"Error: Event directory not found at '{events_dir}'")

    for filename in sorted(os.listdir(events_dir)):
        if not filename.endswith("events.txt"):
            continue
        
        file_path = os.path.join(events_dir, filename)
        file_date = None
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith(':Date:'):
                try:
                    parts = line.split()
                    date_str = f"{parts[1]} {parts[2]} {parts[3]}"
                    file_date = datetime.strptime(date_str, '%Y %m %d')
                except (ValueError, IndexError):
                    file_date = None
                break
        
        if not file_date:
            continue
            
        for line in lines:
            if line.startswith(('#', ':', 'Event')):
                continue

            parts = line.split()
            if len(parts) < 10:
                continue
            
            event_type, flare_class = parts[7], parts[9]
            if event_type == 'XRA' and flare_class.startswith(('M', 'X')):
                max_time_str = parts[3]
                if max_time_str.isdigit() and len(max_time_str) == 4:
                    hour, minute = int(max_time_str[:2]), int(max_time_str[2:])
                    flare_dt = file_date.replace(hour=hour, minute=minute)
                    flare_times.append(flare_dt)

    print(f"Found {len(flare_times)} M/X-class flares.")
    return flare_times


def process_and_label_existing_images(jsoc_dir, flare_times): # uses TensorFlow for GPU acceleration
    """
    Processes all existing FITS directories, labels them, and returns a sorted list.
    """
    processed_data = []
    print(f"Processing existing FITS directories in: {jsoc_dir}")

    if not os.path.isdir(jsoc_dir):
        print(f"Error: JSOC data directory not found at '{jsoc_dir}'")
        return []

    for dirname in tqdm(os.listdir(jsoc_dir), desc="Processing JSOC Folders"):
        dir_path = os.path.join(jsoc_dir, dirname)
        if not os.path.isdir(dir_path):
            continue

        # Parse timestamp from the directory name
        try:
            timestamp_str = dirname.split('.')[2].replace('_TAI', '')
            image_dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        except (IndexError, ValueError):
            print(f"Warning: Could not parse timestamp from directory name: {dirname}")
            continue

        # Label the image: 1 if pre-flare, 0 if not
        label = 0
        for flare_time in flare_times:
            time_diff = flare_time - image_dt
            if timedelta(0) < time_diff <= timedelta(hours=HOURS_BEFORE_FLARE):
                label = 1
                break
        
        # Process the FITS files within the directory
        channels_data = []
        for segment in FITS_SEGMENTS:
            segment_file = next((f for f in os.listdir(dir_path) if f.endswith(f".{segment}.fits")), None)
            if not segment_file:
                channels_data = [] 
                break
            
            with fits.open(os.path.join(dir_path, segment_file)) as hdul:
                # Load data and handle NaNs with NumPy first
                image_data = np.nan_to_num(hdul[1].data)
                
                # 1. Convert numpy array to a TensorFlow tensor
                image_tensor = tf.convert_to_tensor(image_data, dtype=tf.float32)
                
                # 2. Add a 'channels' dimension for resizing (required by tf.image.resize)
                image_tensor = image_tensor[..., tf.newaxis]
                
                # 3. Resize using TensorFlow's GPU stuff
                image_resized = tf.image.resize(image_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH], method='bilinear')
                
                # 4. Remove the extra dimension
                image_resized_squeezed = tf.squeeze(image_resized, axis=-1)
                
                # 5. Normalize
                norm_min = tf.reduce_min(image_resized_squeezed)
                norm_max = tf.reduce_max(image_resized_squeezed)
                if norm_max > norm_min:
                    image_normalized = (image_resized_squeezed - norm_min) / (norm_max - norm_min)
                else:
                    image_normalized = tf.zeros_like(image_resized_squeezed)
                
                # 6. Convert the final tensor back to a NumPy array to be stored
                channels_data.append(image_normalized.numpy())


        if len(channels_data) == CHANNELS:
            stacked_image = np.stack(channels_data, axis=-1)
            processed_data.append({'timestamp': image_dt, 'image': stacked_image, 'label': label})

    # Sort the data by timestamp to ensure chronological order for sequencing
    return sorted(processed_data, key=lambda x: x['timestamp'])


def create_sequences_from_processed_data(processed_data):
    """
    Takes the sorted list of processed images and creates final sequences.
    """
    if len(processed_data) < SEQUENCE_LENGTH:
        print("Error: Not enough processed images to create even one sequence.")
        return None, None

    all_images = np.array([item['image'] for item in processed_data])
    all_labels = np.array([item['label'] for item in processed_data])
    
    print("\nCreating final sequences from processed data...")
    X, y = [], []
    for i in tqdm(range(len(all_images) - SEQUENCE_LENGTH + 1), desc="Creating Sequences"):
        X.append(all_images[i : i + SEQUENCE_LENGTH])
        # The label for a sequence is determined by the label of its LAST image
        y.append(all_labels[i + SEQUENCE_LENGTH - 1])
        
    return np.array(X), np.array(y)


if __name__ == "__main__":
    # 1. Get flare times
    flare_times = parse_flare_events(EVENTS_DIR)
    
    # 2. Process all existing images and label them
    processed_data = process_and_label_existing_images(JSOC_DIR, flare_times)

    if not processed_data:
        print("No data was processed successfully. Exiting.")
    else:
        # 3. Create sequences from the processed data
        X_final, y_final = create_sequences_from_processed_data(processed_data)

        if X_final is not None and X_final.size > 0:
            print(f"\nFinal dataset shapes: X={X_final.shape}, y={y_final.shape}")
            np.savez_compressed(OUTPUT_FILE, X=X_final, y=y_final)
            print(f"Successfully saved processed data to '{OUTPUT_FILE}'")
        else:
            print("Could not generate final dataset from processed images.")