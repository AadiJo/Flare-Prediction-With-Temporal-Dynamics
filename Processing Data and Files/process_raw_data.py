import os
import re
import numpy as np
from datetime import datetime, timedelta
from astropy.io import fits
from skimage.transform import resize
from pathlib import Path
from tqdm import tqdm
from collections import Counter

JSOC_DIR = '../HSRA/async_sharp'
OUTPUT_FILE = 'processed_solar_data.npz'

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
SEQUENCE_LENGTH = 6

# SHARP scalar keywords to extract
SHARP_SCALAR_KEYWORDS = ["USFLUX", "MEANSHR", "EPSZ", "TOTUSJZ", "SHRGT45", "TOTPOT"]

PREFERRED_SEGMENTS = ["Bp", "Bt", "Br", "continuum"]


def normalize_image(image_data):
    image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    if max_val > min_val:
        normalized = (image_data - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(image_data)
    return normalized


def detect_available_segments(data_dir):
    print("Detecting available segments in the dataset...")
    base_path = Path(data_dir)
    segment_counter = Counter()
    total_timestep_folders = 0

    for case_folder in base_path.iterdir():
        if not (case_folder.is_dir() and
                (case_folder.name.startswith('flare_case_') or case_folder.name.startswith('quiet_case_'))):
            continue
        for timestep_folder in case_folder.iterdir():
            if not (timestep_folder.is_dir() and timestep_folder.name.startswith('timestep_')):
                continue
            total_timestep_folders += 1
            fits_files = list(timestep_folder.glob("*.fits"))
            for fits_file in fits_files:
                for segment in PREFERRED_SEGMENTS:
                    if segment in fits_file.name:
                        segment_counter[segment] += 1
                        break
                else:
                    match = re.search(r'\.([^.]+)\.fits$', fits_file.name)
                    if match:
                        segment = match.group(1)
                        segment_counter[segment] += 1

    threshold = max(1, total_timestep_folders * 0.5)
    common_segments = [segment for segment, count in segment_counter.items() if count >= threshold]
    if common_segments:
        print(f"Found {len(common_segments)} commonly available segments: {common_segments}")
        return sorted(common_segments)
    if segment_counter:
        most_common = segment_counter.most_common(1)[0][0]
        print(f"No consistently available segments found. Using most common: {most_common}")
        return [most_common]

    print("No segments found in the dataset!")
    return []


def process_reorganized_data(data_dir="sharp_cnn_lstm_data"):
    available_segments = detect_available_segments(data_dir)
    if not available_segments:
        print("No consistent segments found in the dataset. Cannot proceed.")
        return []

    CHANNELS = len(available_segments)
    print(f"Processing data using {CHANNELS} segments: {available_segments}")

    processed_data = []
    base_path = Path(data_dir)

    case_folders = [f for f in base_path.iterdir()
                    if f.is_dir() and (f.name.startswith('flare_case_') or f.name.startswith('quiet_case_'))]

    print(f"Found {len(case_folders)} case folders")

    for case_folder in tqdm(case_folders, desc="Processing case folders"):
        label = 1 if case_folder.name.startswith('flare_case_') else 0
        timestep_folders = sorted([
            f for f in case_folder.iterdir()
            if f.is_dir() and f.name.startswith('timestep_')
        ])
        if len(timestep_folders) != 6:
            print(f"Warning: {case_folder.name} has {len(timestep_folders)} timesteps, expected 6. Skipping.")
            continue

        timestep_data = []
        timestamps = []

        for timestep_folder in timestep_folders:
            segment_files = {}
            for segment in available_segments:
                candidates = list(timestep_folder.glob(f"*{segment}*.fits"))
                if candidates:
                    segment_files[segment] = candidates[0]
                else:
                    break
            if len(segment_files) != len(available_segments):
                print(f"  Warning: Missing segments in {timestep_folder}. Found {len(segment_files)}/{len(available_segments)}.")
                found_segments = list(segment_files.keys())
                print(f"  Found segments: {found_segments}")
                all_fits_files = list(timestep_folder.glob("*.fits"))
                if all_fits_files:
                    print(f"  Available files: {[f.name for f in all_fits_files[:5]]}" +
                          (f" and {len(all_fits_files)-5} more" if len(all_fits_files) > 5 else ""))
                else:
                    print(f"  No FITS files found in {timestep_folder}")
                continue

            sample_file = list(segment_files.values())[0]
            timestamp_match = re.search(r'\.(\d{8}_\d{6})_TAI\.', sample_file.name)
            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                timestamps.append(timestamp)
            else:
                print(f"  Warning: Could not extract timestamp from {sample_file.name}")
                continue

            keyword_values = {}
            try:
                with fits.open(sample_file) as hdul:
                    header = hdul[1].header
                    for key in SHARP_SCALAR_KEYWORDS:
                        keyword_values[key] = header.get(key, None)
            except Exception as e:
                print(f"  Warning: Could not read keywords from {sample_file}: {e}")
                continue

            channels_data = []
            for segment in available_segments:
                fits_file = segment_files[segment]
                try:
                    with fits.open(fits_file) as hdul:
                        image_data = hdul[1].data
                        image_data = np.nan_to_num(image_data)
                        image_resized = resize(image_data, (IMAGE_HEIGHT, IMAGE_WIDTH),
                                               anti_aliasing=True, preserve_range=True)
                        image_normalized = normalize_image(image_resized)
                        channels_data.append(image_normalized)
                except Exception as e:
                    print(f"  Error processing {fits_file}: {e}")
                    channels_data = []
                    break

            if len(channels_data) == CHANNELS:
                stacked_image = np.stack(channels_data, axis=-1)
                timestep_data.append((timestamp, stacked_image, keyword_values))

        if len(timestep_data) == len(timestep_folders):
            timestep_data.sort(key=lambda x: x[0])
            sequence_images = [item[1] for item in timestep_data]
            sequence_keywords = [item[2] for item in timestep_data]
            sequence_array = np.stack(sequence_images)
            first_timestamp = timestep_data[0][0]
            solar_cycle_phase = get_solar_cycle_phase(first_timestamp)

            processed_data.append({
                'case': case_folder.name,
                'sequence': sequence_array,
                'label': label,
                'solar_cycle_phase': solar_cycle_phase,
                'timestamp': first_timestamp,
                'scalar_keywords': sequence_keywords
            })
            print(f"  ✓ Processed {case_folder.name}: {sequence_array.shape}, label={label}, solar_cycle={solar_cycle_phase}")
        else:
            print(f"  ✗ Skipped {case_folder.name}: only {len(timestep_data)}/{len(timestep_folders)} valid timesteps")

    print(f"Successfully processed {len(processed_data)} sequences")
    return processed_data


def create_final_dataset(processed_data):
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
            'timestamp': item['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'scalar_keywords': item['scalar_keywords']
        })

    X_array = np.array(X)
    y_array = np.array(y)

    print(f"Final dataset shapes: X={X_array.shape}, y={y_array.shape}")

    if len(X_array.shape) != 5:
        print(f"Warning: Expected 5D array for X, got shape {X_array.shape}")

    unique_values, counts = np.unique(y_array, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_values, counts))}")

    solar_phases = [item['solar_cycle_phase'] for item in metadata]
    phase_counts = Counter(solar_phases)
    print(f"Solar cycle phase distribution: {dict(phase_counts)}")

    return X_array, y_array, metadata


def get_solar_cycle_phase(timestamp):
    year = timestamp.year + (timestamp.month - 1) / 12.0
    solar_cycles = [
        {'cycle': 23, 'min_start': 1996.4, 'max': 2000.3, 'min_end': 2008.9},
        {'cycle': 24, 'min_start': 2008.9, 'max': 2014.3, 'min_end': 2019.8},
        {'cycle': 25, 'min_start': 2019.8, 'max': 2025.7, 'min_end': 2030.0},
    ]
    for cycle in solar_cycles:
        if cycle['min_start'] <= year <= cycle['min_end']:
            if year <= cycle['max']:
                return 'rising'
            else:
                return 'falling'
    if year < 1996.4:
        return 'falling'
    elif year > 2030.0:
        return 'rising'
    else:
        return 'falling'


if __name__ == "__main__":
    processed_data = process_reorganized_data(JSOC_DIR)

    if not processed_data:
        print("No data was processed successfully. Exiting.")
    else:
        X_final, y_final, metadata = create_final_dataset(processed_data)

        if X_final is not None and X_final.size > 0:
            print(f"\nFinal dataset shapes: X={X_final.shape}, y={y_final.shape}")
            print(f"Metadata entries: {len(metadata)}")
            np.savez_compressed(OUTPUT_FILE, X=X_final, y=y_final, metadata=metadata)
            print(f"Successfully saved processed data with metadata to '{OUTPUT_FILE}'")
            print(f"Output format: [n_samples={X_final.shape[0]}, "
                  f"timesteps={X_final.shape[1]}, "
                  f"height={X_final.shape[2]}, "
                  f"width={X_final.shape[3]}, "
                  f"channels={X_final.shape[4]}]")

            if metadata:
                print(f"\nSample metadata entries:")
                for i, meta in enumerate(metadata[:3]):
                    print(f"  {i+1}. Case: {meta['case']}, "
                          f"Solar cycle: {meta['solar_cycle_phase']}, "
                          f"Timestamp: {meta['timestamp']}")
                if len(metadata) > 3:
                    print(f"  ... and {len(metadata) - 3} more entries")
            if X_final.shape[0] < 10:
                print("⚠️ Warning: Very small dataset! Consider adding more data.")
        else:
            print("Could not generate final dataset from processed images.")
