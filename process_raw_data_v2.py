import os
import re
import numpy as np
from datetime import datetime, timedelta
from astropy.io import fits
from skimage.transform import resize
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

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
    """
    Legacy normalization function - use normalize_image_fast for better performance.
    Kept for compatibility.
    """
    return normalize_image_fast(image_data)


def detect_available_segments(data_dir):
    """
    Analyze the dataset to determine which segments are consistently available.
    Returns a list of segment names that appear in most cases.
    OPTIMIZED: Uses caching and early exit to avoid redundant file operations.
    """
    print("Detecting available segments in the dataset...")
    base_path = Path(data_dir)
    
    # Cache file to avoid re-scanning every time
    cache_file = base_path / ".segment_cache"
    
    # Check if cache exists and is recent
    if cache_file.exists():
        try:
            cache_time = cache_file.stat().st_mtime
            # If cache is less than 24 hours old, use it
            if (datetime.now().timestamp() - cache_time) < 86400:
                segments = cache_file.read_text().strip().split(',')
                if segments and segments[0]:  # Valid cache
                    print(f"Using cached segments: {segments}")
                    return segments
        except Exception:
            pass  # Ignore cache errors, will rebuild
    
    # Count occurrences of each segment
    segment_counter = Counter()
    total_timestep_folders = 0
    
    # Sample only first few cases to speed up detection
    case_folders = [f for f in base_path.iterdir() 
                   if f.is_dir() and (f.name.startswith('flare_case_') or f.name.startswith('quiet_case_'))]
    
    # Only check first 10 cases for speed
    sample_cases = case_folders[:min(10, len(case_folders))]
    
    for case_folder in sample_cases:
        timestep_folders = [f for f in case_folder.iterdir() 
                           if f.is_dir() and f.name.startswith('timestep_')]
        
        # Check first timestep only for initial detection
        if timestep_folders:
            timestep_folder = timestep_folders[0]
            total_timestep_folders += 1
            
            # Get all fits files at once
            fits_files = list(timestep_folder.glob("*.fits"))
            
            # Extract segment names more efficiently
            for fits_file in fits_files:
                filename = fits_file.name
                # Quick check for preferred segments first
                for segment in PREFERRED_SEGMENTS:
                    if f".{segment}.fits" in filename:
                        segment_counter[segment] += 1
                        break
    
    # Choose segments that appear in most cases
    if segment_counter:
        # Use segments that appear in all sampled cases
        common_segments = [segment for segment, count in segment_counter.items() 
                          if count >= len(sample_cases)]
        
        if common_segments:
            print(f"Found {len(common_segments)} commonly available segments: {common_segments}")
            # Cache the result
            try:
                cache_file.write_text(','.join(common_segments))
            except Exception:
                pass  # Ignore cache write errors
            return sorted(common_segments)
    
    # Fallback to most common
    if segment_counter:
        most_common = segment_counter.most_common(1)[0][0]
        print(f"Using most common segment: {most_common}")
        return [most_common]
    
    print("No segments found in the dataset!")
    return []


def process_reorganized_data(data_dir="sharp_cnn_lstm_data", max_workers=None):
    """
    OPTIMIZED: Process the reorganized data with parallel processing and memory optimization.
    """
    # Detect available segments (cached)
    available_segments = detect_available_segments(data_dir)
    
    if not available_segments:
        print("No consistent segments found in the dataset. Cannot proceed.")
        return []
    
    CHANNELS = len(available_segments)
    print(f"Processing data using {CHANNELS} segments: {available_segments}")
    
    base_path = Path(data_dir)
    
    # Find all case folders
    case_folders = [folder for folder in base_path.iterdir()
                   if folder.is_dir() and (folder.name.startswith('flare_case_') or folder.name.startswith('quiet_case_'))]
    
    print(f"Found {len(case_folders)} case folders")
    
    # Determine optimal number of workers
    if max_workers is None:
        max_workers = min(mp.cpu_count() - 1, 4)  # Leave one core free, max 4 for memory
    
    print(f"Using {max_workers} parallel workers")
    
    # Process cases in batches to manage memory
    batch_size = max(1, min(20, len(case_folders) // max_workers))  # Adaptive batch size
    processed_data = []
    
    # Process in batches
    for i in range(0, len(case_folders), batch_size):
        batch = case_folders[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(case_folders) + batch_size - 1)//batch_size}")
        
        # Use ProcessPoolExecutor for CPU-bound work with proper initialization
        try:
            with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as executor:
                # Submit all jobs in the batch
                futures = {executor.submit(process_case_folder, str(folder), available_segments): folder 
                          for folder in batch}
                
                # Collect results with progress bar
                for future in tqdm(futures, desc=f"Batch {i//batch_size + 1}", leave=False):
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per case
                        if result is not None:
                            processed_data.append(result)
                    except Exception as e:
                        folder = futures[future]
                        print(f"Failed to process {folder.name}: {e}")
        except Exception as e:
            print(f"Multiprocessing error, falling back to sequential processing: {e}")
            # Fallback to sequential processing
            for folder in tqdm(batch, desc=f"Sequential batch {i//batch_size + 1}"):
                try:
                    result = process_case_folder(str(folder), available_segments)
                    if result is not None:
                        processed_data.append(result)
                except Exception as e:
                    print(f"Failed to process {folder.name}: {e}")
    
    print(f"Successfully processed {len(processed_data)} sequences")
    return processed_data


def create_final_dataset(processed_data):
    """
    OPTIMIZED: Convert processed data into the final format with memory efficiency.
    """
    if not processed_data:
        print("No processed data to convert!")
        return None, None, None

    # Pre-allocate arrays for better memory efficiency
    n_samples = len(processed_data)
    first_sequence = processed_data[0]['sequence']
    
    print(f"Creating final dataset with {n_samples} samples...")
    print(f"Sequence shape: {first_sequence.shape}")
    
    # Pre-allocate with correct dtype
    X_array = np.empty((n_samples, *first_sequence.shape), dtype=np.float32)
    y_array = np.empty(n_samples, dtype=np.int8)  # Use int8 for labels
    metadata = []

    # Fill arrays efficiently
    for i, item in enumerate(tqdm(processed_data, desc="Creating arrays")):
        X_array[i] = item['sequence'].astype(np.float32)
        y_array[i] = item['label']
        metadata.append({
            'case': item['case'],
            'solar_cycle_phase': item['solar_cycle_phase'],
            'timestamp': item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        })

    print(f"Final dataset shapes: X={X_array.shape}, y={y_array.shape}")

    # Verify shapes
    if len(X_array.shape) != 5:
        print(f"Warning: Expected 5D array for X, got shape {X_array.shape}")

    # Print statistics
    unique_values, counts = np.unique(y_array, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_values, counts))}")

    # Print solar cycle phase distribution
    solar_phases = [item['solar_cycle_phase'] for item in metadata]
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
    if year < 1996.4:
        return 'falling'  # Assume falling for very old data
    elif year > 2030.0:
        return 'rising'   # Assume rising for future data
    else:
        # Fallback: if we can't determine, assume falling
        return 'falling'


def process_fits_file(fits_file_path):
    """
    OPTIMIZED: Process a single FITS file efficiently with error handling.
    """
    try:
        with fits.open(fits_file_path) as hdul:
            # Load data directly from HDU
            image_data = hdul[1].data
            
            # Fast NaN/infinity handling
            if np.any(~np.isfinite(image_data)):
                image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Resize with optimized parameters
            image_resized = resize(image_data, (IMAGE_HEIGHT, IMAGE_WIDTH),
                                  anti_aliasing=False, preserve_range=True)
            
            # Optimized normalization
            return normalize_image_fast(image_resized)
            
    except Exception as e:
        print(f"Error processing {fits_file_path}: {e}")
        return None


def normalize_image_fast(image_data):
    """
    OPTIMIZED: Faster normalization using vectorized operations.
    """
    # Use percentile-based normalization for better performance
    min_val = np.percentile(image_data, 1)  # Avoid extreme outliers
    max_val = np.percentile(image_data, 99)
    
    if max_val > min_val:
        normalized = np.clip((image_data - min_val) / (max_val - min_val), 0, 1)
    else:
        normalized = np.zeros_like(image_data, dtype=np.float32)
    
    return normalized.astype(np.float32)  # Use float32 to save memory


def process_timestep_optimized(timestep_folder, available_segments):
    """
    OPTIMIZED: Process a single timestep folder with optimized file handling.
    """
    try:
        # Pre-compile regex for timestamp extraction
        timestamp_pattern = re.compile(r'\.(\d{8}_\d{6})_TAI\.')
        
        # Find segment files more efficiently
        segment_files = {}
        all_files = list(timestep_folder.glob("*.fits"))
        
        # Group files by segment
        for fits_file in all_files:
            filename = fits_file.name
            for segment in available_segments:
                if f".{segment}.fits" in filename:
                    if segment not in segment_files:
                        segment_files[segment] = []
                    segment_files[segment].append(fits_file)
        
        # Check if we have all required segments
        if len(segment_files) != len(available_segments):
            return None
        
        # Get the most recent file for each segment (in case of multiple timestamps)
        final_segment_files = {}
        for segment in available_segments:
            if segment in segment_files:
                # Sort by timestamp and take the most recent
                files = sorted(segment_files[segment], key=lambda x: x.name)
                final_segment_files[segment] = files[-1]
        
        # Extract timestamp from first file
        sample_file = final_segment_files[available_segments[0]]
        timestamp_match = timestamp_pattern.search(sample_file.name)
        if not timestamp_match:
            return None
        
        timestamp_str = timestamp_match.group(1)
        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        
        # Process all segments with parallel processing for I/O bound operations
        with ThreadPoolExecutor(max_workers=min(4, len(available_segments))) as executor:
            futures = {executor.submit(process_fits_file, final_segment_files[segment]): segment 
                      for segment in available_segments}
            
            channels_data = []
            for future in futures:
                result = future.result()
                if result is None:
                    return None
                channels_data.append(result)
        
        # Stack channels
        stacked_image = np.stack(channels_data, axis=-1)
        return (timestamp, stacked_image)
        
    except Exception as e:
        print(f"Error processing timestep {timestep_folder}: {e}")
        return None


def process_case_folder(case_folder_path, available_segments):
    """
    OPTIMIZED: Process a single case folder with all timesteps.
    """
    try:
        case_folder = Path(case_folder_path)
        
        # Determine label
        label = 1 if case_folder.name.startswith('flare_case_') else 0
        
        # Get timestep folders
        timestep_folders = sorted([
            f for f in case_folder.iterdir()
            if f.is_dir() and f.name.startswith('timestep_')
        ])
        
        if len(timestep_folders) != 6:
            return None
        
        # Process timesteps - use threading for I/O bound operations instead of multiprocessing
        timestep_data = []
        for folder in timestep_folders:
            result = process_timestep_optimized(folder, available_segments)
            if result is None:
                return None
            timestep_data.append(result)
        
        # Sort by timestamp
        timestep_data.sort(key=lambda x: x[0])
        
        # Create sequence
        sequence_images = [item[1] for item in timestep_data]
        sequence_array = np.stack(sequence_images)
        
        # Get solar cycle phase
        first_timestamp = timestep_data[0][0]
        solar_cycle_phase = get_solar_cycle_phase(first_timestamp)
        
        return {
            'case': case_folder.name,
            'sequence': sequence_array,
            'label': label,
            'solar_cycle_phase': solar_cycle_phase,
            'timestamp': first_timestamp
        }
        
    except Exception as e:
        print(f"Error processing case {case_folder_path}: {e}")
        return None
    

# Multiprocessing fix: Ensure functions are available to worker processes
def _init_worker():
    """Initialize worker process with necessary imports."""
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)


if __name__ == "__main__":
    import time
    start_time = time.time()
    
    print(f"Using {mp.cpu_count()} CPU cores")
    
    # Process the reorganized data structure with optimization
    processed_data = process_reorganized_data(JSOC_DIR)

    if not processed_data:
        print("No data was processed successfully. Exiting.")
    else:
        # Create the final dataset
        X_final, y_final, metadata = create_final_dataset(processed_data)

        if X_final is not None and X_final.size > 0:
            print(f"Final dataset shapes: X={X_final.shape}, y={y_final.shape}")
            print(f"Metadata entries: {len(metadata)}")
            print(f"Memory usage: {X_final.nbytes / 1024**3:.2f} GB")
            
            # Save with compression for better storage efficiency
            print(f"Saving to '{OUTPUT_FILE}' with compression...")
            np.savez_compressed(OUTPUT_FILE, X=X_final, y=y_final, metadata=metadata)
            print(f"Successfully saved optimized data to '{OUTPUT_FILE}'")

            # Processing time
            elapsed_time = time.time() - start_time
            print(f"Total processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
            print(f"verage time per case: {elapsed_time/len(processed_data):.2f} seconds")

            # Verify the output format
            print(f"\n📊 Output format: [n_samples={X_final.shape[0]}, "
                  f"timesteps={X_final.shape[1]}, "
                  f"height={X_final.shape[2]}, "
                  f"width={X_final.shape[3]}, "
                  f"channels={X_final.shape[4]}]")

            # Show sample metadata
            if metadata:
                print(f"\nSample metadata entries:")
                for i, meta in enumerate(metadata[:3]):
                    print(f"  {i+1}. Case: {meta['case']}, "
                          f"Solar cycle: {meta['solar_cycle_phase']}, "
                          f"Timestamp: {meta['timestamp']}")
                if len(metadata) > 3:
                    print(f"  ... and {len(metadata) - 3} more entries")

            
            # Memory cleanup
            del X_final, y_final
            print("Memory cleaned up successfully")
            
        else:
            print("Could not generate final dataset from processed images.")
    
    print(f"\nProcessing completed in {time.time() - start_time:.2f} seconds")