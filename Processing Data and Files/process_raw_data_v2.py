import os, re, numpy as np, multiprocessing as mp, warnings
from datetime import datetime, timedelta
from astropy.io import fits
from skimage.transform import resize
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Constants
<<<<<<< HEAD
JSOC_DIR = "C:\\Users\\suraj\\OneDrive\\Desktop\\CodingProjects\\HSRA\\async_sharp"
=======
JSOC_DIR = "../HSRA/async_sharp"
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
OUTPUT_FILE = 'processed_solar_data.npz'
COMBINE_EXTERNAL_NPZ = True
EXTERNAL_NPZ_FILE = 'processed_solar_data_end.npz' 
IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH = 64, 64, 6
PREFERRED_SEGMENTS = ["Bp", "Bt", "Br", "continuum"]

def normalize_image(image_data):
    return normalize_image_fast(image_data)

def detect_available_segments(data_dir):
    print("Detecting available segments in the dataset...")
    base_path, cache_file = Path(data_dir), Path(data_dir) / ".segment_cache"
    if cache_file.exists():
        try:
            if (datetime.now().timestamp() - cache_file.stat().st_mtime) < 86400:
                segments = cache_file.read_text().strip().split(',')
                if segments and segments[0]:
                    print(f"Using cached segments: {segments}")
                    return segments
        except Exception: pass
    
    segment_counter = Counter()
    case_folders = [f for f in base_path.iterdir() if f.is_dir() and (f.name.startswith('flare_case_') or f.name.startswith('quiet_case_'))]
    sample_cases = case_folders[:min(10, len(case_folders))]
    
    for case_folder in sample_cases:
        timestep_folders = [f for f in case_folder.iterdir() if f.is_dir() and f.name.startswith('timestep_')]
        if timestep_folders:
            for fits_file in list(timestep_folders[0].glob("*.fits")):
                for segment in PREFERRED_SEGMENTS:
                    if f".{segment}.fits" in fits_file.name:
                        segment_counter[segment] += 1
                        break
    
    if segment_counter:
        common_segments = [seg for seg, count in segment_counter.items() if count >= len(sample_cases)]
        if common_segments:
            print(f"Found {len(common_segments)} commonly available segments: {common_segments}")
            try: cache_file.write_text(','.join(common_segments))
            except Exception: pass
            return sorted(common_segments)
        most_common = segment_counter.most_common(1)[0][0]
        print(f"Using most common segment: {most_common}")
        return [most_common]

    print("No segments found in the dataset!")
    return []

def process_reorganized_data(data_dir="sharp_cnn_lstm_data", max_workers=None):
    available_segments = detect_available_segments(data_dir)
    if not available_segments:
        print("No consistent segments found. Cannot proceed.")
        return []
    
    print(f"Processing data using {len(available_segments)} segments: {available_segments}")
    base_path = Path(data_dir)
    case_folders = [f for f in base_path.iterdir() if f.is_dir() and (f.name.startswith('flare_case_') or f.name.startswith('quiet_case_'))]
    print(f"Found {len(case_folders)} case folders")
    
<<<<<<< HEAD
    if max_workers is None: max_workers = min(mp.cpu_count() - 1, 4)
    print(f"Using {max_workers} parallel workers")
    
    batch_size = max(1, min(20, len(case_folders) // max_workers))
=======
    if max_workers is None: max_workers = min(mp.cpu_count() - 1, 3)
    print(f"Using {max_workers} parallel workers")
    
    batch_size = max(1, min(15, len(case_folders) // max_workers))
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
    processed_data = []
    
    for i in range(0, len(case_folders), batch_size):
        batch = case_folders[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(case_folders) + batch_size - 1)//batch_size}")
        try:
            with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as executor:
                futures = {executor.submit(process_case_folder, str(folder), available_segments): folder for folder in batch}
                for future in tqdm(futures, desc=f"Batch {i//batch_size + 1}", leave=False):
                    try:
                        result = future.result(timeout=300)
                        if result is not None: processed_data.append(result)
                    except Exception as e: print(f"Failed to process {futures[future].name}: {e}")
        except Exception as e:
            print(f"Multiprocessing error, falling back to sequential processing: {e}")
            for folder in tqdm(batch, desc=f"Sequential batch {i//batch_size + 1}"):
                try:
                    result = process_case_folder(str(folder), available_segments)
                    if result is not None: processed_data.append(result)
                except Exception as e: print(f"Failed to process {folder.name}: {e}")
    
    print(f"Successfully processed {len(processed_data)} sequences")
    return processed_data

def create_final_dataset(processed_data):
    if not processed_data:
        print("No processed data to convert!")
        return None, None, None, None
    
    n_samples, first_sequence = len(processed_data), processed_data[0]['sequence']
    print(f"Creating final dataset with {n_samples} samples... Sequence shape: {first_sequence.shape}")
    
    X_array = np.empty((n_samples, *first_sequence.shape), dtype=np.float32)
    X_original_array = np.empty((n_samples, *first_sequence.shape), dtype=np.float32) # New array for original data
    y_array = np.empty(n_samples, dtype=np.int8)
    metadata = []

    for i, item in enumerate(tqdm(processed_data, desc="Creating arrays")):
        X_array[i] = item['sequence'].astype(np.float32)
        X_original_array[i] = item['original_sequence'].astype(np.float32) # Store original data
        y_array[i] = item['label']
        metadata.append({'case': item['case'], 'solar_cycle_phase': item['solar_cycle_phase'], 'timestamp': item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')})

    print(f"Final dataset shapes: X={X_array.shape}, X_original={X_original_array.shape}, y={y_array.shape}")
    if len(X_array.shape) != 5: print(f"Warning: Expected 5D array for X, got shape {X_array.shape}")
    
    unique_values, counts = np.unique(y_array, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_values, counts))}")
    phase_counts = Counter([item['solar_cycle_phase'] for item in metadata])
    print(f"Solar cycle phase distribution: {dict(phase_counts)}")
    return X_array, X_original_array, y_array, metadata

def get_solar_cycle_phase(timestamp):
    year = timestamp.year + (timestamp.month - 1) / 12.0
    solar_cycles = [
        {'cycle': 23, 'min_start': 1996.4, 'max': 2000.3, 'min_end': 2008.9},
        {'cycle': 24, 'min_start': 2008.9, 'max': 2014.3, 'min_end': 2019.8},
        {'cycle': 25, 'min_start': 2019.8, 'max': 2025.7, 'min_end': 2030.0},
    ]
    for cycle in solar_cycles:
        if cycle['min_start'] <= year <= cycle['min_end']:
            return 'rising' if year <= cycle['max'] else 'falling'
    if year < 1996.4: return 'falling'
    if year > 2030.0: return 'rising'
    return 'falling'

def process_fits_file(fits_file_path):
    try:
        with fits.open(fits_file_path) as hdul:
            image_data = hdul[1].data
            if np.any(~np.isfinite(image_data)):
                image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
            image_resized = resize(image_data, (IMAGE_HEIGHT, IMAGE_WIDTH), anti_aliasing=False, preserve_range=True)
            # Return both normalized and original resized data
            return normalize_image_fast(image_resized), image_resized
    except Exception as e:
        print(f"Error processing {fits_file_path}: {e}")
        return None, None

def normalize_image_fast(image_data):
    min_val, max_val = np.percentile(image_data, 1), np.percentile(image_data, 99)
    normalized = np.clip((image_data - min_val) / (max_val - min_val), 0, 1) if max_val > min_val else np.zeros_like(image_data, dtype=np.float32)
    return normalized.astype(np.float32)

def process_timestep_optimized(timestep_folder, available_segments):
    try:
        timestamp_pattern = re.compile(r'\.(\d{8}_\d{6})_TAI\.')
        segment_files = {}
        for fits_file in list(timestep_folder.glob("*.fits")):
            for segment in available_segments:
                if f".{segment}.fits" in fits_file.name:
                    if segment not in segment_files: segment_files[segment] = []
                    segment_files[segment].append(fits_file)
        
        if len(segment_files) != len(available_segments): return None
        
        final_segment_files = {seg: sorted(segment_files[seg], key=lambda x: x.name)[-1] for seg in available_segments if seg in segment_files}
        
        sample_file = final_segment_files[available_segments[0]]
        timestamp_match = timestamp_pattern.search(sample_file.name)
        if not timestamp_match: return None
        timestamp = datetime.strptime(timestamp_match.group(1), '%Y%m%d_%H%M%S')
        
        with ThreadPoolExecutor(max_workers=min(4, len(available_segments))) as executor:
            futures = {executor.submit(process_fits_file, final_segment_files[seg]): seg for seg in available_segments}
            # Unpack results into normalized and original
            results = [future.result() for future in futures]
        
        if any(r is None or r[0] is None for r in results): return None
        
        normalized_channels = [r[0] for r in results]
        original_channels = [r[1] for r in results]
        
        return (timestamp, np.stack(normalized_channels, axis=-1), np.stack(original_channels, axis=-1))
    except Exception as e:
        print(f"Error processing timestep {timestep_folder}: {e}")
        return None

def process_case_folder(case_folder_path, available_segments):
    try:
        case_folder = Path(case_folder_path)
        label = 1 if case_folder.name.startswith('flare_case_') else 0
        timestep_folders = sorted([f for f in case_folder.iterdir() if f.is_dir() and f.name.startswith('timestep_')])
        
        if len(timestep_folders) != 6: return None
        
        timestep_data = []
        for folder in timestep_folders:
            result = process_timestep_optimized(folder, available_segments)
            if result is None: return None
            timestep_data.append(result)
        
        timestep_data.sort(key=lambda x: x[0])
        first_timestamp = timestep_data[0][0]
        
        return {
            'case': case_folder.name, 
            'sequence': np.stack([item[1] for item in timestep_data]),
            'original_sequence': np.stack([item[2] for item in timestep_data]), # Add original sequence
            'label': label, 
            'solar_cycle_phase': get_solar_cycle_phase(first_timestamp),
            'timestamp': first_timestamp
        }
    except Exception as e:
        print(f"Error processing case {case_folder_path}: {e}")
        return None

def _init_worker():
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)

def load_external_npz(external_npz_path):
    try:
        if not os.path.exists(external_npz_path):
            print(f"External .npz file not found: {external_npz_path}")
            return None, None, None, None
        print(f"Loading external .npz file: {external_npz_path}")
        data = np.load(external_npz_path, allow_pickle=True)
        X, y, metadata = data['X'], data['y'], data['metadata']
        
        # Safely load X_original, create placeholder if not present
        if 'X_original' in data:
            X_original = data['X_original']
        else:
            print("Warning: 'X_original' not found in external file. Creating NaN placeholder.")
            X_original = np.full_like(X, np.nan, dtype=np.float32)

        print(f"Loaded external data: X={X.shape}, X_original={X_original.shape}, y={y.shape}, metadata={len(metadata)}")
        if isinstance(metadata, np.ndarray): metadata = metadata.tolist()
        return X, X_original, y, metadata
    except Exception as e:
        print(f"Error loading external .npz file: {e}")
        return None, None, None, None

def remove_duplicates_by_case(X_curr, X_orig_curr, y_curr, meta_curr, X_ext, X_orig_ext, y_ext, meta_ext):
    print("Removing duplicates between current and external data...")
    current_cases = {meta['case'] for meta in meta_curr}
    print(f"Current data has {len(current_cases)} unique cases")
    
    ext_indices_to_keep = [i for i, meta in enumerate(meta_ext) if meta['case'] not in current_cases]
    print(f"Found {len(meta_ext) - len(ext_indices_to_keep)} duplicate cases in external data")
    print(f"Keeping {len(ext_indices_to_keep)} unique cases from external data")
    
    if not ext_indices_to_keep:
        print("No unique external data to add")
        return X_curr, X_orig_curr, y_curr, meta_curr
    
    X_ext_filtered, X_orig_ext_filtered, y_ext_filtered = X_ext[ext_indices_to_keep], X_orig_ext[ext_indices_to_keep], y_ext[ext_indices_to_keep]
    meta_ext_filtered = [meta_ext[i] for i in ext_indices_to_keep]
    
    X_combined = np.concatenate([X_curr, X_ext_filtered], axis=0)
    X_orig_combined = np.concatenate([X_orig_curr, X_orig_ext_filtered], axis=0)
    y_combined = np.concatenate([y_curr, y_ext_filtered], axis=0)
    meta_combined = meta_curr + meta_ext_filtered
    print(f"Combined data shapes: X={X_combined.shape}, X_original={X_orig_combined.shape}, y={y_combined.shape}, metadata={len(meta_combined)}")
    return X_combined, X_orig_combined, y_combined, meta_combined

def combine_with_external_data(X_curr, X_orig_curr, y_curr, meta_curr, external_npz_path):
    X_ext, X_orig_ext, y_ext, meta_ext = load_external_npz(external_npz_path)
    if X_ext is None:
        print("Failed to load external data, using only current data")
        return X_curr, X_orig_curr, y_curr, meta_curr
    
    if X_ext.shape[1:] != X_curr.shape[1:]:
        print(f"Warning: External data shape {X_ext.shape} doesn't match current data shape {X_curr.shape}. Sequence dimensions must match.")
        return X_curr, X_orig_curr, y_curr, meta_curr
    
    X_comb, X_orig_comb, y_comb, meta_comb = remove_duplicates_by_case(X_curr, X_orig_curr, y_curr, meta_curr, X_ext, X_orig_ext, y_ext, meta_ext)
    
    unique_values, counts = np.unique(y_comb, return_counts=True)
    print(f"Combined class distribution: {dict(zip(unique_values, counts))}")
    phase_counts = Counter([item['solar_cycle_phase'] for item in meta_comb])
    print(f"Combined solar cycle phase distribution: {dict(phase_counts)}")
    return X_comb, X_orig_comb, y_comb, meta_comb

if __name__ == "__main__":
    import time
    start_time = time.time()
    print(f"Using {mp.cpu_count()} CPU cores")
    
    processed_data = process_reorganized_data(JSOC_DIR)
    if not processed_data:
        print("No data was processed successfully. Exiting.")
    else:
        X_final, X_original_final, y_final, metadata = create_final_dataset(processed_data)
        if X_final is not None and X_final.size > 0:
            print(f"Initial dataset shapes: X={X_final.shape}, X_original={X_original_final.shape}, y={y_final.shape}, metadata entries: {len(metadata)}")
            
            if COMBINE_EXTERNAL_NPZ:
                print(f"\nCombining with external .npz file...")
                X_final, X_original_final, y_final, metadata = combine_with_external_data(X_final, X_original_final, y_final, metadata, EXTERNAL_NPZ_FILE)
            
            print(f"Final dataset shapes: X={X_final.shape}, X_original={X_original_final.shape}, y={y_final.shape}, metadata entries: {len(metadata)}")
            total_mem_gb = (X_final.nbytes + X_original_final.nbytes) / 1024**3
            print(f"Total memory usage: {total_mem_gb:.2f} GB")
            
            print(f"Saving to '{OUTPUT_FILE}' with compression...")
            np.savez_compressed(OUTPUT_FILE, X=X_final, X_original=X_original_final, y=y_final, metadata=metadata)
            print(f"Successfully saved optimized data to '{OUTPUT_FILE}'")

            elapsed_time = time.time() - start_time
            print(f"Total processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
            print(f"Average time per case: {elapsed_time/len(processed_data):.2f} seconds")

            print(f"\nOutput format: [n_samples={X_final.shape[0]}, timesteps={X_final.shape[1]}, height={X_final.shape[2]}, width={X_final.shape[3]}, channels={X_final.shape[4]}]")
            
            if metadata:
                print(f"\nSample metadata entries:")
                for i, meta in enumerate(metadata[:3]):
                    print(f"  {i+1}. Case: {meta['case']}, Solar cycle: {meta['solar_cycle_phase']}, Timestamp: {meta['timestamp']}")
                if len(metadata) > 3: print(f"  ... and {len(metadata) - 3} more entries")
            
            del X_final, X_original_final, y_final
            print("Memory cleaned up successfully")
        else:
            print("Could not generate final dataset from processed images.")
    
    print(f"\nProcessing completed in {time.time() - start_time:.2f} seconds")