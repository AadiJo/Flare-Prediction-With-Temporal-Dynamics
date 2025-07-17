import os
import pandas as pd
from dotenv import load_dotenv
import json
import re
from datetime import datetime
import asyncio
import drms
import aiohttp
import aiofiles
import numpy as np
from astropy.io import fits
from skimage.transform import resize
from pathlib import Path
import warnings
from collections import Counter

# --- Configuration ---
load_dotenv()
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Core Settings ---
# Directory to save the final processed .npz files
OUTPUT_DIR = "processed_data_batches"
# Set to True to process flare cases first, False to process quiet cases first
GET_FLARE_DATA_FIRST = False
MIN_DATE = '2000-06-18'

# --- Model Data Settings ---
IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH = 128, 128, 12
# Define which data to download. These are image-based.
IMAGE_SEGMENTS = ["Bp", "Bt", "Br", "continuum"]
# Define scalar keywords to be extracted from FITS headers.
SCALAR_KEYWORDS = ["USFLUX", "MEANSHR", "EPSZ", "TOTUSJZ", "SHRGT45", "TOTPOT"]
# Combine all requested data channels
ALL_CHANNELS = IMAGE_SEGMENTS + SCALAR_KEYWORDS


# --- Download & Processing Settings ---
# How many cases (flares or quiet periods) to process concurrently
BATCH_SIZE = 5
# How many processed samples to accumulate in memory before saving to a batch file
SAVE_BATCH_SIZE = 50
# Files to track script progress and failures
PROGRESS_FILE = "download_progress_on_the_fly.json"
FAILED_DOWNLOADS_FILE = "failed_downloads_on_the_fly.json"

def normalize_image_fast(image_data):
    """Normalizes a 2D numpy array using percentile clipping."""
    min_val, max_val = np.percentile(image_data, 1), np.percentile(image_data, 99)
    if max_val > min_val:
        normalized = np.clip((image_data - min_val) / (max_val - min_val), 0, 1)
    else:
        normalized = np.zeros_like(image_data, dtype=np.float32)
    return normalized.astype(np.float32)

def get_solar_cycle_phase(timestamp):
    """Determines if a timestamp is in the rising or falling phase of a solar cycle."""
    year = timestamp.year + (timestamp.month - 1) / 12.0
    # Simplified cycle definitions
    solar_cycles = [
        {'cycle': 24, 'min_start': 2008.9, 'max': 2014.3, 'min_end': 2019.8},
        {'cycle': 25, 'min_start': 2019.8, 'max': 2025.7, 'min_end': 2030.0},
    ]
    for cycle in solar_cycles:
        if cycle['min_start'] <= year <= cycle['min_end']:
            return 'rising' if year <= cycle['max'] else 'falling'
    return 'rising' if year > 2025.7 else 'falling'

async def process_fits_data_in_memory(fits_content, segment_name):
    """
    NEW: Processes raw FITS file content from memory.
    Handles both 2D image data and 0D scalar data.
    """
    try:
        # Use astropy to open the FITS content from a memory buffer
        with fits.open(fits_content) as hdul:
            if segment_name in SCALAR_KEYWORDS:
                # It's a scalar keyword, extract it from the primary header
                scalar_value = hdul[0].header.get(segment_name, 0.0)
                # Broadcast the single value into a 2D array to match image channels
                processed_array = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), scalar_value, dtype=np.float32)
                return processed_array
            else:
                # It's an image segment, process it as before
                image_data = hdul[1].data
                image_data = np.nan_to_num(image_data, nan=0.0)
                image_resized = resize(image_data, (IMAGE_HEIGHT, IMAGE_WIDTH), anti_aliasing=False, preserve_range=True)
                return normalize_image_fast(image_resized)
    except Exception as e:
        print(f"✗ Error processing FITS data for segment {segment_name} in memory: {e}")
        return None

async def download_and_process_file(session, url, segment_name, max_retries=3):
    """
    NEW: Asynchronously downloads a file's content into memory and processes it immediately.
    Returns a processed numpy array, not the raw file.
    Includes comprehensive retry logic for rate limiting and server errors.
    """
    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes
            async with session.get(url, timeout=timeout) as response:
                # Handle rate limiting and server errors with exponential backoff
                if response.status in [429, 503, 502, 504] and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Server error {response.status} for {segment_name}, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                
                # Handle other HTTP errors that might be worth retrying
                if response.status in [500, 520, 522, 524] and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Server error {response.status} for {segment_name}, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                # Read the file content into a memory buffer
                file_content_buffer = await response.read()
                # Immediately process the content
                return await process_fits_data_in_memory(file_content_buffer, segment_name)
                
        except asyncio.TimeoutError as e:
            print(f"Timeout error for {segment_name}, attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
                continue
            else:
                print(f"✗ Final timeout failure for {segment_name}: {e}")
                return None
                
        except Exception as e:
            print(f"Download error for {segment_name}, attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
                continue
            else:
                print(f"✗ Final download failure for {segment_name}: {e}")
                return None
    return None

async def download_and_process_time_series(client, harpnum, time_points, case_key, progress_data, failed_downloads, lock, session):
    """
    NEW: Main function to handle a single case (flare or quiet).
    It orchestrates the download and on-the-fly processing for a 12-step time series.
    Now uses a shared session instead of creating its own.
    """
    time_objects = [pd.to_datetime(ts) for ts in time_points]
    start_time, end_time = min(time_objects), max(time_objects)
    first_timestamp = start_time # For metadata

    # Construct the query string with all required image segments and scalar keywords
    segments_query_str = "{" + ", ".join(ALL_CHANNELS) + "}"
    query = f"hmi.sharp_cea_720s[{int(harpnum)}][{start_time.strftime('%Y.%m.%d_%H:%M:%S_TAI')}-{end_time.strftime('%Y.%m.%d_%H:%M:%S_TAI')}]{segments_query_str}"

    try:
        print(f"Submitting request for case: {case_key}")
        # Run synchronous DRMS export request in a thread pool with retry logic
        export_request = None
        max_export_retries = 3
        
        for export_attempt in range(max_export_retries):
            try:
                export_request = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: client.export(query, method='url', protocol='fits')
                )
                # Wait for the export request to complete
                await asyncio.get_event_loop().run_in_executor(None, export_request.wait)
                
                if export_request.status == 0 and hasattr(export_request, 'urls') and not export_request.urls.empty:
                    break  # Success!
                else:
                    print(f"✗ JSOC export failed for {case_key}, attempt {export_attempt + 1}/{max_export_retries}. Status: {export_request.status}")
                    if export_attempt < max_export_retries - 1:
                        wait_time = 10 * (2 ** export_attempt)  # 10s, 20s, 40s
                        print(f"Retrying JSOC export in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        
            except Exception as e:
                print(f"✗ JSOC export error for {case_key}, attempt {export_attempt + 1}/{max_export_retries}: {e}")
                if export_attempt < max_export_retries - 1:
                    wait_time = 10 * (2 ** export_attempt)
                    print(f"Retrying JSOC export in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise e

        if not export_request or export_request.status != 0 or not hasattr(export_request, 'urls') or export_request.urls.empty:
            print(f"✗ No data returned from JSOC for case {case_key} after {max_export_retries} attempts.")
            return None

        # Group files by their observation time
        files_by_time = {}
        for _, row in export_request.urls.iterrows():
            match = re.search(r'\.(\d{8}_\d{6})_TAI\.', row['filename'])
            if match:
                obs_time = datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
                if obs_time not in files_by_time:
                    files_by_time[obs_time] = []
                files_by_time[obs_time].append(row)

        # Find the 12 observation times that are closest to our target time points
        if len(files_by_time) < SEQUENCE_LENGTH:
            print(f"✗ Insufficient data for {case_key}: found {len(files_by_time)}/{SEQUENCE_LENGTH} unique timesteps.")
            return None

        # Find the best matching observation times for our 12 steps (ensuring unique timesteps)
        sorted_obs_times = sorted(files_by_time.keys())
        sequence_files = []
        used_obs_times = set()
        
        for target_time in time_objects:
            # Find the closest available observation time that hasn't been used yet
            available_times = [t for t in sorted_obs_times if t not in used_obs_times]
            if not available_times:
                print(f"✗ Not enough unique observation times for {case_key}")
                return None
            
            best_match_time = min(available_times, key=lambda t: abs((t - target_time).total_seconds()))
            used_obs_times.add(best_match_time)
            sequence_files.append(files_by_time[best_match_time])

        # --- On-the-fly processing loop ---
        all_timesteps_data = []
        for timestep_files_df_list in sequence_files:
            tasks = []
            # Create a download+process task for each channel in the timestep
            for file_info in timestep_files_df_list:
                segment_name = file_info['filename'].split('.')[-2]
                if segment_name in ALL_CHANNELS:
                    url = f"https://jsoc1.stanford.edu{file_info['url']}"
                    tasks.append(download_and_process_file(session, url, segment_name))

            # Run all channel processing for the current timestep concurrently
            processed_channels = await asyncio.gather(*tasks)

            if any(channel is None for channel in processed_channels):
                print(f"✗ Failed to process one or more channels for a timestep in case {case_key}.")
                return None

            # Stack the processed channels to form the (H, W, C) array for this timestep
            timestep_array = np.stack(processed_channels, axis=-1)
            all_timesteps_data.append(timestep_array)

        # Stack the 12 timesteps to form the final (T, H, W, C) sequence array
        final_sequence = np.stack(all_timesteps_data, axis=0)

        label = 1 if 'flare' in case_key else 0
        solar_phase = get_solar_cycle_phase(first_timestamp)

        print(f"✓ Successfully processed case {case_key}")
        return {
            'X': final_sequence.astype(np.float16), # Use float16 to save memory
            'y': np.int8(label),
            'metadata': {'case': case_key, 'solar_cycle_phase': solar_phase, 'timestamp': first_timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        }

    except Exception as e:
        print(f"✗ Major error processing case {case_key}: {e}")
        async with lock:
            failed_downloads.append(case_key)
            async with aiofiles.open(FAILED_DOWNLOADS_FILE, "w") as f:
                await f.write(json.dumps(failed_downloads, indent=4))
        return None

# --- Main Application Logic ---
def load_noaa_to_harpnum_map_local(filepath="HARP_Mapping.txt"):
    if not os.path.exists(filepath): return {}
    noaa_to_harp = {}
    with open(filepath, "r") as f:
        lines = f.read().strip().splitlines()
    for line in lines[1:]:
        if not line.strip(): continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            harpnum, noaa_ars = parts
            for noaa in noaa_ars.split(','):
                if noaa.strip().isdigit():
                    noaa_to_harp.setdefault(noaa.strip(), []).append(harpnum)
    return noaa_to_harp

async def process_data_type(client, data_df, num_samples, progress_key, data_label, all_results, lock, progress_data, failed_downloads):
    print(f"\nProcessing {data_label} samples...")
    noaa_map = load_noaa_to_harpnum_map_local()

    # Collect all the task parameters instead of creating coroutines
    task_params = []
    for row in data_df.itertuples():
        if len(progress_data[progress_key]) + len(task_params) >= num_samples:
            break

        peak_time = pd.to_datetime(row.peak_time)
        noaa = int(row.noaa_active_region)
        item_id = peak_time.strftime('%Y%m%d_%H%M')
        case_key = f"{data_label}_case_{noaa}_{item_id}"

        if case_key in progress_data[progress_key] or case_key in failed_downloads:
            continue

        harpnums = noaa_map.get(str(noaa))
        if not harpnums:
            continue
        harpnum = harpnums[0]

        # Define the 12 time points from 24 hours to 13 hours before the flare (1 per hour)
        flare_minus_24h = peak_time - pd.Timedelta(hours=24)
        time_points = [(flare_minus_24h + pd.Timedelta(hours=t)) for t in range(SEQUENCE_LENGTH)]
        time_points_str = [tp.strftime('%Y-%m-%d %H:%M:%S') for tp in time_points]

        task_params.append({
            'client': client,
            'harpnum': harpnum,
            'time_points_str': time_points_str,
            'case_key': case_key,
            'progress_data': progress_data,
            'failed_downloads': failed_downloads,
            'lock': lock
        })

    # Run tasks in batches with a shared session
    connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)
    async with aiohttp.ClientSession(connector=connector) as shared_session:
        for i in range(0, len(task_params), BATCH_SIZE):
            batch_params = task_params[i:i+BATCH_SIZE]
            print(f"--- Running batch {i//BATCH_SIZE + 1} of {data_label} cases ({len(batch_params)} tasks) ---")
            
            # Create tasks with the shared session
            batch_tasks = []
            for params in batch_params:
                batch_tasks.append(
                    download_and_process_time_series(
                        params['client'], params['harpnum'], params['time_points_str'], 
                        params['case_key'], params['progress_data'], params['failed_downloads'], 
                        params['lock'], shared_session
                    )
                )
            
            results = await asyncio.gather(*batch_tasks)

            for result in results:
                if result:
                    all_results.append(result)
                    # Update progress file immediately after a successful process
                    async with lock:
                        case_key = result['metadata']['case']
                        if case_key not in progress_data[progress_key]:
                            progress_data[progress_key].append(case_key)

            # Save progress to disk after each batch
            async with lock:
                with open(PROGRESS_FILE, "w") as f:
                    json.dump(progress_data, f, indent=4)
            print(f"--- Batch {i//BATCH_SIZE + 1} finished. Total results collected: {len(all_results)} ---")
            await asyncio.sleep(10) # Be nice to the server - increased delay

async def main():
    jsoc_email = os.getenv("EMAIL")
    if not jsoc_email:
        raise Exception("EMAIL environment variable not set.")

    client = drms.Client(email=jsoc_email)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    try:
        with open(PROGRESS_FILE, "r") as f:
            progress_data = json.load(f)
    except FileNotFoundError:
        progress_data = {"flare": [], "quiet": []}

    try:
        with open(FAILED_DOWNLOADS_FILE, "r") as f:
            failed_downloads = json.load(f)
    except FileNotFoundError:
        failed_downloads = []

    # Load and prepare flare data from CSV
    df = pd.read_csv('goes_flares.csv').dropna(subset=['noaa_active_region', 'goes_class'])
    df['noaa_active_region'] = df['noaa_active_region'].astype(int)
    df = df[pd.to_datetime(df['start_time']) >= pd.Timestamp(MIN_DATE)]
    m_x_flares = df[df['goes_class'].str.startswith(('M', 'X'))]
    non_m_x_flares = df[~df['goes_class'].str.startswith(('M', 'X'))].sample(frac=1, random_state=42)

    NUM_PREFLARE_SAMPLES, NUM_QUIET_SAMPLES = 1196, 1196
    all_results = []
    lock = asyncio.Lock()
    batch_file_counter = 0

    async def run_processing(data_df, num_samples, p_key, label):
        nonlocal batch_file_counter
        await process_data_type(client, data_df, num_samples, p_key, label, all_results, lock, progress_data, failed_downloads)
        # Check if we need to save a batch
        if len(all_results) >= SAVE_BATCH_SIZE:
            print(f"\n--- Reached batch size of {SAVE_BATCH_SIZE}. Saving data... ---")
            X_batch = np.stack([item['X'] for item in all_results])
            y_batch = np.array([item['y'] for item in all_results])
            meta_batch = [item['metadata'] for item in all_results]
            
            batch_filename = Path(OUTPUT_DIR) / f"processed_batch_{batch_file_counter:04d}.npz"
            np.savez_compressed(batch_filename, X=X_batch, y=y_batch, metadata=meta_batch)
            print(f"✓ Successfully saved to {batch_filename}")
            
            # Reset results list and increment counter
            all_results.clear()
            batch_file_counter += 1

    # Main execution flow
    if GET_FLARE_DATA_FIRST:
        await run_processing(m_x_flares, NUM_PREFLARE_SAMPLES, "flare", "flare")
        await run_processing(non_m_x_flares, NUM_QUIET_SAMPLES, "quiet", "quiet")
    else:
        await run_processing(non_m_x_flares, NUM_QUIET_SAMPLES, "quiet", "quiet")
        await run_processing(m_x_flares, NUM_PREFLARE_SAMPLES, "flare", "flare")

    # Save any remaining data that didn't make a full batch
    if all_results:
        print(f"\n--- Saving final remaining {len(all_results)} samples... ---")
        X_batch = np.stack([item['X'] for item in all_results])
        y_batch = np.array([item['y'] for item in all_results])
        meta_batch = [item['metadata'] for item in all_results]
        
        batch_filename = Path(OUTPUT_DIR) / f"processed_batch_{batch_file_counter:04d}.npz"
        np.savez_compressed(batch_filename, X=X_batch, y=y_batch, metadata=meta_batch)
        print(f"✓ Successfully saved to {batch_filename}")

    print("\n--- All processing complete. ---")

if __name__ == "__main__":
    print("=== Starting JSOC Data Download Script ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check if basic requirements are met
    if not os.path.exists('.env') and not os.getenv('EMAIL'):
        print("✗ No .env file found and EMAIL environment variable not set")
        print("Create a .env file with: EMAIL=your_email@example.com")
        sys.exit(1)
    
    if not os.path.exists('goes_flares.csv'):
        print("✗ goes_flares.csv not found")
        sys.exit(1)
    
    if not os.path.exists('HARP_Mapping.txt'):
        print("✗ HARP_Mapping.txt not found")
        sys.exit(1)
    
    print("✓ Basic requirements check passed")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✗ Script interrupted by user")
    except Exception as e:
        print(f"\n✗ Script failed with error: {e}")
        import traceback
        traceback.print_exc()
