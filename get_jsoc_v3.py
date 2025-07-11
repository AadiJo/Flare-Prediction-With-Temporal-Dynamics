import os
import pandas as pd
from dotenv import load_dotenv
import json
import random
import re
from datetime import datetime
import shutil
import asyncio
import drms
import aiohttp
import aiofiles

load_dotenv()
GET_FLARE_DATA_FIRST = False
PROGRESS_FILE = "download_progress_async.json"
FAILED_DOWNLOADS_FILE = "failed_downloads.json"
MIN_DATE = '2023-06-18'  # Corrected date format

async def download_file(session, url, path, max_retries=3):
    """Asynchronously downloads a single file using aiohttp with retry logic."""
    # Normalize the path to handle cross-platform issues
    path = os.path.normpath(path)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            # Add timeout and better error handling
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
            async with session.get(url, timeout=timeout) as response:
                if response.status in [429, 503, 502, 504]:
                    # Rate limited, service unavailable, bad gateway, or gateway timeout - wait and retry
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        status_msg = {429: "Rate limited", 503: "Service unavailable", 502: "Bad gateway", 504: "Gateway timeout"}
                        print(f"{status_msg.get(response.status, 'Server error')} for {url}, retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                
                response.raise_for_status()
                async with aiofiles.open(path, 'wb') as f:
                    await f.write(await response.read())
                return True  # Success
                
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Timeout for {url}, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                continue
            else:
                print(f"✗ Failed to download {url}: Timeout after {max_retries} attempts")
                return False
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Error downloading {url}, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
                continue
            else:
                print(f"✗ Failed to download {url}: {e}")
                return False
    
    return False

async def download_sharp_time_series_async(client, harpnum, time_points, base_dir, flare_key, progress_dict_key, processed_flares, failed_downloads, lock):
    """Asynchronously download a single time series using drms and aiohttp."""
    time_objects = [pd.to_datetime(ts) for ts in time_points]
    start_time, end_time = min(time_objects), max(time_objects)
    
    # Normalize base directory path
    base_dir = os.path.normpath(base_dir)
    
    if os.path.exists(base_dir):
        print(f"Found existing case directory {base_dir}, cleaning up and starting fresh.")
        shutil.rmtree(base_dir)

    query = f"hmi.sharp_cea_720s[{int(harpnum)}][{start_time.strftime('%Y.%m.%d_%H:%M:%S_TAI')}-{end_time.strftime('%Y.%m.%d_%H:%M:%S_TAI')}]{{Bp, Bt, Br, continuum}}"
    
    try:
        print(f"Submitting download request for HARPNUM: {harpnum}, Flare/Quiet Key: {flare_key}")
        
        # Retry logic for JSOC export requests
        max_export_retries = 5
        for attempt in range(max_export_retries):
            try:
                # Run the synchronous DRMS operations in a thread pool
                export_request = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: client.export(query, method='url', protocol='fits')
                )
                
                # Wait for the export request to complete (also synchronous)
                await asyncio.get_event_loop().run_in_executor(
                    None, export_request.wait
                )
                
                # Check if export was successful - status 0 means success, and we need URLs
                if export_request.status == 0 and hasattr(export_request, 'urls') and len(export_request.urls) > 0:
                    break  # Success, exit retry loop
                elif export_request.status == 7 or "pending export requests" in str(getattr(export_request, 'error', '')):
                    # Rate limited by JSOC - wait and retry
                    if attempt < max_export_retries - 1:
                        wait_time = 30 + (attempt * 15)  # 30s, 45s, 60s, 75s
                        print(f"JSOC rate limited for {flare_key}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_export_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"✗ Export failed for {flare_key} after {max_export_retries} attempts: rate limited")
                        return
                else:
                    print(f"No data found or export failed for {flare_key} (status: {export_request.status})")
                    return # Exit the function
                    
            except Exception as e:
                if "pending export requests" in str(e) or "status=7" in str(e):
                    # Rate limited by JSOC - wait and retry
                    if attempt < max_export_retries - 1:
                        wait_time = 30 + (attempt * 15)  # 30s, 45s, 60s, 75s
                        print(f"JSOC rate limited for {flare_key}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_export_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"✗ Export failed for {flare_key} after {max_export_retries} attempts: {e}")
                        return
                else:
                    # Other error, re-raise
                    raise e

        os.makedirs(base_dir, exist_ok=True)
        
        # Create timestep directories upfront to avoid path issues during download
        for i in range(len(time_objects)):
            timestep_dir = os.path.join(base_dir, f"timestep_{i+1:02d}")
            os.makedirs(timestep_dir, exist_ok=True)
        
        # Create session with better connection handling
        connector = aiohttp.TCPConnector(
            limit=10,  # Limit concurrent connections
            limit_per_host=5,  # Limit per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
        )
        
        download_tasks = []
        successful_downloads = []
        
        async with aiohttp.ClientSession(connector=connector) as session:
            for i, row in export_request.urls.iterrows():
                filename = row['filename']
                url = row['url']
                
                # Fix URL scheme issues
                if url.startswith('http://jsoc.stanford.edu') and 'jsoc1.stanford.edu' in url:
                    # Sometimes URLs are mixed - ensure consistent HTTPS
                    url = url.replace('http://jsoc.stanford.edu', 'https://jsoc1.stanford.edu')
                elif not url.startswith('http'):
                    url = f"https://jsoc1.stanford.edu{url}"
                elif url.startswith('http://jsoc.stanford.edu'):
                    # Try HTTPS first
                    url = url.replace('http://jsoc.stanford.edu', 'https://jsoc1.stanford.edu')
                
                # Handle case where filename might be empty or just a directory
                if not filename or filename.endswith('/'):
                    # Extract filename from URL as fallback
                    url_filename = url.split('/')[-1]
                    if url_filename and '.' in url_filename:
                        filename = url_filename
                    else:
                        filename = f"file_{i:04d}.fits"  # Generic fallback
                
                # Use os.path.join for proper path construction
                filepath = os.path.join(base_dir, filename)
                filepath = os.path.normpath(filepath)  # Normalize the path
                
                download_tasks.append(download_file(session, url, filepath))
            
            # Execute all download tasks concurrently and collect results
            if download_tasks:
                results = await asyncio.gather(*download_tasks, return_exceptions=True)
                successful_downloads = [i for i, result in enumerate(results) if result is True]
        
        # Check if we got any successful downloads
        if not successful_downloads:
            print(f"✗ All downloads failed for {flare_key}. Skipping.")
            # Add to failed downloads
            failed_downloads.append(flare_key)
            async with lock:
                with open(FAILED_DOWNLOADS_FILE, "w") as f:
                    json.dump(failed_downloads, f, indent=4)
            return

        # Your file organization logic...
        files_in_dir = []
        if os.path.exists(base_dir):
            files_in_dir = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.fits')]
        
        if not files_in_dir:
            print(f"✗ No FITS files found in {base_dir} after download attempt. Skipping.")
            # Add to failed downloads
            failed_downloads.append(flare_key)
            async with lock:
                with open(FAILED_DOWNLOADS_FILE, "w") as f:
                    json.dump(failed_downloads, f, indent=4)
            return

        for file_path in files_in_dir:
            file_path = os.path.normpath(file_path)  # Normalize path
            filename = os.path.basename(file_path)
            m = re.search(r'\.(\d{8}_\d{6})_TAI\.', filename)
            if m:
                file_time = datetime.strptime(m.group(1), '%Y%m%d_%H%M%S')
                best_timestep = min(range(len(time_objects)), key=lambda i: abs((file_time - time_objects[i].to_pydatetime()).total_seconds()))
                new_path = os.path.join(base_dir, f"timestep_{best_timestep+1:02d}", filename)
            else:
                new_path = os.path.join(base_dir, "timestep_01", filename)
            
            new_path = os.path.normpath(new_path)  # Normalize new path
            
            if os.path.exists(file_path) and file_path != new_path:
                 os.rename(file_path, new_path)

        # Update progress safely
        processed_flares[progress_dict_key].append(flare_key)
        async with lock:
            with open(PROGRESS_FILE, "w") as f:
                json.dump(processed_flares, f, indent=4)

        print(f"✓ Completed time series for {flare_key}({base_dir})")

    except Exception as e:
        print(f"✗ Error processing time series for {flare_key}: {e}")
        # Add to failed downloads
        failed_downloads.append(flare_key)
        async with lock:
            with open(FAILED_DOWNLOADS_FILE, "w") as f:
                json.dump(failed_downloads, f, indent=4)


# Keep all your other functions (load_noaa_to_harpnum_map_local, etc.) unchanged
def load_noaa_to_harpnum_map_local(filepath="HARP_Mapping.txt"):
    noaa_to_harp = {}
    with open(filepath, "r") as f:
        lines = f.read().strip().splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        harpnum, noaa_ars = line.split(maxsplit=1)
        for noaa in noaa_ars.split(','):
            if noaa.strip().isdigit():
                noaa_to_harp.setdefault(noaa.strip(), []).append(harpnum)
    return noaa_to_harp

def load_noaa_to_harpnum_map(txt_url="http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa_ars/all_harps_with_noaa_ars.txt"):
    import requests
    resp = requests.get(txt_url)
    resp.raise_for_status()
    lines = resp.text.strip().splitlines()
    noaa_to_harp = {}
    for line in lines[1:]:
        harpnum, noaa_ars = line.split(maxsplit=1)
        for noaa in noaa_ars.split(','):
            if noaa.strip().isdigit():
                noaa_to_harp.setdefault(noaa.strip(), []).append(harpnum)
    return noaa_to_harp

def get_harpnum_for_noaa(noaa):
    try:
        noaa_to_harp = load_noaa_to_harpnum_map_local()
    except Exception:
        noaa_to_harp = load_noaa_to_harpnum_map()
    harpnums = noaa_to_harp.get(str(noaa), [])
    if not harpnums:
        print(f"No HARPNUM found for NOAA AR {noaa}")
        return None
    return harpnums[0]

async def main():
    jsoc_email = os.getenv("EMAIL")
    if not jsoc_email:
        raise Exception("EMAIL environment variable not set.")
    
    # Create a single drms client for all operations
    client = drms.Client(email=jsoc_email)

    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            processed_flares = json.load(f)
    else:
        processed_flares = {"preflare": [], "quiet": []}

    # Load failed downloads
    if os.path.exists(FAILED_DOWNLOADS_FILE):
        with open(FAILED_DOWNLOADS_FILE, "r") as f:
            failed_downloads = json.load(f)
    else:
        failed_downloads = []

    FLARE_CSV = 'goes_flares.csv'
    if not os.path.exists(FLARE_CSV):
        print(f"Flare catalog '{FLARE_CSV}' not found.")
        return
    df = pd.read_csv(FLARE_CSV).dropna(subset=['noaa_active_region', 'goes_class'])
    df['noaa_active_region'] = df['noaa_active_region'].astype(int)
    df = df[pd.to_datetime(df['peak_time']) >= pd.Timestamp('2010-05-01')]
    # Apply minimum date filter
    df = df[pd.to_datetime(df['peak_time']) >= pd.Timestamp(MIN_DATE)]
    m_x_flares = df[df['goes_class'].str.startswith(('M', 'X'))]
    QUIET_END_DATE = '2024-05-29'
    non_m_x_flares = df[~df['goes_class'].str.startswith(('M', 'X'))]
    non_m_x_flares = non_m_x_flares[pd.to_datetime(non_m_x_flares['peak_time']) <= pd.Timestamp(QUIET_END_DATE)]
    non_m_x_flares = non_m_x_flares.sample(frac=1, random_state=42).reset_index(drop=True)
    
    NUM_PREFLARE_SAMPLES, NUM_QUIET_SAMPLES = 400, 400
    TIME_STEPS, HOURS_BETWEEN_STEPS = 6, 1
    PREDICTION_HORIZON = 12
    BATCH_SIZE = 5  # Process samples in batches to avoid overwhelming the system

    async def process_data(data_df, num_samples, progress_key, data_label):
        """Generic function to collect and run download tasks in batches."""
        print(f"\nProcessing {data_label} samples...")
        print(f"Starting processing for {data_label} samples. Total to process: {num_samples}")
        
        lock = asyncio.Lock()
        total_processed = 0
        
        # Continue processing until we reach the desired number of samples
        while len(processed_flares[progress_key]) + total_processed < num_samples:
            tasks = []
            
            for row in data_df.itertuples():
                if len(processed_flares[progress_key]) + total_processed + len(tasks) >= num_samples:
                    break

                peak_time = pd.to_datetime(row.peak_time)
                noaa = int(row.noaa_active_region)
                item_id = peak_time.strftime('%Y%m%d_%H%M')
                item_key = f"{noaa}_{item_id}"

                if item_key in processed_flares[progress_key]:
                    continue # Skip already processed item

                harpnum = get_harpnum_for_noaa(noaa)
                if harpnum is None:
                    continue

                # Also check if directory already exists and has files
                base_dir = f"async_sharp/{data_label}_case_{harpnum}_NOAA_{noaa}_{item_id}"
                base_dir = os.path.normpath(base_dir)
                
                if os.path.exists(base_dir):
                    # Check if there are any .fits files in the directory
                    fits_files = []
                    for root, dirs, files in os.walk(base_dir):
                        fits_files.extend([f for f in files if f.endswith('.fits')])
                    
                    if fits_files:
                        print(f"Found existing data for {item_key}, skipping...")
                        # Add to processed list if not already there
                        if item_key not in processed_flares[progress_key]:
                            processed_flares[progress_key].append(item_key)
                            async with asyncio.Lock():
                                with open(PROGRESS_FILE, "w") as f:
                                    json.dump(processed_flares, f, indent=4)
                        continue

                
                flare_minus_12h = peak_time - pd.Timedelta(hours=PREDICTION_HORIZON)
                time_points = [(flare_minus_12h - pd.Timedelta(hours=(TIME_STEPS - t))) for t in range(1, TIME_STEPS + 1)]
                time_points_str = [tp.strftime('%Y-%m-%d %H:%M:%S') for tp in time_points]
                
                # Use forward slashes for consistency across platforms
                base_dir = f"async_sharp/{data_label}_case_{harpnum}_NOAA_{noaa}_{item_id}"
                base_dir = os.path.normpath(base_dir)  # Normalize the path
                
                tasks.append(download_sharp_time_series_async(client, harpnum, time_points_str, base_dir, item_key, progress_key, processed_flares, failed_downloads, lock))
                
                # Stop collecting tasks when we reach batch size
                if len(tasks) >= BATCH_SIZE:
                    break
            
            if not tasks:
                print(f"No more valid {data_label} cases found to process.")
                break
            
            print(f"Processing batch of {len(tasks)} {data_label} cases...")
            await asyncio.gather(*tasks)
            total_processed += len(tasks)
            print(f"Batch completed. Total processed: {len(processed_flares[progress_key]) + total_processed}/{num_samples}.")
            
            print(f"Batch progress: {len(processed_flares[progress_key]) + total_processed}/{num_samples} completed.")
            
            # Add a small delay between batches to be nice to the server
            if len(processed_flares[progress_key]) + total_processed < num_samples:
                remaining_batches = (num_samples - (len(processed_flares[progress_key]) + total_processed)) // BATCH_SIZE
                print(f"Waiting 7 seconds before next batch. Estimated remaining batches: {remaining_batches}")
                await asyncio.sleep(7)
        
        return total_processed

    # Simplified execution flow
    if GET_FLARE_DATA_FIRST:
        await process_data(m_x_flares, NUM_PREFLARE_SAMPLES, "preflare", "flare")
        await process_data(non_m_x_flares, NUM_QUIET_SAMPLES, "quiet", "quiet")
    else:
        await process_data(non_m_x_flares, NUM_QUIET_SAMPLES, "quiet", "quiet")
        await process_data(m_x_flares, NUM_PREFLARE_SAMPLES, "preflare", "flare")

    print(f"\n--- Finished processing ---")
    print(f"Total failed downloads: {len(failed_downloads)}")

if __name__ == "__main__":
    asyncio.run(main())