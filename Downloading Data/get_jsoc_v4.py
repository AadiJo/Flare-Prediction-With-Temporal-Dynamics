import os
import pandas as pd
from dotenv import load_dotenv
import json
import re
from datetime import datetime
import shutil
import asyncio
import drms
import aiohttp
import aiofiles
import glob

load_dotenv()
GET_FLARE_DATA_FIRST = False
PROGRESS_FILE = "download_progress_async.json"
FAILED_DOWNLOADS_FILE = "failed_downloads.json"
MIN_DATE = '2023-06-18'

# --- Main Async Function Stuffs ---

async def download_file(session, url, path, max_retries=3):
    """Asynchronously downloads a single file with exponential backoff."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=300)
            async with session.get(url, timeout=timeout) as response:
                response.raise_for_status()
                async with aiofiles.open(path, 'wb') as f:
                    await f.write(await response.read())
                return True
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            is_retriable = isinstance(e, asyncio.TimeoutError) or \
                           (isinstance(e, aiohttp.ClientResponseError) and e.status in [429, 503, 502, 504])
            if attempt < max_retries - 1 and is_retriable:
                wait = 2 ** attempt
                print(f"Retriable error for {url} ({type(e).__name__}), retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"✗ Failed to download {url}: {e}")
                return False
    return False

async def _update_json_list(lock, item, item_list, filepath):
    """Safely appends an item to a list and writes it to a JSON file."""
    async with lock:
        if item not in item_list:
            item_list.append(item)
            with open(filepath, "w") as f:
                json.dump(item_list, f, indent=4)

async def download_sharp_time_series_async(client, harpnum, time_points_str, base_dir, flare_key, progress_dict, failed_downloads, lock):
    """Handles the entire pipeline for one time-series: DRMS query, download, and file organization."""
    try:
        if os.path.exists(base_dir): shutil.rmtree(base_dir)

        # 1. JSOC Export Request with Retry
        time_objects = [pd.to_datetime(ts) for ts in time_points_str]
        start_time, end_time = min(time_objects), max(time_objects)
        query = f"hmi.sharp_cea_720s[{int(harpnum)}][{start_time.strftime('%Y.%m.%d_%H:%M:%S_TAI')}-{end_time.strftime('%Y.%m.%d_%H:%M:%S_TAI@1h')}]{{Bp, Bt, Br, continuum}}"
        
        export_request = None
        for attempt in range(5):
            try:
                loop = asyncio.get_event_loop()
                export_request = await loop.run_in_executor(None, lambda: client.export(query, method='url', protocol='fits'))
                await loop.run_in_executor(None, export_request.wait)
                if export_request.status == 0 and not export_request.urls.empty: break
                if export_request.status != 7: raise RuntimeError(f"Export failed with status {export_request.status}")
            except Exception as e:
                if "pending export requests" not in str(e) and (not hasattr(export_request, 'status') or export_request.status != 7):
                    raise e
            
            if attempt < 4:
                await asyncio.sleep(30 + (attempt * 15))
            else:
                raise RuntimeError("JSOC rate limited after 5 attempts.")

        # 2. Asynchronous File Downloads
        os.makedirs(base_dir, exist_ok=True)
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit_per_host=5)) as session:
            tasks = []
            for _, row in export_request.urls.iterrows():
                url = row['url'].replace('http://jsoc.stanford.edu', 'https://jsoc1.stanford.edu')
                if not url.startswith('http'): url = f"https://jsoc1.stanford.edu{url}"
                filepath = os.path.join(base_dir, os.path.basename(row['filename']))
                tasks.append(download_file(session, url, filepath))
            if not any(await asyncio.gather(*tasks)): raise RuntimeError("All file downloads failed.")

        # 3. Organize Files
        files_in_dir = [f for f in os.listdir(base_dir) if f.endswith('.fits')]
        if not files_in_dir: raise RuntimeError("No FITS files found after download.")
        
        for filename in files_in_dir:
            match = re.search(r'\.(\d{8}_\d{6})_TAI\.', filename)
            if match:
                file_time = datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
                best_idx = min(range(len(time_objects)), key=lambda i: abs((file_time - time_objects[i].to_pydatetime()).total_seconds()))
                timestep_dir = os.path.join(base_dir, f"timestep_{best_idx+1:02d}")
                os.makedirs(timestep_dir, exist_ok=True)
                os.rename(os.path.join(base_dir, filename), os.path.join(timestep_dir, filename))

        # 4. Update Progress
        progress_key = "preflare" if "flare" in base_dir else "quiet"
        async with lock:
            if flare_key not in progress_dict[progress_key]:
                progress_dict[progress_key].append(flare_key)
                with open(PROGRESS_FILE, "w") as f: json.dump(progress_dict, f, indent=4)
        
        print(f"✓ Completed time series for {flare_key}")
        return True

    except Exception as e:
        print(f"✗ Error processing time series for {flare_key}: {e}")
        await _update_json_list(lock, flare_key, failed_downloads, FAILED_DOWNLOADS_FILE)
        return False

# --- Helper Functions ---
def load_noaa_to_harpnum_map_local(filepath="HARP_Mapping.txt"):
    noaa_to_harp = {}
    with open(filepath, "r") as f:
        for line in f.read().strip().splitlines()[1:]:
            if not line.strip(): continue
            harpnum, noaa_ars = line.split(maxsplit=1)
            for noaa in noaa_ars.split(','):
                if noaa.strip().isdigit():
                    noaa_to_harp.setdefault(noaa.strip(), []).append(harpnum)
    return noaa_to_harp

def get_harpnum_for_noaa(noaa, noaa_to_harp_map):
    return noaa_to_harp_map.get(str(noaa), [None])[0]

# --- Main Execution ---
async def main():
    jsoc_email = os.getenv("EMAIL")
    if not jsoc_email: raise Exception("EMAIL environment variable not set.")
    client = drms.Client(email=jsoc_email)

    try:
        with open(PROGRESS_FILE, "r") as f: processed_flares = json.load(f)
    except FileNotFoundError:
        processed_flares = {"preflare": [], "quiet": []}

    try:
        with open(FAILED_DOWNLOADS_FILE, "r") as f: failed_downloads = json.load(f)
    except FileNotFoundError:
        failed_downloads = []

    try:
        noaa_to_harp = load_noaa_to_harpnum_map_local()
    except Exception:
        import requests
        resp = requests.get("http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa_ars/all_harps_with_noaa_ars.txt")
        resp.raise_for_status()
        noaa_to_harp = {}
        for line in resp.text.strip().splitlines()[1:]:
            harpnum, noaa_ars = line.split(maxsplit=1)
            for noaa in noaa_ars.split(','):
                if noaa.strip().isdigit():
                    noaa_to_harp.setdefault(noaa.strip(), []).append(harpnum)

    df = pd.read_csv('goes_flares.csv').dropna(subset=['noaa_active_region', 'goes_class'])
    df['noaa_active_region'] = df['noaa_active_region'].astype(int)
    df = df[pd.to_datetime(df['start_time']) >= pd.Timestamp(MIN_DATE)]
    
    m_x_flares = df[df['goes_class'].str.startswith(('M', 'X'))]
    non_m_x_flares = df[~df['goes_class'].str.startswith(('M', 'X')) & (pd.to_datetime(df['start_time']) <= pd.Timestamp('2025-07-23'))]
    non_m_x_flares = non_m_x_flares.sample(frac=1, random_state=42).reset_index(drop=True)

    NUM_PREFLARE_SAMPLES, NUM_QUIET_SAMPLES = 0, 1196
    TIME_STEPS, PREDICTION_HORIZON, BATCH_SIZE = 12, 12, 5

    async def process_data(data_df, num_samples, progress_key, data_label):
        print(f"\nProcessing {num_samples} {data_label} samples...")
        lock = asyncio.Lock()
        
        while len(processed_flares[progress_key]) < num_samples:
            tasks, batch_keys = [], []
            for row in data_df.itertuples():
                if len(processed_flares[progress_key]) + len(tasks) >= num_samples: break
                
                item_key = f"{int(row.noaa_active_region)}_{pd.to_datetime(row.peak_time).strftime('%Y%m%d_%H%M')}"
                if item_key in processed_flares[progress_key] or item_key in failed_downloads: continue

                harpnum = get_harpnum_for_noaa(row.noaa_active_region, noaa_to_harp)
                if not harpnum: continue

                base_dir = f"D:/async_sharp/{data_label}_case_{harpnum}_NOAA_{int(row.noaa_active_region)}_{pd.to_datetime(row.peak_time).strftime('%Y%m%d_%H%M')}"
                if os.path.exists(base_dir) and glob.glob(os.path.join(base_dir, '**', '*.fits'), recursive=True):
                    print(f"Found existing data for {item_key}, skipping...")
                    await _update_json_list(lock, item_key, processed_flares[progress_key], PROGRESS_FILE)
                    continue

                peak_time = pd.to_datetime(row.peak_time)
                flare_minus_12h = peak_time - pd.Timedelta(hours=PREDICTION_HORIZON)
                time_points = [(flare_minus_12h - pd.Timedelta(hours=(TIME_STEPS - t))).strftime('%Y-%m-%d %H:%M:%S') for t in range(1, TIME_STEPS + 1)]
                
                tasks.append(download_sharp_time_series_async(client, harpnum, time_points, base_dir, item_key, processed_flares, failed_downloads, lock))
                batch_keys.append(item_key)
                if len(tasks) >= BATCH_SIZE: break
            
            if not tasks:
                print(f"No more valid {data_label} cases to process.")
                break
            
            print(f"Processing batch of {len(tasks)} {data_label} cases...")
            await asyncio.gather(*tasks)
            print(f"Overall progress: {len(processed_flares[progress_key])}/{num_samples} completed.")
            if len(processed_flares[progress_key]) < num_samples: await asyncio.sleep(7)

    data_sets = [
        (m_x_flares, NUM_PREFLARE_SAMPLES, "preflare", "flare"),
        (non_m_x_flares, NUM_QUIET_SAMPLES, "quiet", "quiet")
    ]
    if not GET_FLARE_DATA_FIRST: data_sets.reverse()
    
    for df, num, key, label in data_sets:
        if num > 0: await process_data(df, num, key, label)

    print(f"\n--- Finished ---\nPre-flare samples: {len(processed_flares['preflare'])}/{NUM_PREFLARE_SAMPLES}\nQuiet samples: {len(processed_flares['quiet'])}/{NUM_QUIET_SAMPLES}")
    if failed_downloads: print(f"Total failed downloads: {len(failed_downloads)}")

if __name__ == "__main__":
    asyncio.run(main())