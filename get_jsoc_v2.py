import os
import pandas as pd
from dotenv import load_dotenv
from sunpy.net import Fido, attrs as a
from concurrent.futures import ThreadPoolExecutor, as_completed
import astropy.units as u
import json
import random

load_dotenv()
# SEGMENTS = ["magnetogram", "continuum", "Dopplergram"]
GET_FLARE_DATA_FIRST = True

PROGRESS_FILE = "download_progress.json"

def download_sharp_time_series(harpnum, time_points, base_dir="sharp_cnn_lstm_data", single_file_per_step=False): 
    # Check if this is a specific case directory that might have been interrupted
    if os.path.exists(base_dir) and os.listdir(base_dir):
        import shutil
        print(f"[INFO] Found existing case directory {base_dir}, cleaning up and starting fresh")
        shutil.rmtree(base_dir)
    
    jsoc_email = os.getenv("EMAIL")
    if not jsoc_email:
        raise Exception("EMAIL environment variable not set.")
    time_objects = [pd.to_datetime(ts) for ts in time_points]
    start_time, end_time = min(time_objects), max(time_objects)
    try:
        print(f"[INFO] Querying JSOC for HARPNUM={harpnum} from {start_time} to {end_time} ...")
        result = Fido.search(
            a.Time(start_time.strftime('%Y-%m-%d %H:%M:%S'), end_time.strftime('%Y-%m-%d %H:%M:%S')),
            a.jsoc.Series("hmi.sharp_cea_720s"),
            a.jsoc.PrimeKey("HARPNUM", int(harpnum)),
            a.jsoc.Notify(jsoc_email),
            a.jsoc.Segment("Bp") &
            a.jsoc.Segment("Bt") &
            a.jsoc.Segment("Br") &
            a.jsoc.Segment("continuum"),
        )
        if len(result) == 0:
            print(f"[WARN] No data found for the time range {start_time} to {end_time} (HARPNUM={harpnum})")
            return False
        os.makedirs(base_dir, exist_ok=True)
        print(f"[INFO] Downloading files to {base_dir} ... (this may take a while)")
        files = Fido.fetch(result, path=base_dir, progress=False)  # Disable Fido's progress bar
        print(f"[INFO] Downloaded {len(files)} files for HARPNUM={harpnum}.")
        import re
        from datetime import datetime
        
        # Only create subdirectories if we have files
        if len(files) > 0:
            for i in range(len(time_objects)):
                os.makedirs(os.path.join(base_dir, f"timestep_{i+1:02d}"), exist_ok=True)
                
        for file_path in files:
            filename = os.path.basename(file_path)
            m = re.search(r'\.(\d{8}_\d{6})_TAI\.', filename)
            if m:
                file_time = datetime.strptime(m.group(1), '%Y%m%d_%H%M%S')
                best_timestep = min(range(len(time_objects)), key=lambda i: abs((file_time - time_objects[i].to_pydatetime()).total_seconds()))
                new_path = os.path.join(base_dir, f"timestep_{best_timestep+1:02d}", filename)
            else:
                new_path = os.path.join(base_dir, "timestep_01", filename)
            if file_path != new_path:
                os.rename(file_path, new_path)
        return len(files) > 0
    except Exception as e:
        print(f"[ERROR] Error downloading time series for HARPNUM={harpnum}: {e}")
        return False

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

def main():
    # Load progress file
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            processed_flares = json.load(f)
    else:
        processed_flares = {"preflare": [], "quiet": []}

    # --- Load and prepare your flare data (no changes here) ---
    FLARE_CSV = 'goes_flares.csv'
    if not os.path.exists(FLARE_CSV):
        print(f"Flare catalog '{FLARE_CSV}' not found.")
        return
    df = pd.read_csv(FLARE_CSV).dropna(subset=['noaa_active_region', 'goes_class'])
    df['noaa_active_region'] = df['noaa_active_region'].astype(int)
    df = df[pd.to_datetime(df['peak_time']) >= pd.Timestamp('2010-05-01')]
    m_x_flares = df[df['goes_class'].str.startswith(('M', 'X'))]
    non_m_x_flares = df[~df['goes_class'].str.startswith(('M', 'X'))]
    non_m_x_flares = non_m_x_flares.sample(frac=1, random_state=42).reset_index(drop=True)
    
    NUM_PREFLARE_SAMPLES, NUM_QUIET_SAMPLES = 1148, 1148
    TIME_STEPS, HOURS_BETWEEN_STEPS = 6, 1
    PREDICTION_HORIZON = 12
    SINGLE_FILE_PER_STEP = False

    # Set the number of parallel downloads. 3 is a good starting point.
    MAX_WORKERS = 3

    def process_cases(case_type, dataframe, num_samples_to_get):
        """
        Generic function to process flare or quiet cases in parallel, with improved logging and progress.
        """
        print(f"\n--- Processing {case_type} cases ---")
        jobs_to_run = []
        for row in dataframe.itertuples():
            if len(jobs_to_run) + len(processed_flares[case_type]) >= num_samples_to_get:
                break
            peak_time = pd.to_datetime(row.peak_time)
            noaa = int(row.noaa_active_region)
            case_id_time = peak_time.strftime('%Y%m%d_%H%M')
            case_key = f"{noaa}_{case_id_time}"
            if case_key in processed_flares[case_type]:
                continue # Skip already processed cases
            harpnum = get_harpnum_for_noaa(str(noaa))
            if not harpnum:
                continue
            flare_minus_12h = peak_time - pd.Timedelta(hours=PREDICTION_HORIZON)
            time_points = [(flare_minus_12h - pd.Timedelta(hours=(TIME_STEPS - t))) for t in range(1, TIME_STEPS + 1)]
            time_points_str = [tp.strftime('%Y-%m-%d %H:%M:%S') for tp in time_points]
            base_dir = f"sharp_cnn_lstm_data/{'flare' if case_type == 'preflare' else 'quiet'}_case_{harpnum}_NOAA_{noaa}_{case_id_time}"
            jobs_to_run.append({
                "case_key": case_key,
                "args": [harpnum, time_points_str, base_dir, SINGLE_FILE_PER_STEP],
                "desc": f"NOAA {noaa} | HARPNUM {harpnum} | {case_id_time}"
            })
        total_jobs = len(jobs_to_run)
        print(f"[INFO] Found {total_jobs} new {case_type} cases to process.")
        if not jobs_to_run:
            return
        completed = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_job = {executor.submit(download_sharp_time_series, *job["args"]): job for job in jobs_to_run}
            for idx, future in enumerate(as_completed(future_to_job), 1):
                job = future_to_job[future]
                case_key = job["case_key"]
                desc = job["desc"]
                print(f"[PROGRESS] ({idx}/{total_jobs}) Starting download for case: {desc}")
                try:
                    success = future.result()
                    if success:
                        print(f"[SUCCESS] ({idx}/{total_jobs}) Completed download for case {case_key}")
                        processed_flares[case_type].append(case_key)
                        completed += 1
                        with open(PROGRESS_FILE, "w") as f:
                            json.dump(processed_flares, f)
                    else:
                        print(f"[FAIL] ({idx}/{total_jobs}) Download failed for case {case_key}, but no error was raised.")
                        failed += 1
                except Exception as e:
                    print(f"[ERROR] ({idx}/{total_jobs}) Error processing case {case_key}: {e}")
                    failed += 1
                print(f"[SUMMARY] {completed} completed, {failed} failed, {total_jobs-idx} remaining.")
        print(f"[BATCH SUMMARY] {completed} completed, {failed} failed, {total_jobs} attempted for {case_type} cases.")

    # Process flare and quiet data based on the flag
    if GET_FLARE_DATA_FIRST:
        process_cases("preflare", m_x_flares, NUM_PREFLARE_SAMPLES)
        process_cases("quiet", non_m_x_flares, NUM_QUIET_SAMPLES)
    else:
        process_cases("quiet", non_m_x_flares, NUM_QUIET_SAMPLES)
        process_cases("preflare", m_x_flares, NUM_PREFLARE_SAMPLES)

    print(f"\n--- Finished all processing ---")


if __name__ == "__main__":
    main()