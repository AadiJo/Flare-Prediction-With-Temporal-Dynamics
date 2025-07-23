import os
import pandas as pd
from dotenv import load_dotenv
from sunpy.net import Fido, attrs as a
import astropy.units as u
import json
import random

load_dotenv()
GET_FLARE_DATA_FIRST = True

PROGRESS_FILE = "download_progress.json"

def download_sharp_time_series(harpnum, time_points, base_dir="sharp_cnn_lstm_data", single_file_per_step=False): 
    # Check if this is a specific case directory that might have been interrupted
    if os.path.exists(base_dir) and ("flare_case_" in base_dir or "quiet_case_" in base_dir):
        import shutil
        print(f"Found existing case directory {base_dir}, cleaning up and starting fresh")
        shutil.rmtree(base_dir)
    
    jsoc_email = os.getenv("EMAIL")
    if not jsoc_email:
        raise Exception("EMAIL environment variable not set.")
    time_objects = [pd.to_datetime(ts) for ts in time_points]
    start_time, end_time = min(time_objects), max(time_objects)  # No extra padding needed
    try:
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
            print(f"No data found for the time range")
            return False
        os.makedirs(base_dir, exist_ok=True)
        files = Fido.fetch(result, path=base_dir)
        import re
        from datetime import datetime
        for i in range(len(time_objects)):
            os.makedirs(os.path.join(base_dir, f"timestep_{i+1:02d}"), exist_ok=True)
        for file_path in files:
            filename = os.path.basename(file_path)
            m = re.search(r'\.(\d{8}_\d{6})_TAI\.', filename) # Match the timestamp in the filename using CSUIL regex ;)
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
        print(f"Error downloading time series: {e}")
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
    # Load progress file if it exists
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            processed_flares = json.load(f)
    else:
        processed_flares = {
            "preflare": [],
            "quiet": []
        }

    FLARE_CSV = 'goes_flares.csv'
    if not os.path.exists(FLARE_CSV):
        print(f"Flare catalog '{FLARE_CSV}' not found.")
        return
    df = pd.read_csv(FLARE_CSV).dropna(subset=['noaa_active_region', 'goes_class'])
    df['noaa_active_region'] = df['noaa_active_region'].astype(int)
    df = df[pd.to_datetime(df['peak_time']) >= pd.Timestamp('2010-05-01')]

    # Filter for M and X class flares for pre-flare data
    m_x_flares = df[df['goes_class'].str.startswith(('M', 'X'))]

    # Filter for non-M/X class flares for quiet data (with end date filter and randomization)
    QUIET_END_DATE = '2024-05-29'  # End date for random sampling of quiet data
    non_m_x_flares = df[~df['goes_class'].str.startswith(('M', 'X'))]
    non_m_x_flares = non_m_x_flares[pd.to_datetime(non_m_x_flares['peak_time']) <= pd.Timestamp(QUIET_END_DATE)]
    # Randomize the order of non-M/X flares
    non_m_x_flares = non_m_x_flares.sample(frac=1, random_state=42).reset_index(drop=True)

    # Dynamically extract all NOAA active regions from the dataset
    valid_noaa_list = m_x_flares['noaa_active_region'].unique()

    NUM_PREFLARE_SAMPLES, NUM_QUIET_SAMPLES = 1148, 1148
    TIME_STEPS, HOURS_BETWEEN_STEPS = 6, 1  # Add hours between steps later if needed

    PREDICTION_HORIZON = 12  # 12 hours before flare
    SINGLE_FILE_PER_STEP = False

    def process_preflare_data():
        """Process pre-flare (positive) samples"""
        print(f"Processing {len(m_x_flares)} pre-flare (positive) samples...")
        preflare_df = m_x_flares.sort_values(by='peak_time')
        preflare_samples_processed = 0
        for i, row in enumerate(preflare_df.itertuples(), 1):
            if preflare_samples_processed >= NUM_PREFLARE_SAMPLES:
                break
            print(f"Processing flare case {i} of {len(preflare_df)} - {preflare_samples_processed + 1} / {NUM_PREFLARE_SAMPLES}")

            peak_time = pd.to_datetime(row.peak_time)
            noaa = int(row.noaa_active_region)
            flare_id = peak_time.strftime('%Y%m%d_%H%M')
            
            # Skip if we've already processed this flare
            flare_key = f"{noaa}_{flare_id}"
            if flare_key in processed_flares["preflare"]:
                print(f"Skipping already processed flare case {flare_id}")
                preflare_samples_processed += 1
                continue
                
            # Calculate 6 hourly time points ending 12 hours before flare
            flare_minus_12h = peak_time - pd.Timedelta(hours=PREDICTION_HORIZON)
            time_points = [(flare_minus_12h - pd.Timedelta(hours=(TIME_STEPS - t))) for t in range(1, TIME_STEPS + 1)]
            time_points = [tp.strftime('%Y-%m-%d %H:%M:%S') for tp in time_points]
            
            harpnum = get_harpnum_for_noaa(str(noaa))
            if not harpnum:
                continue
            base_dir = f"sharp_cnn_lstm_data/flare_case_{harpnum}_NOAA_{noaa}_flare_{flare_id}"
            try:
                success = download_sharp_time_series(harpnum, time_points, base_dir=base_dir, single_file_per_step=SINGLE_FILE_PER_STEP)
                if success:
                    preflare_samples_processed += 1
                    processed_flares["preflare"].append(flare_key)
                    # Save progress after each successful download
                    with open(PROGRESS_FILE, "w") as f:
                        json.dump(processed_flares, f)
                    print(f"✓ Completed time series for flare case {flare_id}")
            except Exception as e:
                print(f"✗ Error processing time series: {e}")
        return preflare_samples_processed

    def process_quiet_data():
        """Process quiet (non-pre-flare) samples using non-M/X class flares"""
        total_quiet_existing = len(processed_flares["quiet"])
        quiet_samples_needed = max(0, NUM_QUIET_SAMPLES - total_quiet_existing)
        print(f"\nProcessing quiet samples: {total_quiet_existing} existing + {quiet_samples_needed} needed = {NUM_QUIET_SAMPLES} total")
        
        quiet_samples_processed = 0
        for i, row in enumerate(non_m_x_flares.itertuples(), 1):
            if quiet_samples_processed >= quiet_samples_needed:
                break
            print(f"Processing flare case {i} of {len(non_m_x_flares)} - {quiet_samples_processed + 1} / {quiet_samples_needed}")

            peak_time = pd.to_datetime(row.peak_time)
            noaa = int(row.noaa_active_region)
            quiet_id = peak_time.strftime('%Y%m%d_%H%M')
            
            # Skip if we've already processed this quiet case
            quiet_key = f"{noaa}_{quiet_id}"
            if quiet_key in processed_flares["quiet"]:
                print(f"Skipping already processed quiet case {quiet_id}")
                continue
            
            # Same 6-hour window ending 12 hours before the flare peak for consistency
            flare_minus_12h = peak_time - pd.Timedelta(hours=PREDICTION_HORIZON)
            time_points = [(flare_minus_12h - pd.Timedelta(hours=(TIME_STEPS - t))) for t in range(1, TIME_STEPS + 1)]
            time_points = [tp.strftime('%Y-%m-%d %H:%M:%S') for tp in time_points]
            
            harpnum = get_harpnum_for_noaa(str(noaa))
            if not harpnum:
                continue
            base_dir = f"sharp_cnn_lstm_data/quiet_case_{harpnum}_NOAA_{noaa}_quiet_{quiet_id}"
            try:
                success = download_sharp_time_series(harpnum, time_points, base_dir=base_dir, single_file_per_step=SINGLE_FILE_PER_STEP)
                if success:
                    quiet_samples_processed += 1
                    processed_flares["quiet"].append(quiet_key)
                    # Save progress after each successful download
                    with open(PROGRESS_FILE, "w") as f:
                        json.dump(processed_flares, f)
                    print(f"✓ Completed time series for quiet case {quiet_id}")
            except Exception as e:
                print(f"✗ Error processing time series: {e}")
        return quiet_samples_processed

    # Process data in order based on GET_FLARE_DATA_FIRST
    if GET_FLARE_DATA_FIRST:
        preflare_processed = process_preflare_data()
        quiet_processed = process_quiet_data()
    else:
        quiet_processed = process_quiet_data()
        preflare_processed = process_preflare_data()

    print(f"\n--- Finished processing {preflare_processed} pre-flare and {len(processed_flares['quiet'])} quiet samples ---")

if __name__ == "__main__":
    main()

