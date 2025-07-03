import os
import time
import requests
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

JSOC_BASE = "http://jsoc.stanford.edu/cgi-bin/ajax/jsocextfetch"
DOWNLOAD_BASE = "http://jsoc1.stanford.edu"

# Segments to download
SEGMENTS = ["azimuth", "field", "disambig", "inclination"]

def build_recordset(timestamp):
    return f"hmi.B_720s[{timestamp}]"

def submit_request(recordset):
    email = os.getenv('EMAIL')
    if not email:
        raise Exception("EMAIL environment variable not set. Please check your .env file.")
    
    payload = {
        "op": "exp_request",
        "ds": recordset,
        "sizeratio": "1",
        "process": "n=0|no_op",
        "requestor": "",
        "notify": email,
        "method": "url",
        "filenamefmt": "hmi.B_720s.{T_REC:A}.{segment}",
        "format": "json",
        "protocol": "FITS,compress Rice",
        "dbhost": "hmidb2",
    }

    response = requests.post(JSOC_BASE, data=payload)
    response.raise_for_status()
    result = response.json()

    if result.get("status") != 2:
        raise Exception(f"Request submission failed: {result}")

    return result["requestid"]

def wait_for_data(requestid):
    status_payload = {
        "op": "exp_status",
        "requestid": requestid,
        "dbhost": "hmidb2",
    }

    while True:
        response = requests.post(JSOC_BASE, data=status_payload)
        response.raise_for_status()
        status = response.json()

        if status.get("status") == "0":
            return status
        else:
            print("Waiting for export to complete...")
            time.sleep(5)

def download_files(status, segments=SEGMENTS, base_dir="data"):
    data = status["data"]
    dir_path = status["dir"]

    downloaded = 0
    for entry in data:
        filename = entry["filename"]
        if any(seg in filename for seg in segments):
            # Extract file header (e.g., "hmi.B_720s.20221215_163600_TAI" from "hmi.B_720s.20221215_163600_TAI.azimuth.fits")
            file_header = filename.split('.')[0] + '.' + filename.split('.')[1] + '.' + filename.split('.')[2]
            
            # Create subdirectory based on file header
            out_dir = os.path.join(base_dir, file_header)
            os.makedirs(out_dir, exist_ok=True)
            
            url = f"{DOWNLOAD_BASE}{dir_path}/{filename}"
            output_path = os.path.join(out_dir, filename)

            print(f"Downloading {filename} to {out_dir}...")
            r = requests.get(url, stream=True)
            r.raise_for_status()

            total_size = int(r.headers.get('content-length', 0))
            with open(output_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            downloaded += 1

    if downloaded == 0:
        print("No matching segment files were found.")
    else:
        print(f"Downloaded {downloaded} segment(s).")

def main():
    timestamps = [
        '2021.07.03_16:59:00_TAI',
        '2021.10.26_15:42:00_TAI',
        '2021.10.29_02:22:00_TAI',
        '2022.01.29_22:45:00_TAI',
        '2022.03.14_08:29:00_TAI',
        '2022.03.25_05:02:00_TAI',
        '2022.03.29_09:17:00_TAI',
        '2022.04.17_02:00:00_TAI',
        '2022.04.17_02:00:00_TAI',
        '2022.04.29_18:01:00_TAI',
        '2022.08.15_21:47:00_TAI',
        '2022.08.26_10:41:00_TAI',
        '2022.08.29_18:45:00_TAI',
        '2022.08.30_18:04:00_TAI',
        '2022.09.16_09:44:00_TAI',
        '2022.10.04_12:48:00_TAI',
        '2022.10.11_08:36:00_TAI',
        '2022.12.14_07:30:00_TAI'
    ]
    
    print(f"Processing {len(timestamps)} timestamps...")
    
    for i, timestamp in enumerate(timestamps, 1):
        print(f"\n--- Processing {i}/{len(timestamps)}: {timestamp} ---")
        recordset = build_recordset(timestamp)
        print(f"Using RecordSet: {recordset}")

        try:
            print("Submitting request to JSOC...")
            requestid = submit_request(recordset)

            print(f"Submitted. Request ID: {requestid}")
            print("Waiting for data to be ready...")
            status = wait_for_data(requestid)

            print("Downloading selected segment files...")
            download_files(status)
            print(f"✓ Completed {timestamp}")

        except Exception as e:
            print(f"✗ Error processing {timestamp}: {e}")
            continue  # Continue with the next timestamp even if one fails
    
    print(f"\n--- Finished processing all {len(timestamps)} timestamps ---")

if __name__ == "__main__":
    main()
