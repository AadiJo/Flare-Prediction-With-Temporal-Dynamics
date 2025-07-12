import ftplib
import os
import tarfile

def list_year_directories(ftp_host='ftp.swpc.noaa.gov', base_dir='/pub/warehouse'):
    ftp = ftplib.FTP(ftp_host)
    ftp.login()
    ftp.cwd(base_dir)
    dirs = []
    ftp.retrlines('LIST', lambda line: dirs.append(line.split()[-1]))
    ftp.quit()
    # Filter only directories by checking name format YYYY (4 digits)
    year_dirs = [d for d in dirs if d.isdigit() and len(d) == 4]
    return year_dirs

def list_archives_in_year(ftp_host, base_dir, year):
    ftp = ftplib.FTP(ftp_host)
    ftp.login()
    ftp.cwd(f"{base_dir}/{year}")
    files = ftp.nlst()
    ftp.quit()
    archives = [f for f in files if f.endswith('_events.tar.gz')]
    return archives

def download_file(ftp_host, remote_path, local_path):
    ftp = ftplib.FTP(ftp_host)
    ftp.login()
    with open(local_path, 'wb') as f:
        print(f"Downloading {remote_path} ...")
        ftp.retrbinary(f'RETR {remote_path}', f.write)
    ftp.quit()
    print(f"Downloaded to {local_path}")

def extract_archive(archive_path, extract_to):
    print(f"Extracting {archive_path} to {extract_to} ...")
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)
    print("Extraction complete.")

def main():
    ftp_host = 'ftp.swpc.noaa.gov'
    base_dir = '/pub/warehouse'

    print("Fetching available years from NOAA FTP...")
    years = list_year_directories(ftp_host, base_dir)
    if not years:
        print("No year directories found.")
        return

    print("Available years:")
    for i, year in enumerate(years):
        print(f"{i+1}. {year}")

    choice = input(f"Select a year to list event archives (1-{len(years)}): ")
    try:
        choice_idx = int(choice) - 1
        if choice_idx < 0 or choice_idx >= len(years):
            print("Invalid selection.")
            return
    except ValueError:
        print("Invalid input.")
        return

    selected_year = years[choice_idx]

    print(f"Listing event archives in year {selected_year}...")
    archives = list_archives_in_year(ftp_host, base_dir, selected_year)
    if not archives:
        print(f"No event archives found for year {selected_year}.")
        return

    print("Available event archives:")
    for i, archive in enumerate(archives):
        print(f"{i+1}. {archive}")

    choice2 = input(f"Select an archive to download (1-{len(archives)}): ")
    try:
        choice2_idx = int(choice2) - 1
        if choice2_idx < 0 or choice2_idx >= len(archives):
            print("Invalid selection.")
            return
    except ValueError:
        print("Invalid input.")
        return

    selected_archive = archives[choice2_idx]
    local_filename = selected_archive
    remote_path = f"{base_dir}/{selected_year}/{selected_archive}"
    extract_folder = os.path.join("data")

    # Download
    download_file(ftp_host, remote_path, local_filename)

    # Extract
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)
    extract_archive(local_filename, extract_folder)

    # Clean up - delete the downloaded tar.gz file
    os.remove(local_filename)
    print(f"Deleted downloaded file: {local_filename}")

    print(f"All files extracted to folder: {extract_folder}")

if __name__ == "__main__":
    main()
