import os
import json
import shutil

def cleanup_empty_directories():
    """Clean up empty directories and remove them from download_progress.json"""
    
    data_dir = "sharp_cnn_lstm_data"
    progress_file = "download_progress.json"
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist")
        return
    
    # Load progress file
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress = json.load(f)
    else:
        progress = {"preflare": [], "quiet": []}
    
    removed_dirs = []
    cleaned_progress_keys = []
    
    # Look through all directories in sharp_cnn_lstm_data
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        
        # Skip if not a directory
        if not os.path.isdir(item_path):
            continue
            
        # Check if it's a flare_case or quiet_case directory
        if not (item.startswith("flare_case_") or item.startswith("quiet_case_")):
            continue
            
        # Check if directory is empty or has no files (only empty subdirs)
        is_empty = True
        for root, dirs, files in os.walk(item_path):
            if files:  # If any files found, it's not empty
                is_empty = False
                break
        
        if is_empty:
            print(f"Found empty directory: {item}")
            
            # Extract the key for progress file
            # Format: flare_case_{harpnum}_NOAA_{noaa}_flare_{flare_id}
            # or: quiet_case_{harpnum}_NOAA_{noaa}_quiet_{quiet_id}
            try:
                if item.startswith("flare_case_"):
                    # Extract NOAA and flare_id from flare_case_{harpnum}_NOAA_{noaa}_flare_{flare_id}
                    parts = item.split("_")
                    noaa_idx = parts.index("NOAA")
                    flare_idx = parts.index("flare")
                    noaa = parts[noaa_idx + 1]
                    flare_id = "_".join(parts[flare_idx + 1:])
                    progress_key = f"{noaa}_{flare_id}"
                    
                    # Remove from preflare progress
                    if progress_key in progress["preflare"]:
                        progress["preflare"].remove(progress_key)
                        cleaned_progress_keys.append(f"preflare: {progress_key}")
                        
                elif item.startswith("quiet_case_"):
                    # Extract NOAA and quiet_id from quiet_case_{harpnum}_NOAA_{noaa}_quiet_{quiet_id}
                    parts = item.split("_")
                    noaa_idx = parts.index("NOAA")
                    quiet_idx = parts.index("quiet")
                    noaa = parts[noaa_idx + 1]
                    quiet_id = "_".join(parts[quiet_idx + 1:])
                    progress_key = f"{noaa}_{quiet_id}"
                    
                    # Remove from quiet progress
                    if progress_key in progress["quiet"]:
                        progress["quiet"].remove(progress_key)
                        cleaned_progress_keys.append(f"quiet: {progress_key}")
                        
            except (ValueError, IndexError):
                print(f"Could not parse directory name: {item}")
            
            # Remove the empty directory
            shutil.rmtree(item_path)
            removed_dirs.append(item)
            print(f"Removed empty directory: {item}")
    
    # Save updated progress file if changes were made
    if cleaned_progress_keys:
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)
        print(f"\nUpdated {progress_file}")
    
    # Summary
    print(f"\n=== CLEANUP SUMMARY ===")
    print(f"Removed {len(removed_dirs)} empty directories")
    print(f"Cleaned {len(cleaned_progress_keys)} progress entries")
    
    if removed_dirs:
        print("\nRemoved directories:")
        for dir_name in removed_dirs:
            print(f"  - {dir_name}")
    
    if cleaned_progress_keys:
        print("\nCleaned progress entries:")
        for key in cleaned_progress_keys:
            print(f"  - {key}")

if __name__ == "__main__":
    cleanup_empty_directories()
