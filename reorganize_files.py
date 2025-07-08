import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import pandas as pd

def reorganize_sharp_files(base_dir="sharp_cnn_lstm_data"):
    """
    Reorganize SHARP files to separate different flare sequences while preserving temporal order.
    Also ensures quiet cases are properly organized, including files outside existing timestep folders.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Base directory {base_dir} does not exist!")
        return

    # Find all case folders (flare_case_* and quiet_case_*)
    case_folders = []
    for folder in base_path.iterdir():
        if folder.is_dir() and (folder.name.startswith('flare_case_') or folder.name.startswith('quiet_case_')):
            case_folders.append(folder)
    
    if len(case_folders) == 0:
        print("No case folders found!")
        return
    
    print(f"Found {len(case_folders)} case folders to process")
    
    for case_folder in case_folders:
        print(f"\nProcessing case folder: {case_folder.name}")
        
        # Handle quiet cases
        if case_folder.name.startswith('quiet_case_'):
            print(f"  Processing quiet case")
            
            # Collect all files in the quiet case folder, including those outside timestep folders
            all_files = []
            for file in case_folder.rglob("*.fits"):
                all_files.append(file)
            
            print(f"  Found {len(all_files)} total .fits files")
            
            if not all_files:
                print(f"  No .fits files found in {case_folder.name}")
                continue
            
            # Create timestep folders if they don't exist
            for i in range(1, 7):
                timestep_folder = case_folder / f"timestep_{i:02d}"
                timestep_folder.mkdir(exist_ok=True)
            
            # Extract timestamps from filenames and sort files chronologically
            files_with_timestamps = []
            
            for file in all_files:
                try:
                    filename = file.name
                    
                    # Extract timestamp from filename
                    timestamp_match = re.search(r'\.(\d{8}_\d{6})_TAI\.', filename)
                    if timestamp_match:
                        timestamp_str = timestamp_match.group(1)
                        file_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        files_with_timestamps.append((file, file_time))
                    else:
                        print(f"    Warning: Could not parse timestamp from {filename}, skipping")
                except Exception as e:
                    print(f"    Warning: Could not process file {file}: {e}")
            
            # Sort files by timestamp
            files_with_timestamps.sort(key=lambda x: x[1])
            
            if not files_with_timestamps:
                print(f"  No files with valid timestamps found in {case_folder.name}")
                continue
            
            print(f"  Files span from {files_with_timestamps[0][1]} to {files_with_timestamps[-1][1]}")
            
            # Distribute files evenly across 6 timesteps
            files_per_timestep = len(files_with_timestamps) // 6
            remainder = len(files_with_timestamps) % 6
            
            file_index = 0
            for timestep in range(1, 7):
                timestep_folder = case_folder / f"timestep_{timestep:02d}"
                
                files_in_this_timestep = files_per_timestep
                if timestep <= remainder:
                    files_in_this_timestep += 1
                
                moved_count = 0
                for _ in range(files_in_this_timestep):
                    if file_index < len(files_with_timestamps):
                        source_file, file_time = files_with_timestamps[file_index]
                        
                        if source_file.exists():
                            dest_file = timestep_folder / source_file.name
                            
                            if source_file != dest_file:
                                shutil.move(str(source_file), str(dest_file))
                            
                            moved_count += 1
                        
                        file_index += 1
                
                print(f"    Timestep {timestep:02d}: {moved_count} files")
            
            print(f"  ✓ Organized files in {case_folder.name}")
            continue
        
        # Collect all files from the case folder (including those in existing timestep folders)
        all_files = []
        
        # Find all .fits files recursively
        fits_files = list(case_folder.rglob("*.fits"))
        print(f"  Found {len(fits_files)} total .fits files")
        
        if not fits_files:
            print(f"  No .fits files found in {case_folder.name}")
            continue
        
        for fits_file in fits_files:
            filename = fits_file.name
            
            # Extract HARPNUM and timestamp from filename
            # Example: hmi.sharp_cea_720s.7115.20170828_090000_TAI.Bp.fits
            match = re.search(r'hmi\.sharp_cea_720s\.(\d+)\.(\d{8}_\d{6})_TAI\.', filename)
            
            if match:
                harpnum = match.group(1)
                timestamp_str = match.group(2)
                
                # Parse timestamp for sorting
                try:
                    file_datetime = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    
                    all_files.append({
                        'file': fits_file,
                        'timestamp': file_datetime,
                        'harpnum': harpnum,
                        'original_location': fits_file.parent.name
                    })
                    
                except ValueError as e:
                    print(f"    Warning: Could not parse timestamp from {filename}: {e}")
                    continue
            else:
                print(f"    Warning: Could not extract HARPNUM from {filename}")
                continue
        
        if not all_files:
            print(f"  No valid files found in {case_folder.name}")
            continue
        
        # Sort all files by timestamp
        all_files.sort(key=lambda x: x['timestamp'])
        
        print(f"  Files span from {all_files[0]['timestamp']} to {all_files[-1]['timestamp']}")
        
        # Detect separate flare sequences using time gaps
        flare_sequences = []
        current_sequence = []
        max_gap_hours = 24  # If gap > 24 hours, consider it a new flare sequence
        
        for i, file_info in enumerate(all_files):
            if not current_sequence:
                # Start first sequence
                current_sequence = [file_info]
            else:
                # Check time gap from last file in current sequence
                time_gap = (file_info['timestamp'] - current_sequence[-1]['timestamp']).total_seconds() / 3600
                
                if time_gap > max_gap_hours:
                    # Large gap detected - start new sequence
                    if current_sequence:
                        flare_sequences.append(current_sequence)
                    current_sequence = [file_info]
                else:
                    # Continue current sequence
                    current_sequence.append(file_info)
        
        # Don't forget the last sequence
        if current_sequence:
            flare_sequences.append(current_sequence)
        
        print(f"  Detected {len(flare_sequences)} separate flare sequences:")
        for i, sequence in enumerate(flare_sequences, 1):
            start_time = sequence[0]['timestamp']
            end_time = sequence[-1]['timestamp']
            duration = (end_time - start_time).total_seconds() / 3600
            print(f"    Sequence {i}: {len(sequence)} files, {duration:.1f}h span ({start_time} to {end_time})")
        
        # If only one sequence, just organize the existing folder properly
        if len(flare_sequences) == 1:
            print(f"  Only one flare sequence detected - organizing existing folder")
            sequence = flare_sequences[0]
            
            # Clear existing timestep folders if they exist
            existing_timestep_folders = [f for f in case_folder.iterdir() 
                                       if f.is_dir() and f.name.startswith('timestep_')]
            
            # Move files to temp if reorganizing
            if existing_timestep_folders:
                temp_dir = case_folder / "temp_reorganize"
                temp_dir.mkdir(exist_ok=True)
                
                for file_info in sequence:
                    if file_info['file'].parent.name.startswith('timestep_'):
                        temp_path = temp_dir / file_info['file'].name
                        if file_info['file'].exists():
                            shutil.move(str(file_info['file']), str(temp_path))
                            file_info['file'] = temp_path
                
                # Remove empty timestep folders
                for folder in existing_timestep_folders:
                    if not any(folder.iterdir()):
                        folder.rmdir()
            
            # Distribute files evenly across 6 timesteps
            files_per_timestep = len(sequence) // 6
            remainder = len(sequence) % 6
            
            file_index = 0
            for timestep in range(1, 7):
                timestep_folder = case_folder / f"timestep_{timestep:02d}"
                timestep_folder.mkdir(exist_ok=True)
                
                files_in_this_timestep = files_per_timestep
                if timestep <= remainder:
                    files_in_this_timestep += 1
                
                moved_count = 0
                for _ in range(files_in_this_timestep):
                    if file_index < len(sequence):
                        source_file = sequence[file_index]['file']
                        
                        if source_file.exists():
                            dest_file = timestep_folder / source_file.name
                            shutil.move(str(source_file), str(dest_file))
                            moved_count += 1
                        
                        file_index += 1
                
                print(f"    Timestep {timestep:02d}: {moved_count} files")
            
            # Clean up temp directory
            temp_dir = case_folder / "temp_reorganize"
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()
        
        else:
            # Multiple sequences - create separate folders for each
            print(f"  Creating separate folders for {len(flare_sequences)} flare sequences")
            
            for i, sequence in enumerate(flare_sequences, 1):
                harpnum = sequence[0]['harpnum']
                start_time = sequence[0]['timestamp']
                
                # Extract NOAA from original folder name
                noaa_match = re.search(r'NOAA_(\d+)', case_folder.name)
                noaa = noaa_match.group(1) if noaa_match else "unknown"
                
                # Create folder name with sequence number and start time
                start_time_str = start_time.strftime('%Y%m%d_%H%M%S')
                new_folder_name = f"flare_case_{i:03d}_{harpnum}_NOAA_{noaa}_{start_time_str}"
                new_folder_path = base_path / new_folder_name
                
                if new_folder_path.exists():
                    print(f"    Warning: Folder {new_folder_name} already exists, skipping")
                    continue
                
                new_folder_path.mkdir()
                print(f"    Creating {new_folder_name} with {len(sequence)} files")
                
                # Distribute files evenly across 6 timesteps
                files_per_timestep = len(sequence) // 6
                remainder = len(sequence) % 6
                
                file_index = 0
                for timestep in range(1, 7):
                    timestep_folder = new_folder_path / f"timestep_{timestep:02d}"
                    timestep_folder.mkdir()
                    
                    files_in_this_timestep = files_per_timestep
                    if timestep <= remainder:
                        files_in_this_timestep += 1
                    
                    moved_count = 0
                    for _ in range(files_in_this_timestep):
                        if file_index < len(sequence):
                            source_file = sequence[file_index]['file']
                            
                            if source_file.exists():
                                dest_file = timestep_folder / source_file.name
                                shutil.copy2(str(source_file), str(dest_file))
                                moved_count += 1
                            
                            file_index += 1
                    
                    print(f"      Timestep {timestep:02d}: {moved_count} files")
            
            # Ask if user wants to remove original mixed folder
            print(f"\n    Successfully separated {len(flare_sequences)} flare sequences")
            response = input(f"    Remove original mixed folder '{case_folder.name}'? (y/n): ")
            if response.lower() in ['y', 'yes']:
                shutil.rmtree(case_folder)
                print(f"    Removed original folder: {case_folder.name}")
            else:
                print(f"    Kept original folder: {case_folder.name}")
        
        print(f"  ✓ Processed {case_folder.name}")
    
    print(f"\n{'='*60}")
    print("SUCCESS: Reorganized all flare case folders")
    print(f"{'='*60}")

def print_reorganization_summary(base_dir="sharp_cnn_lstm_data"):
    """Print a summary of the reorganized case folders"""
    
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Base directory {base_dir} does not exist!")
        return
    
    case_folders = []
    for folder in base_path.iterdir():
        if folder.is_dir() and (folder.name.startswith('flare_case_') or folder.name.startswith('quiet_case_')):
            case_folders.append(folder)
    
    case_folders.sort()
    
    print(f"\n{'='*60}")
    print(f"REORGANIZATION SUMMARY: {len(case_folders)} case folders")
    print(f"{'='*60}")
    
    for case_folder in case_folders:
        timestep_folders = sorted([f for f in case_folder.iterdir() 
                                 if f.is_dir() and f.name.startswith('timestep_')])
        
        loose_files = [f for f in case_folder.iterdir() 
                      if f.is_file() and f.suffix == '.fits']
        
        total_files = 0
        timestep_info = []
        
        for timestep_folder in timestep_folders:
            fits_files = list(timestep_folder.glob("*.fits"))
            total_files += len(fits_files)
            timestep_info.append(f"{timestep_folder.name}: {len(fits_files)} files")
        
        print(f"\n{case_folder.name}:")
        if timestep_info:
            for info in timestep_info:
                print(f"  {info}")
        if loose_files:
            print(f"  Loose files: {len(loose_files)}")
            total_files += len(loose_files)
        if not timestep_info and not loose_files:
            print(f"  No .fits files found")
        else:
            print(f"  Total: {total_files} files")

if __name__ == "__main__":
    # Run the reorganization
    reorganize_sharp_files()
    
    # Print summary
    print_reorganization_summary()