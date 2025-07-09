import os

base_dir = "sharp_cnn_lstm_data"
flare_count = 0
quiet_count = 0

if os.path.exists(base_dir):
    for name in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, name)):
            if name.startswith("flare_case_"):
                flare_count += 1
            elif name.startswith("quiet_case_"):
                quiet_count += 1

print(f"flare_case folders: {flare_count}")
print(f"quiet_case folders: {quiet_count}")
