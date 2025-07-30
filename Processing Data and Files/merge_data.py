import numpy as np
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python merge_data.py file1.npz file2.npz [output.npz]")
        return
    file1, file2 = sys.argv[1], sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else "all_data.npz"

    data1 = np.load(file1, allow_pickle=True, mmap_mode='r')
    data2 = np.load(file2, allow_pickle=True, mmap_mode='r')

    merged = {}
    keys = set(data1.keys()) & set(data2.keys())
    chunk_size = 1000  # Change as needed for memory
    for key in keys:
        arr1, arr2 = data1[key], data2[key]
        print(f"Merging key: {key}")
        # Merge arrays in chunks
        if isinstance(arr1, np.ndarray) and arr1.ndim > 0:
            merged_list = []
            for start in range(0, arr1.shape[0], chunk_size):
                merged_list.append(arr1[start:start+chunk_size])
            for start in range(0, arr2.shape[0], chunk_size):
                merged_list.append(arr2[start:start+chunk_size])
            merged[key] = np.concatenate(merged_list, axis=0)
        else:
            merged[key] = list(arr1) + list(arr2)

    np.savez_compressed(output, **merged)
    print(f"Merged keys: {list(merged.keys())}")
    print(f"Merged and saved to {output}")

if __name__ == "__main__":
    main()
