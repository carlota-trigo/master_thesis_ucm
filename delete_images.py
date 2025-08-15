import os
import glob

# Path to the folder
folder_path = r"F:\TFM\data_clean\images"

# Get all files in the folder
files = glob.glob(os.path.join(folder_path, "*"))

# Delete each file
for file_path in files:
    if os.path.isfile(file_path):
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

print("All files deleted.")