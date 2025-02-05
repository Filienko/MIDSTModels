import os
import shutil
import glob
import pandas as pd

# Define source files (JSON files to copy)
source_folder = r"C:\Users\ksush\attacks\MIDSTModels\tabddpm_white_box\train\tabddpm_1"
source_files = ["trans.json", "dataset_meta.json", "trans_domain.json"]

# Define destination parent directories
dest_folders = [
    r"C:\Users\ksush\attacks\MIDSTModels\tabddpm_white_box\dev",
    r"C:\Users\ksush\attacks\MIDSTModels\tabddpm_white_box\final"
]

# Iterate through each destination folder
for parent in dest_folders:
    # Find all matching tabddpm_{n} subfolders
    subfolders = glob.glob(os.path.join(parent, "tabddpm_*"))
    
    for folder in subfolders:
        # Copy JSON files
        for file in source_files:
            src_path = os.path.join(source_folder, file)
            dest_path = os.path.join(folder, file)
            shutil.copy2(src_path, dest_path)
            print(f"Copied {file} to {dest_path}")
        
        # Process challenge_with_id.csv in the current folder
        csv_source = os.path.join(folder, "challenge_with_id.csv")
        csv_dest = os.path.join(folder, "challenge.csv")
        
        if os.path.exists(csv_source):
            train_w_id_df = pd.read_csv(csv_source)
            train_df = train_w_id_df.iloc[:, 2:]  # Drop first two columns
            train_df.to_csv(csv_dest, index=False)
            print(f"Created {csv_dest} from {csv_source}")

print("All files copied and CSV files processed successfully!")
