# prepare_data.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# ---- 1. Configuration (edit these values) ----
DATA_DIR = Path("../data") # Directory where your data is
METADATA_CSV = DATA_DIR.joinpath("metadata_clean.csv") # Input metadata file
OUT_CSV      = DATA_DIR.joinpath("training_prepared_data.csv")

# -- Column Names --
IMAGE_ID_COL = "image_id"
DATASET_COL  = "origin_dataset"
HEAD1_LABEL_COL = "lesion_type"         # Head 1 (Primary: benign, malignant, no_lesion)
HEAD2_LABEL_COL = "diagnosis_grouped"   # Head 2 (Secondary: fine-grained diagnosis)

# -- Class Definitions --
# Head 1 classes (Primary task - coarse classification)
HEAD1_CLASSES = ['benign', 'malignant', 'no_lesion']
# Head 2 classes (Secondary task - fine-grained diagnosis, MUST NOT include 'unknown')
HEAD2_CLASSES = ['nv', 'mel', 'bkl', 'bcc', 'scc_akiec', 'vasc', 'df', 'other', 'no_lesion']

# -- Splitting Parameters --
VAL_SIZE  = 0.15 # 15% of all data for validation
TEST_SIZE = 0.15 # 15% of all data for testing
SEED = 999 # For reproducible results

# ---- 2. Main Data Preparation Logic ----
def main():
    """
    Prepares the dataset by splitting it into train, val, test, and a dedicated
    ood set for the second head (fine-grained diagnosis).
    
    Head 1: Coarse classification (benign, malignant, no_lesion) - all samples
    Head 2: Fine-grained diagnosis (nv, mel, bkl, etc.) - only known samples
    """
    print(f"Reading metadata from: {METADATA_CSV}")
    assert METADATA_CSV.is_file(), "Metadata file not found!"
    # The sep=None and engine='python' are good fallbacks for CSV reading
    df = pd.read_csv(METADATA_CSV, sep=None, engine="python")

    # --- Separate data based on the secondary head's label ---
    known_df = df[df[HEAD2_LABEL_COL] != 'unknown'].copy()
    unknown_df = df[df[HEAD2_LABEL_COL] == 'unknown'].copy()
    print(f"Found {len(known_df)} samples with known diagnosis and {len(unknown_df)} with unknown diagnosis.")
    print(f"Total samples: {len(df)}")

    # --- Split KNOWN data into train, val, and test for Head 2 ---
    # Stratify by lesion_type to ensure balanced splits
    stratify_key = known_df[HEAD1_LABEL_COL]
    
    # First, split known samples into (train+val) and test
    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_val_idx, test_idx = next(splitter1.split(known_df, stratify_key))
    known_df.loc[known_df.index[test_idx], "split"] = "test"
    
    # Next, split (train+val) into train and val
    train_val_set = known_df.iloc[train_val_idx].copy()
    stratify_key_2 = stratify_key.iloc[train_val_idx]
    # Adjust validation size for the second split
    val_proportion = VAL_SIZE / (1 - TEST_SIZE)
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=val_proportion, random_state=SEED)
    train_idx, val_idx = next(splitter2.split(train_val_set, stratify_key_2))
    
    train_val_set.loc[train_val_set.index[train_idx], "split"] = "train"
    train_val_set.loc[train_val_set.index[val_idx], "split"] = "val"

    # --- Add unknown samples to train/val/test for Head 1 ---
    # First, reserve some unknown samples for OOD testing
    ood_reserve_ratio = 0.1  # Reserve 10% of unknown samples for OOD testing
    ood_count = int(len(unknown_df) * ood_reserve_ratio)
    ood_samples = unknown_df.iloc[:ood_count].copy()
    ood_samples["split"] = "test_ood"
    
    # Use remaining unknown samples for train/val/test
    remaining_unknown_df = unknown_df.iloc[ood_count:].copy()
    
    # Distribute remaining unknown samples proportionally across splits
    train_size = len(train_val_set[train_val_set.split == 'train'])
    val_size = len(train_val_set[train_val_set.split == 'val'])
    test_size = len(known_df[known_df.split == 'test'])
    total_known = train_size + val_size + test_size
    
    # Calculate how many unknowns to add to each split
    unknown_train_count = int(len(remaining_unknown_df) * train_size / total_known)
    unknown_val_count = int(len(remaining_unknown_df) * val_size / total_known)
    unknown_test_count = len(remaining_unknown_df) - unknown_train_count - unknown_val_count
    
    # Split remaining unknown samples
    remaining_unknown_df = remaining_unknown_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    unknown_train = remaining_unknown_df.iloc[:unknown_train_count].copy()
    unknown_val = remaining_unknown_df.iloc[unknown_train_count:unknown_train_count + unknown_val_count].copy()
    unknown_test = remaining_unknown_df.iloc[unknown_train_count + unknown_val_count:].copy()
    
    # Assign splits to unknown samples
    unknown_train["split"] = "train"
    unknown_val["split"] = "val"
    unknown_test["split"] = "test"

    # --- Combine all parts ---
    final_df = pd.concat([
        train_val_set, 
        known_df[known_df.split == 'test'],
        unknown_train,
        unknown_val, 
        unknown_test,
        ood_samples
    ], ignore_index=True)

    # Create numerical labels for Head 1 (coarse classification - all samples have valid labels)
    head1_map = {cls: i for i, cls in enumerate(HEAD1_CLASSES)}
    final_df['head1_idx'] = final_df[HEAD1_LABEL_COL].map(head1_map)

    # Create numerical labels for Head 2 (fine-grained diagnosis - 'unknown' from OOD set will become NaN)
    head2_map = {cls: i for i, cls in enumerate(HEAD2_CLASSES)}
    final_df['head2_idx'] = final_df[HEAD2_LABEL_COL].map(head2_map)

    # --- Save results and print summary ---
    final_df.to_csv(OUT_CSV, index=False)
    print(f"\nSuccessfully saved prepared data to: {OUT_CSV}")
    print("\n--- Report ---")
    print("Split counts:")
    print(final_df["split"].value_counts())
    print("\nTraining set counts for Head 1 (lesion_type - coarse classification):")
    print(final_df[final_df.split == 'train'][HEAD1_LABEL_COL].value_counts())
    print("\nTraining set counts for Head 2 (diagnosis_grouped - fine-grained diagnosis):")
    print(final_df[final_df.split == 'train'][HEAD2_LABEL_COL].value_counts())

if __name__ == "__main__":
    main()