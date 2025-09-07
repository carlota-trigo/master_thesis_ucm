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
HEAD1_LABEL_COL = "diagnosis_grouped"   # Head 1 
HEAD2_LABEL_COL = "lesion_type"         # Head 2 (e.g., benign, malignant, unknown)

# -- Class Definitions --
# Head 1 classes (MUST NOT include 'unknown')
HEAD1_CLASSES = ['nv', 'mel', 'bkl', 'bcc', 'scc_akiec', 'vasc', 'df', 'other', 'no_lesion']
# Head 2 classes (SHOULD include 'unknown')
HEAD2_CLASSES = ['benign', 'malignant',  'no_lesion']

# -- Splitting Parameters --
VAL_SIZE  = 0.15 # 15% of known data for validation
TEST_SIZE = 0.15 # 15% of known data for testing
SEED = 999 # For reproducible results

# ---- 2. Main Data Preparation Logic ----
def main():
    """
    Prepares the dataset by splitting it into train, val, test, and a dedicated
    ood set for the first head (diagnosis).
    """
    print(f"Reading metadata from: {METADATA_CSV}")
    assert METADATA_CSV.is_file(), "Metadata file not found!"
    # The sep=None and engine='python' are good fallbacks for CSV reading
    df = pd.read_csv(METADATA_CSV, sep=None, engine="python")

    # --- Separate data based on the primary head's label ---
    # We define the main splits based on whether the 'diagnosis' is known.
    known_df = df[df[HEAD1_LABEL_COL] != 'unknown'].copy()
    unknown_df = df[df[HEAD1_LABEL_COL] == 'unknown'].copy()
    print(f"Found {len(known_df)} samples with known diagnosis and {len(unknown_df)} with unknown diagnosis.")

    # --- Split the 'known' data into train, val, and test ---
    # This data will be used for training and evaluating both heads.
    # Stratify by diagnosis and origin dataset to ensure balanced splits.
    stratify_key = known_df[HEAD1_LABEL_COL].astype(str) + "||" + known_df[DATASET_COL].astype(str)
    
    # First, split into (train+val) and test
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

    # --- Assign the 'unknown' data to the OOD split ---
    # This entire set is reserved for testing Head 1's uncertainty.
    if not unknown_df.empty:
        unknown_df["split"] = "test_ood"

    # --- Combine all parts and create final labels ---
    known_df_split = pd.concat([train_val_set, known_df[known_df.split == 'test']])
    final_df = pd.concat([known_df_split, unknown_df], ignore_index=True)

    # Create numerical labels for Head 1 ('unknown' from OOD set will become NaN)
    head1_map = {cls: i for i, cls in enumerate(HEAD1_CLASSES)}
    final_df['head1_idx'] = final_df[HEAD1_LABEL_COL].map(head1_map)

    # Create numerical labels for Head 2 (all splits will have valid labels)
    # IMPORTANT: Ensure 'unknown' is NOT in your HEAD2_CLASSES list in the config section!
    head2_map = {cls: i for i, cls in enumerate(HEAD2_CLASSES)}
    final_df['head2_idx'] = final_df[HEAD2_LABEL_COL].map(head2_map)

    # --- Save results and print summary ---
    final_df.to_csv(OUT_CSV, index=False)
    print(f"\nSuccessfully saved prepared data to: {OUT_CSV}")
    print("\n--- Report ---")
    print("Split counts:")
    print(final_df["split"].value_counts())
    print("\nTraining set counts for Head 2 (lesion_type):")
    # This should now only show counts for your valid lesion types
    print(final_df[final_df.split == 'train'][HEAD2_LABEL_COL].value_counts())

if __name__ == "__main__":
    main()