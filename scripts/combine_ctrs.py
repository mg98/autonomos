import os
import lmdb
import pickle
from tqdm import tqdm
import glob
import shutil

def combine_lmdb_files():
    # Find all individual LMDB files
    lmdb_files = sorted(glob.glob('data/ctrs_*.lmdb'))
    if not lmdb_files:
        print("No individual LMDB files found!")
        return

    print(f"Found {len(lmdb_files)} LMDB files to combine")
    
    # Open the target LMDB file
    with lmdb.open('data/ctrs.lmdb', map_size=2**46) as target_db:
        with target_db.begin(write=True) as target_txn:
            # Process each source LMDB file
            for lmdb_file in tqdm(lmdb_files, desc="Processing LMDB files"):
                with lmdb.open(lmdb_file, readonly=True) as source_db:
                    with source_db.begin() as source_txn:
                        cursor = source_txn.cursor()
                        # Copy all entries from source to target
                        for key, value in cursor:
                            target_txn.put(key, value)

    print("Successfully combined all LMDB files into data/ctrs.lmdb")

    # Delete original LMDB files after successful combination
    print("Removing original LMDB files...")
    for lmdb_file in tqdm(lmdb_files, desc="Deleting LMDB files"):
        try:
            # Remove the directory and all its contents
            if os.path.isdir(lmdb_file):
                shutil.rmtree(lmdb_file)
                print(f"Deleted {lmdb_file}")
        except Exception as e:
            print(f"Error deleting {lmdb_file}: {e}")

if __name__ == "__main__":
    combine_lmdb_files()
