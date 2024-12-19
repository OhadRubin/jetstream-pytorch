#!/usr/bin/env python3

import fire
import glob
import os
import safetensors.torch

def merge_safetensors(input_folder):
    """Merge multiple safetensors files in a folder into a single file.
    
    Args:
        input_folder: Folder containing safetensors files to merge
    """
    input_pattern = os.path.join(input_folder, "*.safetensors")
    output_file = os.path.join(input_folder, "merged.safetensors")
    
    merge_state_dict = {}
    input_files = glob.glob(input_pattern)
    
    if not input_files:
        print(f"No safetensors files found in {input_folder}")
        return
        
    for file in input_files:
        load_files_dict = safetensors.torch.load_file(file)
        merge_state_dict.update(load_files_dict)
    
    safetensors.torch.save_file(merge_state_dict, output_file)
    
    # Delete original files after successful merge
    for file in input_files:
        os.remove(file)
        
    print(f"Merged {len(input_files)} files into {output_file}")
    print("Original files deleted")

def print_usage():
    """Print usage instructions for the script."""
    print("""
Usage: merge.py [input_folder]

Merges all safetensors files in the input folder into a single file.
Original files are deleted after successful merge.

Arguments:
    input_folder: Folder containing safetensors files to merge

Example:
    # Merge all safetensors files in model_weights folder
    python merge.py model_weights
""")

if __name__ == '__main__':
    fire.Fire(merge_safetensors)