#!/usr/bin/env python3
"""
Simple script to combine all joint data pickle files in a directory.
Each pkl file has timestamp as filename and contains joint data.
"""

import os
import pickle
import glob
import numpy as np
import sys
import signal


def combine_joints(input_dir, output_file=None):
    """
    Combine all joint data pickle files in a directory.

    Args:
        input_dir: Directory containing timestamped pickle files
        output_file: Output file path (default: combined_joints.pkl in input_dir)
    """
    # Temporarily ignore SIGINT to prevent interruption during combining
    original_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    try:
        # Set default output file to be in the input directory
        if output_file is None:
            output_file = os.path.join(input_dir, "combined_joints.pkl")

        # Find all pickle files and sort by timestamp
        # Exclude the combined file if it already exists
        files = sorted([f for f in glob.glob(os.path.join(input_dir, "*.pkl"))
                        if f != output_file])
        
        if not files:
            print(f"No pickle files found in {input_dir}")
            return None
        
        print(f"Found {len(files)} pickle files")
        
        # Load all data
        timestamps = []
        joint_positions = []
        control = []
        control_timestamp = []
        follower_joints = []
        follower_timestamp = []
        
        for file_path in files:
            # Extract timestamp from filename
            timestamp = int(os.path.basename(file_path).split('.')[0])
            
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                timestamps.append(timestamp)
                
                # Required fields
                if 'joint_positions' in data:
                    joint_positions.append(data['joint_positions'])
                else:
                    joint_positions.append(None)
                
                control.append(data['control'])
                
                # Optional fields
                if 'control_timestamp' in data:
                    control_timestamp.append(data['control_timestamp'])
                else:
                    control_timestamp.append(None)
                
                if 'follower_joints' in data:
                    follower_joints.append(data['follower_joints'])
                else:
                    follower_joints.append(None)
                
                if 'follower_timestamp' in data:
                    follower_timestamp.append(data['follower_timestamp'])
                else:
                    follower_timestamp.append(None)
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        # Combine into single arrays, handling None values
        combined_data = {
            'timestamps': np.array(timestamps),
            'control': np.array(control)
        }
        
        # Add joint_positions if all are not None
        if all(jp is not None for jp in joint_positions):
            combined_data['joint_positions'] = np.array(joint_positions)
        elif any(jp is not None for jp in joint_positions):
            # Handle mixed None/not-None case - store as object array
            combined_data['joint_positions'] = np.array(joint_positions, dtype=object)
        
        # Add control_timestamp if any exist
        if any(ct is not None for ct in control_timestamp):
            combined_data['control_timestamp'] = np.array(control_timestamp, dtype=object)
        
        # Add follower_joints if any exist
        if any(fj is not None for fj in follower_joints):
            combined_data['follower_joints'] = np.array(follower_joints, dtype=object)
        
        # Add follower_timestamp if any exist
        if any(ft is not None for ft in follower_timestamp):
            combined_data['follower_timestamp'] = np.array(follower_timestamp, dtype=object)
        
        # Save combined data
        with open(output_file, 'wb') as f:
            pickle.dump(combined_data, f)

        # print(f"Successfully combined {len(files)} files into {output_file}")
        # print(f"Data shapes:")
        # for key, value in combined_data.items():
            # print(f"  {key}: {value.shape}")

        # Delete individual pkl files
        # print(f"\nDeleting {len(files)} individual pickle files...")
        for file_path in files:
            try:
                os.remove(file_path)
                # print(f"  Deleted: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  Error deleting {file_path}: {e}")
        
        # print("Cleanup complete!")
        return combined_data
    
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # print("Usage: python combine_joints.py <input_directory> [output_file]")
        # print("Example: python combine_joints.py /path/to/joint/data")
        # print("         python combine_joints.py /path/to/joint/data /custom/path/combined_joints.pkl")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    combine_joints(input_dir, output_file)
