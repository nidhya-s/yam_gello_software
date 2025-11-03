#!/usr/bin/env python3
"""
Script to plot control and follower joint values over time from h5 file.
Loads an h5 file and visualizes each joint's control and follower values vs their timestamps.
"""

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_joint_values_h5(h5_file_path: str, output_file: Optional[str] = None):
    """
    Plot control and follower joint values for each joint over their timestamps from h5 file.
    
    Args:
        h5_file_path: Path to the h5 file
        output_file: Optional path to save the plot (if None, displays interactively)
    """
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # Print available keys for debugging
            available_keys = list(f.keys())
            print(f"Available keys in h5 file: {available_keys}")
            
            # Extract control data (try multiple possible key names)
            control_data = None
            control_timestamps = None
            
            # Try different key names for control
            control_keys = ['control', 'gello_position', 'control/values', 'gello_position/values']
            for key in control_keys:
                if key in f:
                    control_data = f[key][:]
                    print(f"Found control data at '{key}', shape: {control_data.shape}")
                    break
            
            # Try different key names for control timestamps
            control_ts_keys = ['control_timestamp', 'gello_timestamp', 'control/timestamp', 'gello_position/timestamp']
            for key in control_ts_keys:
                if key in f:
                    control_timestamps = f[key][:]
                    print(f"Found control timestamps at '{key}', shape: {control_timestamps.shape}")
                    break
            
            # Extract follower data (try multiple possible key names)
            follower_data = None
            follower_timestamps = None
            
            # Try different key names for follower
            follower_keys = ['follower_joints', 'yam_position', 'follower_joints/values', 'yam_position/values']
            for key in follower_keys:
                if key in f:
                    follower_data = f[key][:]
                    print(f"Found follower data at '{key}', shape: {follower_data.shape}")
                    break
            
            # Try different key names for follower timestamps
            follower_ts_keys = ['follower_timestamp', 'yam_timestamp', 'follower_joints/timestamp', 'yam_position/timestamp']
            for key in follower_ts_keys:
                if key in f:
                    follower_timestamps = f[key][:]
                    print(f"Found follower timestamps at '{key}', shape: {follower_timestamps.shape}")
                    break
            
            # Handle 1D vs 2D arrays
            if control_data is not None:
                if len(control_data.shape) == 1:
                    # Reshape if needed - assume it's a flat array
                    # Try to infer dimensions - assume 14 joints for bimanual
                    num_samples = len(control_data) // 14
                    if len(control_data) % 14 == 0:
                        control_data = control_data.reshape(num_samples, 14)
                        print(f"Reshaped control_data to {control_data.shape}")
                elif len(control_data.shape) == 2:
                    print(f"Control data already 2D: {control_data.shape}")
                else:
                    print(f"Warning: Unexpected control_data shape: {control_data.shape}")
            
            if follower_data is not None:
                if len(follower_data.shape) == 1:
                    # Reshape if needed - assume 14 joints for bimanual
                    num_samples = len(follower_data) // 14
                    if len(follower_data) % 14 == 0:
                        follower_data = follower_data.reshape(num_samples, 14)
                        print(f"Reshaped follower_data to {follower_data.shape}")
                elif len(follower_data.shape) == 2:
                    print(f"Follower data already 2D: {follower_data.shape}")
                else:
                    print(f"Warning: Unexpected follower_data shape: {follower_data.shape}")
            
            # Convert timestamps to numpy arrays and handle masking
            # Data is already synced by timestamp, so we use timestamps as-is
            reference_time = None
            
            if control_timestamps is not None:
                control_timestamps = np.array(control_timestamps, dtype=np.float64)
                # Filter out invalid values (NaN, inf, etc.)
                valid_control_mask = np.isfinite(control_timestamps)
                control_timestamps = control_timestamps[valid_control_mask]
                if control_data is not None:
                    control_data = control_data[valid_control_mask]
                # Use first timestamp as reference (data is already synced)
                if len(control_timestamps) > 0:
                    reference_time = control_timestamps[0]
            elif control_data is not None and len(control_data.shape) > 1:
                # If no timestamps, use sample indices as time (assume 30 Hz for visualization)
                num_samples = control_data.shape[0]
                control_timestamps = np.arange(num_samples) / 30.0  # Convert to seconds
                reference_time = 0.0
                print(f"Using sample indices as control timestamps (assuming 30 Hz)")
            
            if follower_timestamps is not None:
                follower_timestamps = np.array(follower_timestamps, dtype=np.float64)
                # Filter out invalid values
                valid_follower_mask = np.isfinite(follower_timestamps)
                follower_timestamps = follower_timestamps[valid_follower_mask]
                if follower_data is not None:
                    follower_data = follower_data[valid_follower_mask]
                # Use same reference time if not set yet, otherwise use existing reference
                if reference_time is None and len(follower_timestamps) > 0:
                    reference_time = follower_timestamps[0]
            elif follower_data is not None and len(follower_data.shape) > 1:
                # If no timestamps, use sample indices as time (assume 30 Hz for visualization)
                num_samples = follower_data.shape[0]
                follower_timestamps = np.arange(num_samples) / 30.0  # Convert to seconds
                if reference_time is None:
                    reference_time = 0.0
                print(f"Using sample indices as follower timestamps (assuming 30 Hz)")
            
            # Normalize both to same reference (data is already synced, just normalize for visualization)
            if reference_time is not None:
                if control_timestamps is not None and len(control_timestamps) > 0:
                    control_timestamps = control_timestamps - reference_time
                if follower_timestamps is not None and len(follower_timestamps) > 0:
                    follower_timestamps = follower_timestamps - reference_time
            
            # Determine number of joints
            num_joints = 0
            if control_data is not None and len(control_data.shape) > 1:
                num_joints = control_data.shape[1]
                print(f"Determined {num_joints} joints from control_data")
            elif follower_data is not None and len(follower_data.shape) > 1:
                num_joints = follower_data.shape[1]
                print(f"Determined {num_joints} joints from follower_data")
            
            if num_joints == 0:
                print("Error: Could not determine number of joints")
                if control_data is not None:
                    print(f"  control_data shape: {control_data.shape}")
                if follower_data is not None:
                    print(f"  follower_data shape: {follower_data.shape}")
                return
            
            print(f"Plotting {num_joints} joints")
            
            # Create subplots - side by side: 1-7 in first col, 8-14 in second col
            ncols = 2
            nrows = 7  # 7 joints per column
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
            fig.suptitle('Control vs Follower Joint Values Over Time (H5)', fontsize=16, fontweight='bold')
            
            # Plot each joint
            # First column: joints 0-6 (1-7)
            # Second column: joints 7-13 (8-14)
            for joint_idx in range(num_joints):
                if joint_idx < 7:
                    # First column, row = joint_idx
                    row = joint_idx
                    col = 0
                elif joint_idx < 14:
                    # Second column, row = joint_idx - 7
                    row = joint_idx - 7
                    col = 1
                else:
                    # Skip if more than 14 joints
                    continue
                
                ax = axes[row, col]
                
                # Plot control values for this joint
                if control_data is not None and control_timestamps is not None:
                    if len(control_data.shape) > 1 and control_data.shape[1] > joint_idx:
                        control_values = control_data[:, joint_idx]
                        ax.plot(control_timestamps, control_values, 'b-', label='Control', 
                               linewidth=1.5, marker='o', markersize=2, alpha=0.7)
                
                # Plot follower values for this joint
                if follower_data is not None and follower_timestamps is not None:
                    if len(follower_data.shape) > 1 and follower_data.shape[1] > joint_idx:
                        follower_values = follower_data[:, joint_idx]
                        ax.plot(follower_timestamps, follower_values, 'r-', label='Follower', 
                               linewidth=1.5, marker='s', markersize=2, alpha=0.7)
                
                ax.set_title(f'Joint {joint_idx + 1}', fontsize=11, fontweight='bold')
                ax.set_ylabel('Joint Value (rad)', fontsize=10)
                ax.set_xlabel('Time (s)', fontsize=10)
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            # If we have fewer than 14 joints, hide the extra subplots
            if num_joints < 14:
                # Hide remaining spots in first column (if num_joints < 7)
                if num_joints < 7:
                    for row in range(num_joints, 7):
                        axes[row, 0].set_visible(False)
                # Hide remaining spots in second column (if num_joints < 14)
                if num_joints < 14:
                    start_row = max(0, num_joints - 7)
                    for row in range(start_row, 7):
                        axes[row, 1].set_visible(False)
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f"Plot saved to {output_file}")
            else:
                plt.show()
    
    except FileNotFoundError:
        print(f"Error: File {h5_file_path} not found")
        return
    except Exception as e:
        print(f"Error loading h5 file: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_joint_timestamps_h5.py <h5_file_path> [output_image_path]")
        print("Example: python plot_joint_timestamps_h5.py joints.h5")
        print("         python plot_joint_timestamps_h5.py joints.h5 plot.png")
        sys.exit(1)
    
    h5_file_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    plot_joint_values_h5(h5_file_path, output_file)

