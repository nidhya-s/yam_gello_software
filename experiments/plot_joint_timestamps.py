#!/usr/bin/env python3
"""
Script to plot control and follower joint values over time.
Loads a combined pkl file and visualizes each joint's control and follower values vs their timestamps.
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_joint_values(pkl_file_path: str, output_file: Optional[str] = None):
    """
    Plot control and follower joint values for each joint over their timestamps.
    
    Args:
        pkl_file_path: Path to the combined pkl file
        output_file: Optional path to save the plot (if None, displays interactively)
    """
    try:
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File {pkl_file_path} not found")
        return
    except Exception as e:
        print(f"Error loading pkl file: {e}")
        return
    
    # Extract control data
    control_data = None
    control_timestamps = None
    
    if 'control' in data:
        control_data = data['control']
        print(f"Control data shape: {control_data.shape if hasattr(control_data, 'shape') else 'unknown'}")
    
    if 'control_timestamp' in data:
        control_ts_array = data['control_timestamp']
        # Filter out None values
        valid_control_indices = []
        valid_control_ts = []
        
        for i, ts in enumerate(control_ts_array):
            if ts is not None:
                valid_control_indices.append(i)
                valid_control_ts.append(ts)
        
        if len(valid_control_ts) > 0:
            control_timestamps = np.array(valid_control_ts, dtype=np.float64)
            # Adjust timestamps to be relative to first timestamp (for better visualization)
            if len(control_timestamps) > 0:
                control_timestamps = control_timestamps - control_timestamps[0]
    else:
        print("Warning: 'control_timestamp' not found in pkl file")
    
    # Extract follower data
    follower_data = None
    follower_timestamps = None
    
    if 'follower_joints' in data:
        follower_data = data['follower_joints']
        print(f"Follower data shape: {follower_data.shape if hasattr(follower_data, 'shape') else 'unknown'}")
    
    if 'follower_timestamp' in data:
        follower_ts_array = data['follower_timestamp']
        # Filter out None values
        valid_follower_indices = []
        valid_follower_ts = []
        
        for i, ts in enumerate(follower_ts_array):
            if ts is not None:
                valid_follower_indices.append(i)
                valid_follower_ts.append(ts)
        
        if len(valid_follower_ts) > 0:
            follower_timestamps = np.array(valid_follower_ts, dtype=np.float64)
            # Adjust timestamps to be relative to first timestamp (for better visualization)
            if len(follower_timestamps) > 0:
                follower_timestamps = follower_timestamps - follower_timestamps[0]
    else:
        print("Warning: 'follower_timestamp' not found in pkl file")
    
    # Determine number of joints
    num_joints = 0
    if control_data is not None:
        if hasattr(control_data, 'shape') and len(control_data.shape) > 1:
            num_joints = control_data.shape[1]
        elif len(control_data) > 0:
            # Handle object array case - check first element
            if isinstance(control_data, np.ndarray) and control_data.dtype == object:
                if control_data[0] is not None:
                    num_joints = len(control_data[0]) if hasattr(control_data[0], '__len__') else 1
            else:
                num_joints = len(control_data[0]) if hasattr(control_data[0], '__len__') else 1
    elif follower_data is not None:
        if hasattr(follower_data, 'shape') and len(follower_data.shape) > 1:
            num_joints = follower_data.shape[1]
        elif len(follower_data) > 0:
            if isinstance(follower_data, np.ndarray) and follower_data.dtype == object:
                if follower_data[0] is not None:
                    num_joints = len(follower_data[0]) if hasattr(follower_data[0], '__len__') else 1
            else:
                num_joints = len(follower_data[0]) if hasattr(follower_data[0], '__len__') else 1
    
    if num_joints == 0:
        print("Error: Could not determine number of joints")
        return
    
    print(f"Plotting {num_joints} joints")
    
    # Create subplots - arrange in a grid
    ncols = 2
    nrows = (num_joints + ncols - 1) // ncols  # Round up
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    fig.suptitle('Control vs Follower Joint Values Over Time', fontsize=16, fontweight='bold')
    
    # Flatten axes if needed
    if num_joints == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    # Plot each joint
    for joint_idx in range(num_joints):
        ax = axes[joint_idx]
        
        # Extract control values for this joint
        if control_data is not None and control_timestamps is not None:
            control_values = []
            for i in valid_control_indices:
                try:
                    if hasattr(control_data, 'shape') and len(control_data.shape) > 1:
                        # Regular array
                        val = control_data[i, joint_idx]
                    else:
                        # Object array or list
                        val = control_data[i][joint_idx] if control_data[i] is not None else None
                    if val is not None:
                        control_values.append(float(val))
                except (IndexError, TypeError):
                    pass
            
            if len(control_values) > 0:
                control_ts_for_plot = control_timestamps[:len(control_values)]
                ax.plot(control_ts_for_plot, control_values, 'b-', label='Control', 
                       linewidth=1.5, marker='o', markersize=2, alpha=0.7)
        
        # Extract follower values for this joint
        if follower_data is not None and follower_timestamps is not None:
            follower_values = []
            for i in valid_follower_indices:
                try:
                    if hasattr(follower_data, 'shape') and len(follower_data.shape) > 1:
                        # Regular array
                        val = follower_data[i, joint_idx]
                    else:
                        # Object array or list
                        val = follower_data[i][joint_idx] if follower_data[i] is not None else None
                    if val is not None:
                        follower_values.append(float(val))
                except (IndexError, TypeError):
                    pass
            
            if len(follower_values) > 0:
                follower_ts_for_plot = follower_timestamps[:len(follower_values)]
                ax.plot(follower_ts_for_plot, follower_values, 'r-', label='Follower', 
                       linewidth=1.5, marker='s', markersize=2, alpha=0.7)
        
        ax.set_title(f'Joint {joint_idx + 1}', fontsize=11, fontweight='bold')
        ax.set_ylabel('Joint Value (rad)', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_joints, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_joint_timestamps.py <pkl_file_path> [output_image_path]")
        print("Example: python plot_joint_timestamps.py combined_joints.pkl")
        print("         python plot_joint_timestamps.py combined_joints.pkl plot.png")
        sys.exit(1)
    
    pkl_file_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    plot_joint_values(pkl_file_path, output_file)

