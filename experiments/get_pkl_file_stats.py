#!/usr/bin/env python3
"""
Script to analyze pkl file statistics for control and follower timestamps.
Calculates average time differences between consecutive timestamps.

Note: Timestamps are measured with time.perf_counter() which returns
monotonic clock values in seconds (as float). Differences represent elapsed time
between consecutive measurements.
"""

import sys
import pickle
import numpy as np


def get_pkl_file_stats(pkl_file_path):
    """
    Calculate average differences between consecutive control and follower timestamps.
    
    Args:
        pkl_file_path: Path to the pkl file containing combined data
        
    Returns:
        dict: Dictionary containing statistics
    """
    try:
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File {pkl_file_path} not found")
        return None
    except Exception as e:
        print(f"Error loading pkl file: {e}")
        return None
    
    stats = {}
    
    # Calculate control timestamp differences
    # Timestamps are from time.perf_counter() - monotonic clock in seconds (float)
    if 'control_timestamp' in data:
        control_timestamps = data['control_timestamp']
        # Filter out None values
        valid_control_ts = [ts for ts in control_timestamps if ts is not None]
        
        if len(valid_control_ts) > 1:
            # Convert to numpy array (perf_counter returns float seconds)
            control_ts_array = np.array(valid_control_ts, dtype=np.float64)
            control_diffs = np.diff(control_ts_array)  # Differences in seconds
            avg_control_diff = np.mean(control_diffs)
            stats['control_timestamp'] = {
                'average_diff_s': avg_control_diff,
                'average_diff_ms': avg_control_diff * 1000,
                'average_diff_ns': avg_control_diff * 1e9,
                'num_samples': len(valid_control_ts),
                'min_diff_s': np.min(control_diffs),
                'max_diff_s': np.max(control_diffs),
                'std_diff_s': np.std(control_diffs)
            }
        elif len(valid_control_ts) == 1:
            print("Warning: Only one control timestamp found, cannot calculate differences")
            stats['control_timestamp'] = {'num_samples': 1}
        else:
            print("Warning: No valid control timestamps found")
            stats['control_timestamp'] = {'num_samples': 0}
    else:
        print("Warning: 'control_timestamp' not found in pkl file")
        stats['control_timestamp'] = None
    
    # Calculate follower timestamp differences
    # Timestamps are from time.perf_counter() - monotonic clock in seconds (float)
    if 'follower_timestamp' in data:
        follower_timestamps = data['follower_timestamp']
        # Filter out None values
        valid_follower_ts = [ts for ts in follower_timestamps if ts is not None]
        
        if len(valid_follower_ts) > 1:
            # Convert to numpy array (perf_counter returns float seconds)
            follower_ts_array = np.array(valid_follower_ts, dtype=np.float64)
            follower_diffs = np.diff(follower_ts_array)  # Differences in seconds
            avg_follower_diff = np.mean(follower_diffs)
            stats['follower_timestamp'] = {
                'average_diff_s': avg_follower_diff,
                'average_diff_ms': avg_follower_diff * 1000,
                'average_diff_ns': avg_follower_diff * 1e9,
                'num_samples': len(valid_follower_ts),
                'min_diff_s': np.min(follower_diffs),
                'max_diff_s': np.max(follower_diffs),
                'std_diff_s': np.std(follower_diffs)
            }
        elif len(valid_follower_ts) == 1:
            print("Warning: Only one follower timestamp found, cannot calculate differences")
            stats['follower_timestamp'] = {'num_samples': 1}
        else:
            print("Warning: No valid follower timestamps found")
            stats['follower_timestamp'] = {'num_samples': 0}
    else:
        print("Warning: 'follower_timestamp' not found in pkl file")
        stats['follower_timestamp'] = None
    
    return stats


def print_stats(stats):
    """Print statistics in a readable format."""
    print("\n" + "="*60)
    print("PKL File Statistics")
    print("="*60)
    
    if stats['control_timestamp'] and 'average_diff_s' in stats['control_timestamp']:
        ct = stats['control_timestamp']
        print(f"\nControl Timestamps:")
        print(f"  Number of samples: {ct['num_samples']}")
        print(f"  Average difference: {ct['average_diff_ms']:.3f} ms ({ct['average_diff_s']:.6f} s, {ct['average_diff_ns']:.0f} ns)")
        print(f"  Min difference:    {ct['min_diff_s'] * 1000:.3f} ms ({ct['min_diff_s']:.6f} s, {ct['min_diff_s'] * 1e9:.0f} ns)")
        print(f"  Max difference:    {ct['max_diff_s'] * 1000:.3f} ms ({ct['max_diff_s']:.6f} s, {ct['max_diff_s'] * 1e9:.0f} ns)")
        print(f"  Std deviation:      {ct['std_diff_s'] * 1000:.3f} ms ({ct['std_diff_s']:.6f} s, {ct['std_diff_s'] * 1e9:.0f} ns)")
    else:
        print("\nControl Timestamps: Not available")
    
    if stats['follower_timestamp'] and 'average_diff_s' in stats['follower_timestamp']:
        ft = stats['follower_timestamp']
        print(f"\nFollower Timestamps:")
        print(f"  Number of samples: {ft['num_samples']}")
        print(f"  Average difference: {ft['average_diff_ms']:.3f} ms ({ft['average_diff_s']:.6f} s, {ft['average_diff_ns']:.0f} ns)")
        print(f"  Min difference:    {ft['min_diff_s'] * 1000:.3f} ms ({ft['min_diff_s']:.6f} s, {ft['min_diff_s'] * 1e9:.0f} ns)")
        print(f"  Max difference:    {ft['max_diff_s'] * 1000:.3f} ms ({ft['max_diff_s']:.6f} s, {ft['max_diff_s'] * 1e9:.0f} ns)")
        print(f"  Std deviation:      {ft['std_diff_s'] * 1000:.3f} ms ({ft['std_diff_s']:.6f} s, {ft['std_diff_s'] * 1e9:.0f} ns)")
    else:
        print("\nFollower Timestamps: Not available")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_pkl_file_stats.py <pkl_file_path>")
        print("Example: python get_pkl_file_stats.py combined_joints.pkl")
        sys.exit(1)
    
    pkl_file_path = sys.argv[1]
    stats = get_pkl_file_stats(pkl_file_path)
    
    if stats:
        print_stats(stats)

