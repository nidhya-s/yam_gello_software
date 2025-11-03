import sys
import pickle

def print_pkl_from_file(filepath):
    print(f"Reading file: {filepath}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    # Print all data in the pkl file
    for key, value in data.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_pkl.py /path/to/file.pkl")
        sys.exit(1)
    fname = sys.argv[1]
    print_pkl_from_file(fname)
