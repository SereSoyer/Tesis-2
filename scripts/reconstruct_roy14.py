"""Rebuild Roy from TIFF+PNG under the explicit fourteen-channel policy."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.roy_reconstruction import reconstruct

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--policy', default='configs/reconstruct_roy14_v1.json')
    args = parser.parse_args()
    print(json.dumps(reconstruct(args.root,args.policy),indent=2))
