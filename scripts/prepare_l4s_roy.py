"""Prepare a new integrated dataset version after explicit methodological decisions."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.integration import prepare

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--policy', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.policy), indent=2))
