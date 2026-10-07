"""Audit current source files into a NEW review directory, preserving originals."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.roy_review import audit

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.output), indent=2))
