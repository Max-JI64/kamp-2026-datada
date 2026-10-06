"""Replay selected modeling from raw data into a new output folder.

Example: python Modeling/scripts/modeling_reproduce_selected.py --output-dir Modeling/tables/modeling_replay
Run with regular CPython3.13. Existing outputs are never deleted or overwritten.
Frozen source definitions/model settings and reference results must be retained.
"""
import argparse
import sys
from pathlib import Path
import modeling_close_analysis as pipeline


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',required=True,help='New directory under project Modeling/')
    args=p.parse_args()
    root=pipeline.ROOT.resolve()
    destination=(root/args.output_dir).resolve()
    if not destination.is_relative_to(root/'Modeling') or destination==root/'Modeling':
        raise ValueError('Choose a new subdirectory inside project Modeling/.')
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Output directory is not empty; choose a new path. Nothing was overwritten.')
    assert sys.version_info[:2]==(3,13) and sys._is_gil_enabled()
    pipeline.OUT=destination
    pipeline.freeze()
    pipeline.run()


if __name__=='__main__':main()
