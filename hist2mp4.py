#!/usr/bin/env python3
# Create a video from a LUDOpy history
#
# Takes a LUDOpy history file from STDIN

import argparse
import io
import sys

import numpy as np

from ludopy.visualizer import save_hist_video

def main():
    # https://gist.github.com/CMCDragonkai/3c99fd4aabc8278b9e17f50494fcc30a
    # this should be `b'\x93NUMPY'`
    # this can be useful to doing file detection
    np_magic = sys.stdin.buffer.read(6)
    # use the sys.stdin.buffer to read binary data
    np_data = sys.stdin.buffer.read()

    with io.BytesIO(np_magic + np_data) as f_np:
        history = np.load(f_np, allow_pickle=True)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fps",
        help="Framerate of Ludo video",
        type=int,
        default=8
    )
    parser.add_argument(
        "--output-file",
        help="Filename of the output file",
        type=str,
        default="game.mp4",
    )
    args = parser.parse_args()

    save_hist_video(args.output_file, history, fps=args.fps) 

    return

if __name__ == "__main__":
    main()

