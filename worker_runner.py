#!/usr/bin/env python3
"""
Worker runner: reads a queue file and processes buckets.
Called by master.py for each GPU.
"""

import argparse
import json
import sys
import traceback

from worker import process_bucket


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--queue-file", type=str, required=True)
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    with open(args.queue_file) as f:
        queue = json.load(f)

    print(f"Worker GPU {args.gpu}: processing {len(queue)} buckets", flush=True)

    for i, (dimension, bucket) in enumerate(queue):
        bucket = tuple(bucket)
        print(f"\n[{i+1}/{len(queue)}] {dimension} {bucket}", flush=True)
        try:
            process_bucket(dimension, bucket, args.gpu, upload=not args.no_upload)
        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            traceback.print_exc()

    print(f"Worker GPU {args.gpu}: all done!", flush=True)


if __name__ == "__main__":
    main()
