#!/usr/bin/env python3
"""Validate the structure of a materialized StackCube dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from tuco.baseline_artifacts import load_curated_selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--selected-ids", type=Path, required=True)
    args = parser.parse_args()
    selected = load_curated_selection(
        args.selected_ids, expected_budget=600, expected_candidates=1200
    ).astype(np.int64)
    selected = np.sort(selected)
    method = json.loads(
        (args.selected_ids.parent / "metadata.json").read_text(encoding="utf-8")
    )["method"]
    expected_ids = np.repeat(selected, 10)
    with h5py.File(args.hdf5, "r") as source:
        if source.attrs.get("state_filter_method") != method:
            raise ValueError("materialized dataset method does not match selection")
        if int(source.attrs.get("visual_repetitions", -1)) != 10:
            raise ValueError("StackCube requires ten visual repeats per selected state")
        if len(source["data"]) != 6000 or int(source["data"].attrs["total"]) != 6000:
            raise ValueError("StackCube materialized dataset must contain 6000 episodes")
        if not np.array_equal(source["meta/selected_physical_state_ids"][:], selected):
            raise ValueError("selected physical states do not match the artifact")
        if not np.array_equal(source["meta/reset_state_ids"][:], expected_ids):
            raise ValueError("reset-state order is not 600 states x 10 repeats")
        for index in (0, 2999, 5999):
            _ = source[f"data/demo_{index}/actions"].shape
    print("validated StackCube 600-state x 10-repeat HDF5", flush=True)


if __name__ == "__main__":
    main()
