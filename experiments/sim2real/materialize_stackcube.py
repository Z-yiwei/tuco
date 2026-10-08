#!/usr/bin/env python3
"""Build the paper's 600-state x 10-repeat StackCube HDF5 view."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import zarr

from tuco.baseline_artifacts import load_curated_selection


ROOT = Path(__file__).resolve().parents[2]
CONVERTER = ROOT / "third_party/cupid/scripts/tools/convert_omnireset_image_to_cupid.py"
ENV_NAME = "OmniReset-StackCube-FR3-XY5-T0-v020-Normal-JointTarget-Image"


def state_ids(path: Path) -> np.ndarray:
    root = zarr.open(str(path), mode="r")
    for key in ("meta/reset_state_indices", "meta/reset_state_ids"):
        if key in root:
            return np.asarray(root[key], dtype=np.int64)
    raise KeyError(f"no reset-state IDs in {path}")


def hdf5_episode_count(path: Path) -> int:
    with h5py.File(path, "r") as source:
        if "data" not in source:
            raise KeyError(f"{path}: missing data group")
        count = len(source["data"])
        if int(source["data"].attrs.get("total", -1)) != count:
            raise ValueError(f"{path}: inconsistent HDF5 episode count")
        return count


def episode_map(ids: np.ndarray) -> dict[int, list[int]]:
    result: dict[int, list[int]] = defaultdict(list)
    for episode, state_id in enumerate(ids.tolist()):
        result[int(state_id)].append(episode)
    return result


def convert_variant(zarr_path: Path, hdf5_path: Path, python: str) -> None:
    if hdf5_path.is_file():
        return
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            python,
            str(CONVERTER),
            "--zarr_path",
            str(zarr_path),
            "--hdf5_path",
            str(hdf5_path),
            "--val_ratio",
            "0.04",
            "--env_name",
            ENV_NAME,
            "--image_compression",
            "none",
            "--image_size",
            "84",
        ],
        cwd=ROOT,
        check=True,
    )


def load_variant(variant: int, root: Path, python: str) -> dict[int, tuple[Path, int]]:
    chunks = sorted((root / f"variant{variant}" / "chunks").glob("*.zarr"))
    if not chunks:
        raise FileNotFoundError(f"variant {variant} has no Zarr chunks under {root}")
    result: dict[int, tuple[Path, int]] = {}
    for chunk in chunks:
        hdf5 = root / f"variant{variant}" / "hdf5" / chunk.stem / "image.hdf5"
        convert_variant(chunk, hdf5, python)
        ids = state_ids(chunk)
        if hdf5_episode_count(hdf5) != len(ids):
            raise ValueError(f"{hdf5}: episode order cannot match {chunk}")
        for episode, state_id in enumerate(ids.tolist()):
            state_id = int(state_id)
            if state_id in result:
                raise ValueError(f"duplicate state {state_id} in variant {variant}")
            result[state_id] = (hdf5, episode)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zarr", type=Path, required=True)
    parser.add_argument("--source-hdf5", type=Path, required=True)
    parser.add_argument("--variants-root", type=Path, required=True)
    parser.add_argument("--selected-ids", type=Path, required=True)
    parser.add_argument("--output-hdf5", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    output = args.output_hdf5.resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")

    selected = load_curated_selection(
        args.selected_ids, expected_budget=600, expected_candidates=1200
    ).astype(np.int64)
    selected = np.sort(selected)
    selection_method = json.loads(
        (args.selected_ids.parent / "metadata.json").read_text(encoding="utf-8")
    )["method"]
    source_zarr = args.source_zarr.resolve()
    source_hdf5 = args.source_hdf5.resolve()
    original = episode_map(state_ids(source_zarr))
    if len(original) != 1200 or any(len(value) != 5 for value in original.values()):
        raise ValueError("source dataset must be exactly 1200 states x 5 repeats")
    if hdf5_episode_count(source_hdf5) != 6000:
        raise ValueError("source HDF5 must preserve all 6,000 source episodes")
    variants = {
        index: load_variant(index, args.variants_root.resolve(), args.python)
        for index in range(6, 11)
    }
    for state_id in selected.tolist():
        if state_id not in original:
            raise KeyError(f"selected state {state_id} is absent from the source pool")
        for variant in variants.values():
            if state_id not in variant:
                raise KeyError(f"selected state {state_id} is absent from a top-up")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with h5py.File(source_hdf5, "r") as source, h5py.File(temporary, "w") as target:
            for key, value in source.attrs.items():
                target.attrs[key] = value
            target.attrs["state_filter_method"] = selection_method
            target.attrs["visual_repetitions"] = 10
            data = target.create_group("data")
            data.attrs["env_args"] = source["data"].attrs["env_args"]
            data.attrs["total"] = 6000
            reset_ids = []
            output_episode = 0
            for state_id in selected.tolist():
                for source_episode in original[state_id]:
                    source.copy(f"data/demo_{source_episode}", data,
                                name=f"demo_{output_episode}", expand_external=True,
                                expand_soft=True)
                    reset_ids.append(state_id)
                    output_episode += 1
                for variant_index in range(6, 11):
                    hdf5, source_episode = variants[variant_index][state_id]
                    with h5py.File(hdf5, "r") as variant_source:
                        variant_source.copy(f"data/demo_{source_episode}", data,
                                            name=f"demo_{output_episode}",
                                            expand_external=True, expand_soft=True)
                    reset_ids.append(state_id)
                    output_episode += 1
            metadata = target.create_group("meta")
            metadata.create_dataset(
                "reset_state_ids", data=np.asarray(reset_ids, np.int64)
            )
            metadata.create_dataset("selected_physical_state_ids", data=selected)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"materialized 600 states x 10 repeats: {output}", flush=True)


if __name__ == "__main__":
    main()
