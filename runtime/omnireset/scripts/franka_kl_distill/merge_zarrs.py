#!/usr/bin/env python3
"""Stream-concatenate compatible OmniReset Zarr shards without dropping metadata."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import zarr


TRANSIENT_ERRNOS = {
    errno.EAGAIN,
    errno.EINTR,
    errno.EIO,
    errno.ESTALE,
    errno.ETIMEDOUT,
}


def retry_io(label: str, callback, retries: int = 60, delay_s: float = 5.0):
    for attempt in range(retries + 1):
        try:
            return callback()
        except OSError as error:
            if error.errno not in TRANSIENT_ERRNOS or attempt >= retries:
                raise
            print(
                f"[merge] transient I/O error during {label}: {error!r}; "
                f"retry {attempt + 1}/{retries} after {delay_s:.1f}s",
                flush=True,
            )
            time.sleep(delay_s)


def _create_appendable_dataset(group, key: str, source) -> object:
    source_chunks = source.chunks or source.shape
    chunk_len = max(1, min(int(source_chunks[0]), 64 if source.ndim > 2 else 1024))
    return group.create_dataset(
        key,
        shape=(0,) + source.shape[1:],
        chunks=(chunk_len,) + source.shape[1:],
        dtype=source.dtype,
        compressor=source.compressor,
    )


def _append(
    destination,
    source,
    block_frames: int,
    rotate_180: bool = False,
    verify_write: bool = False,
) -> dict[str, str] | None:
    old_size = int(destination.shape[0])
    destination.resize((old_size + int(source.shape[0]),) + destination.shape[1:])
    source_digest = hashlib.sha256() if verify_write else None
    written_digest = hashlib.sha256() if verify_write else None
    for start in range(0, int(source.shape[0]), block_frames):
        end = min(start + block_frames, int(source.shape[0]))
        block = retry_io(
            f"read {getattr(source, 'path', '<array>')}[{start}:{end}]",
            lambda start=start, end=end: np.asarray(source[start:end]),
        )
        if rotate_180:
            if source.ndim != 4:
                raise ValueError(
                    f"180-degree image rotation requires [T,H,W,C], got {source.shape}"
                )
            block = np.asarray(block)[:, ::-1, ::-1, :]
        block = np.ascontiguousarray(block)
        if source_digest is not None:
            source_digest.update(block.tobytes())
        destination_slice = slice(old_size + start, old_size + end)
        retry_io(
            f"write {getattr(destination, 'path', '<array>')}[{old_size + start}:{old_size + end}]",
            lambda destination_slice=destination_slice, block=block: destination.__setitem__(
                destination_slice, block
            ),
        )
        if verify_write:
            written = retry_io(
                f"verify {getattr(destination, 'path', '<array>')}"
                f"[{old_size + start}:{old_size + end}]",
                lambda destination_slice=destination_slice: np.asarray(
                    destination[destination_slice]
                ),
            )
            if not np.array_equal(written, np.asarray(block)):
                raise IOError(
                    f"Zarr write verification failed for frames "
                    f"[{old_size + start}, {old_size + end})"
                )
            written_digest.update(np.ascontiguousarray(written).tobytes())
    if source_digest is None or written_digest is None:
        return None
    return {
        "source_sha256": source_digest.hexdigest(),
        "merged_sha256": written_digest.hexdigest(),
    }


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _directory_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _truncate_to_committed_offsets(
    output,
    data_keys: list[str],
    meta_keys: list[str],
    frame_offset: int,
    episode_offset: int,
) -> None:
    for key in data_keys:
        array = output[f"data/{key}"]
        if int(array.shape[0]) != frame_offset:
            array.resize((frame_offset,) + array.shape[1:])
    for key in (*meta_keys, "episode_ends"):
        array = output[f"meta/{key}"]
        if int(array.shape[0]) != episode_offset:
            array.resize((episode_offset,) + array.shape[1:])


def _resumable_verified_merge(
    args: argparse.Namespace,
    state_path: Path,
    delete_root: Path,
) -> None:
    output_path = Path(args.out).resolve()
    input_paths = [Path(path).resolve() for path in args.inputs]
    delete_root = delete_root.resolve()
    input_strings = [str(path) for path in input_paths]

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("output") != str(output_path) or state.get("inputs") != input_strings:
            raise ValueError("resumable merge state does not match output/input contract")
        if not output_path.is_dir():
            raise FileNotFoundError("resumable merge state exists without its output Zarr")
        output = zarr.open(str(output_path), mode="a")
        data_keys = list(state["data_keys"])
        meta_keys = list(state["meta_keys"])
        frame_offset = int(state["frame_offset"])
        episode_offset = int(state["episode_offset"])
        _truncate_to_committed_offsets(
            output, data_keys, meta_keys, frame_offset, episode_offset
        )
    else:
        if output_path.exists():
            raise FileExistsError(
                f"output exists without resumable merge state: {output_path}"
            )
        first_existing = next((path for path in input_paths if path.is_dir()), None)
        if first_existing is None:
            raise FileNotFoundError("no merge input exists")
        first = retry_io(
            f"open first merge input {first_existing}",
            lambda: zarr.open(str(first_existing), mode="r"),
        )
        data_keys = sorted(first["data"].keys())
        meta_keys = sorted(key for key in first["meta"].keys() if key != "episode_ends")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = zarr.open(str(output_path), mode="w")
        data_output = output.create_group("data")
        meta_output = output.create_group("meta")
        for key in data_keys:
            _create_appendable_dataset(data_output, key, first[f"data/{key}"])
        for key in meta_keys:
            _create_appendable_dataset(meta_output, key, first[f"meta/{key}"])
        _create_appendable_dataset(meta_output, "episode_ends", first["meta/episode_ends"])
        frame_offset = 0
        episode_offset = 0
        state = {
            "schema_version": 1,
            "mode": "resumable_per_block_exact_write_v1",
            "output": str(output_path),
            "inputs": input_strings,
            "delete_verified_inputs_under": str(delete_root),
            "data_keys": data_keys,
            "meta_keys": meta_keys,
            "frame_offset": 0,
            "episode_offset": 0,
            "completed_sources": [],
            "complete": False,
        }
        _write_json_atomic(state_path, state)

    completed_by_path = {
        item["path"]: item for item in state.get("completed_sources", [])
    }
    for input_path in input_paths:
        input_string = str(input_path)
        if input_string in completed_by_path:
            record = completed_by_path[input_string]
            if input_path.is_dir() and bool(record.get("delete_eligible")):
                shutil.rmtree(input_path)
                record["deleted_after_verified_merge"] = True
                _write_json_atomic(state_path, state)
            continue
        if not input_path.is_dir():
            raise FileNotFoundError(f"uncommitted merge input is missing: {input_path}")

        root = retry_io(
            f"open merge input {input_path}",
            lambda input_path=input_path: zarr.open(str(input_path), mode="r"),
        )
        if sorted(root["data"].keys()) != data_keys:
            raise ValueError(f"data keys differ in {input_path}")
        current_meta_keys = sorted(
            key for key in root["meta"].keys() if key != "episode_ends"
        )
        if current_meta_keys != meta_keys:
            raise ValueError(f"episode metadata keys differ in {input_path}")
        ends = retry_io(
            f"read episode_ends from {input_path}",
            lambda: np.asarray(root["meta/episode_ends"][:], dtype=np.int64),
        )
        if ends.ndim != 1 or len(ends) == 0 or np.any(np.diff(ends) <= 0):
            raise ValueError(f"invalid episode_ends in {input_path}")
        frame_count = int(ends[-1])
        array_hashes: dict[str, dict[str, str]] = {}
        for key in data_keys:
            source = root[f"data/{key}"]
            destination = output[f"data/{key}"]
            if int(source.shape[0]) != frame_count:
                raise ValueError(
                    f"{input_path}: data/{key} has {source.shape[0]} frames, "
                    f"expected {frame_count}"
                )
            if source.shape[1:] != destination.shape[1:] or source.dtype != destination.dtype:
                raise ValueError(f"{input_path}: incompatible data/{key} shape or dtype")
            array_hashes[f"data/{key}"] = _append(
                destination,
                source,
                args.block_frames,
                rotate_180=key in set(args.rotate_180_keys),
                verify_write=True,
            )
        for key in meta_keys:
            source = root[f"meta/{key}"]
            destination = output[f"meta/{key}"]
            if int(source.shape[0]) != len(ends):
                raise ValueError(
                    f"{input_path}: meta/{key} has {source.shape[0]} rows, "
                    f"expected {len(ends)}"
                )
            if source.shape[1:] != destination.shape[1:] or source.dtype != destination.dtype:
                raise ValueError(f"{input_path}: incompatible meta/{key} shape or dtype")
            array_hashes[f"meta/{key}"] = _append(
                destination,
                source,
                max(args.block_frames, 1024),
                verify_write=True,
            )
        adjusted_ends = ends + frame_offset
        array_hashes["meta/episode_ends"] = _append(
            output["meta/episode_ends"],
            adjusted_ends,
            max(args.block_frames, 1024),
            verify_write=True,
        )
        if any(
            hashes["source_sha256"] != hashes["merged_sha256"]
            for hashes in array_hashes.values()
        ):
            raise IOError(f"write digest mismatch after merging {input_path}")

        delete_eligible = input_path.is_relative_to(delete_root)
        source_size_bytes = _directory_size_bytes(input_path) if delete_eligible else None
        record = {
            "path": input_string,
            "episodes": int(len(ends)),
            "frames": frame_count,
            "frame_start": frame_offset,
            "frame_end": frame_offset + frame_count,
            "episode_start": episode_offset,
            "episode_end": episode_offset + len(ends),
            "episode_ends": adjusted_ends.tolist(),
            "array_hashes": array_hashes,
            "collection_config": dict(root.attrs.get("collection_config", {})),
            "write_verified": True,
            "delete_eligible": delete_eligible,
            "deleted_after_verified_merge": False,
            "source_size_bytes": source_size_bytes,
        }
        frame_offset += frame_count
        episode_offset += len(ends)
        state["frame_offset"] = frame_offset
        state["episode_offset"] = episode_offset
        state["completed_sources"].append(record)
        completed_by_path[input_string] = record
        _write_json_atomic(state_path, state)
        if delete_eligible:
            shutil.rmtree(input_path)
            record["deleted_after_verified_merge"] = True
            _write_json_atomic(state_path, state)
        print(
            f"[merge-resume] + {input_path}: frames={frame_count} "
            f"episodes={len(ends)} deleted_local={delete_eligible}",
            flush=True,
        )

    source_configs = [
        item.get("collection_config", {}) for item in state["completed_sources"]
    ]
    collection_config = dict(source_configs[0])
    collection_config.update(
        {
            "merged_shard_paths": input_strings,
            "merged_shard_collection_configs": source_configs,
            "merged_total_frames": frame_offset,
            "merged_total_episodes": episode_offset,
            "merged_image_transforms": {
                key: {"rotation_deg": 180, "operation": "rotate_180_during_merge"}
                for key in sorted(set(args.rotate_180_keys))
            },
            "resumable_verified_merge": True,
        }
    )
    output.attrs["collection_config"] = collection_config
    output.attrs["merge_write_verified"] = True
    state["complete"] = True
    state["completed_at_unix_s"] = time.time()
    _write_json_atomic(state_path, state)
    output.attrs["resumable_merge_write_verification"] = state
    print(
        f"[merge-resume] done: frames={frame_offset} episodes={episode_offset} "
        f"state={state_path} -> {output_path}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--in", dest="inputs", nargs="+", required=True)
    parser.add_argument("--block_frames", type=int, default=128)
    parser.add_argument(
        "--verify_write",
        action="store_true",
        help="Read every written block back and require exact equality before success",
    )
    parser.add_argument(
        "--rotate_180_keys",
        nargs="*",
        default=(),
        help="Image data keys to rotate 180 degrees while writing the merged dataset",
    )
    parser.add_argument(
        "--resume_verified",
        action="store_true",
        help="Resume at verified shard boundaries and record per-array write digests",
    )
    parser.add_argument(
        "--delete_verified_inputs_under",
        type=Path,
        help="Delete inputs below this root only after their exact write is committed",
    )
    parser.add_argument("--state_report", type=Path)
    args = parser.parse_args()
    if args.block_frames < 1:
        parser.error("--block_frames must be positive")

    output_root_string = os.environ.get("OUTPUT_ROOT")
    reference_root_string = os.environ.get("REFERENCE_ROOT")
    auto_cross_root_resume = bool(
        output_root_string
        and reference_root_string
        and Path(output_root_string).resolve() != Path(reference_root_string).resolve()
        and Path(args.out).resolve().is_relative_to(Path(output_root_string).resolve())
    )
    if auto_cross_root_resume:
        args.resume_verified = True
        if args.delete_verified_inputs_under is None:
            args.delete_verified_inputs_under = (
                Path(output_root_string) / "visual/recovery/chunks"
            )
    if args.resume_verified:
        if args.delete_verified_inputs_under is None:
            parser.error("--resume_verified requires --delete_verified_inputs_under")
        state_path = args.state_report or Path(args.out).with_name(
            f".{Path(args.out).name}.merge_state.json"
        )
        _resumable_verified_merge(args, state_path.resolve(), args.delete_verified_inputs_under)
        return
    if Path(args.out).exists():
        raise FileExistsError(f"output already exists: {args.out}")

    roots = [zarr.open(path, mode="r") for path in args.inputs]
    data_keys = sorted(roots[0]["data"].keys())
    rotate_180_keys = set(args.rotate_180_keys)
    unknown_rotate_keys = rotate_180_keys.difference(data_keys)
    if unknown_rotate_keys:
        raise ValueError(f"rotation keys are absent from the input data: {sorted(unknown_rotate_keys)}")
    meta_keys = sorted(key for key in roots[0]["meta"].keys() if key != "episode_ends")
    for path, root in zip(args.inputs[1:], roots[1:]):
        if sorted(root["data"].keys()) != data_keys:
            raise ValueError(f"data keys differ in {path}")
        current_meta_keys = sorted(key for key in root["meta"].keys() if key != "episode_ends")
        if current_meta_keys != meta_keys:
            raise ValueError(f"episode metadata keys differ in {path}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    output = zarr.open(args.out, mode="w")
    data_output = output.create_group("data")
    meta_output = output.create_group("meta")
    data_arrays = {
        key: _create_appendable_dataset(data_output, key, roots[0][f"data/{key}"])
        for key in data_keys
    }
    meta_arrays = {
        key: _create_appendable_dataset(meta_output, key, roots[0][f"meta/{key}"])
        for key in meta_keys
    }

    episode_ends = []
    frame_offset = 0
    total_episodes = 0
    source_configs = []
    for path, root in zip(args.inputs, roots):
        ends = np.asarray(root["meta/episode_ends"], dtype=np.int64)
        if ends.ndim != 1 or len(ends) == 0 or np.any(np.diff(ends) <= 0):
            raise ValueError(f"invalid episode_ends in {path}")
        frame_count = int(ends[-1])
        for key in data_keys:
            source = root[f"data/{key}"]
            if int(source.shape[0]) != frame_count:
                raise ValueError(
                    f"{path}: data/{key} has {source.shape[0]} frames, expected {frame_count}"
                )
            destination = data_arrays[key]
            if source.shape[1:] != destination.shape[1:] or source.dtype != destination.dtype:
                raise ValueError(f"{path}: incompatible data/{key} shape or dtype")
            _append(
                destination,
                source,
                args.block_frames,
                rotate_180=key in rotate_180_keys,
                verify_write=args.verify_write,
            )
        for key in meta_keys:
            source = root[f"meta/{key}"]
            if int(source.shape[0]) != len(ends):
                raise ValueError(
                    f"{path}: meta/{key} has {source.shape[0]} rows, expected {len(ends)}"
                )
            destination = meta_arrays[key]
            if source.shape[1:] != destination.shape[1:] or source.dtype != destination.dtype:
                raise ValueError(f"{path}: incompatible meta/{key} shape or dtype")
            _append(
                destination,
                source,
                args.block_frames,
                verify_write=args.verify_write,
            )
        episode_ends.append(ends + frame_offset)
        frame_offset += frame_count
        total_episodes += len(ends)
        source_configs.append(dict(root.attrs.get("collection_config", {})))
        print(f"[merge] + {path}: frames={frame_count} episodes={len(ends)}")

    meta_output.create_dataset(
        "episode_ends", data=np.concatenate(episode_ends).astype(np.int64, copy=False)
    )
    collection_config = dict(source_configs[0])
    collection_config.update(
        {
            "merged_shard_paths": [os.path.abspath(path) for path in args.inputs],
            "merged_shard_collection_configs": source_configs,
            "merged_total_frames": frame_offset,
            "merged_total_episodes": total_episodes,
            "merged_image_transforms": {
                key: {"rotation_deg": 180, "operation": "rotate_180_during_merge"}
                for key in sorted(rotate_180_keys)
            },
        }
    )
    output.attrs["collection_config"] = collection_config
    output.attrs["merge_write_verified"] = bool(args.verify_write)
    print(
        f"[merge] done: frames={frame_offset} episodes={total_episodes} "
        f"verify_write={args.verify_write} -> {args.out}"
    )


if __name__ == "__main__":
    main()
