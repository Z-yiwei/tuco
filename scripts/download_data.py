#!/usr/bin/env python3
"""Download the public dataset used by the single-simulator experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse


ROBOMIMIC_URL = (
    "https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip"
)


def fetch(source: str, destination: Path, workers: int = 8) -> None:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        request = urllib.request.Request(source, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            size = int(response.headers.get("Content-Length", 0))
            ranges = response.headers.get("Accept-Ranges") == "bytes"
        if size and ranges:
            block_size = 8 * 1024 * 1024
            blocks = list(range(0, size, block_size))

            def download_block(start):
                end = min(start + block_size, size) - 1
                for attempt in range(3):
                    try:
                        request = urllib.request.Request(source, headers={"Range": f"bytes={start}-{end}"})
                        with urllib.request.urlopen(request, timeout=60) as response:
                            expected = f"bytes {start}-{end}/{size}"
                            if response.status != 206 or response.headers.get("Content-Range") != expected:
                                raise IOError("Server did not honor the requested byte range")
                            data = response.read()
                        if len(data) != end - start + 1:
                            raise IOError("Incomplete download block")
                        with destination.open("r+b") as output:
                            output.seek(start)
                            output.write(data)
                        return len(data)
                    except (OSError, ValueError):
                        if attempt == 2:
                            raise

            with destination.open("wb") as output:
                output.truncate(size)
            completed = 0
            with ThreadPoolExecutor(max_workers=workers) as executor:
                jobs = [executor.submit(download_block, start) for start in blocks]
                for future in as_completed(jobs):
                    completed += future.result()
                    print(f"[download] {completed / size:.0%} ({completed // 1048576}/{size // 1048576} MiB)", flush=True)
            return
    if parsed.scheme in {"http", "https", "file"}:
        with urllib.request.urlopen(source, timeout=60) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    else:
        local = Path(source).expanduser().resolve()
        if not local.is_file():
            raise FileNotFoundError(local)
        shutil.copyfile(local, destination)


def safe_zip_members(archive: zipfile.ZipFile, root: Path) -> None:
    base = root.resolve()
    for member in archive.infolist():
        target = (root / member.filename).resolve()
        if target != base and base not in target.parents:
            raise ValueError(f"archive member escapes destination: {member.filename}")
        if target.exists() and not (member.is_dir() and target.is_dir()):
            raise FileExistsError(f"refusing to replace an existing dataset file: {target}")


def safe_tar_members(archive: tarfile.TarFile, root: Path) -> None:
    base = root.resolve()
    for member in archive.getmembers():
        target = (root / member.name).resolve()
        if target != base and base not in target.parents:
            raise ValueError(f"archive member escapes destination: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"links are not allowed in data bundles: {member.name}")
        if target.exists() and not (member.isdir() and target.is_dir()):
            raise FileExistsError(f"refusing to replace an existing dataset file: {target}")


def extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            safe_zip_members(archive, destination)
            archive.extractall(destination)
        return
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            safe_tar_members(archive, destination)
            archive.extractall(destination)
    except tarfile.ReadError as error:
        raise ValueError(f"unsupported archive: {archive_path}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("setting", choices=("single-sim",))
    parser.add_argument("--destination", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--source",
        help=(
            "Archive path or URL. Defaults to the public Diffusion Policy "
            "RoboMimic archive."
        ),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    source = args.source
    if source is None:
        source = ROBOMIMIC_URL
    destination = args.destination.expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        parser.error(f"destination is not a directory: {destination}")

    with tempfile.TemporaryDirectory(prefix="tuco-download-") as temporary:
        archive_path = Path(temporary) / "dataset.archive"
        print(f"[download] {source}", flush=True)
        fetch(source, archive_path, args.workers)
        extract(archive_path, destination)
    print(f"[installed] {destination}", flush=True)


if __name__ == "__main__":
    main()
