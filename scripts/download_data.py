#!/usr/bin/env python3
"""Download the public dataset used by the single-simulator experiments."""

from __future__ import annotations

import argparse
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


def fetch(source: str, destination: Path) -> None:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https", "file"}:
        with urllib.request.urlopen(source) as response, destination.open("wb") as output:
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


def safe_tar_members(archive: tarfile.TarFile, root: Path) -> None:
    base = root.resolve()
    for member in archive.getmembers():
        target = (root / member.name).resolve()
        if target != base and base not in target.parents:
            raise ValueError(f"archive member escapes destination: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"links are not allowed in data bundles: {member.name}")


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
    parser.add_argument(
        "--source",
        help=(
            "Archive path or URL. Defaults to the public Diffusion Policy "
            "RoboMimic archive."
        ),
    )
    args = parser.parse_args()
    source = args.source
    if source is None:
        source = ROBOMIMIC_URL
    destination = args.destination.expanduser().resolve()
    if any(destination.iterdir()) if destination.exists() else False:
        parser.error(f"destination must be absent or empty: {destination}")

    with tempfile.TemporaryDirectory(prefix="tuco-download-") as temporary:
        archive_path = Path(temporary) / "dataset.archive"
        print(f"[download] {source}", flush=True)
        fetch(source, archive_path)
        extract(archive_path, destination)
    print(f"[installed] {destination}", flush=True)


if __name__ == "__main__":
    main()
