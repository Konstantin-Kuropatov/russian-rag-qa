"""Download public immutable snapshots and record checksums, never dataset code."""

import hashlib
import json
import logging
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from .paths import RAW, REPORTS, ROOT

LOGGER = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_all() -> dict:
    spec = json.loads((ROOT / "config/data_sources.json").read_text())
    manifest_path = REPORTS / "data_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": []}
    expected = {row["path"]: row["sha256"] for row in previous["files"]}
    previous_versions = {
        row["path"]: (row["repo_id"], row["revision"]) for row in previous["files"]
    }
    manifest = {"files": []}
    for source in spec["sources"]:
        for filename in source["files"]:
            destination = RAW / source["name"] / filename
            relative = destination.relative_to(ROOT).as_posix()
            if relative in previous_versions and previous_versions[relative] != (
                source["repo_id"],
                source["revision"],
            ):
                raise ValueError(
                    f"Source revision changed for {relative}; use an explicit new snapshot"
                )
            if not destination.exists():
                LOGGER.info("Downloading %s / %s", source["repo_id"], filename)
                cached = hf_hub_download(
                    repo_id=source["repo_id"],
                    repo_type=source["repo_type"],
                    revision=source["revision"],
                    filename=filename,
                    token=False,
                    cache_dir=ROOT / ".cache/huggingface",
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cached, destination)
            checksum = file_sha256(destination)
            if relative in expected and checksum != expected[relative]:
                raise ValueError(
                    f"Checksum mismatch: {relative}; restore this raw file before analysis"
                )
            manifest["files"].append(
                {
                    "repo_id": source["repo_id"],
                    "revision": source["revision"],
                    "path": relative,
                    "size_bytes": destination.stat().st_size,
                    "sha256": checksum,
                }
            )
    REPORTS.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    LOGGER.info("Verified %s files", len(manifest["files"]))
    return manifest


def verify_raw_files() -> None:
    manifest_path = REPORTS / "data_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run `uv run rag-eda download` first")
    manifest = json.loads(manifest_path.read_text())
    spec = json.loads((ROOT / "config/data_sources.json").read_text())
    pinned = {
        f"data/raw/{source['name']}/{filename}": (source["repo_id"], source["revision"])
        for source in spec["sources"]
        for filename in source["files"]
    }
    if set(pinned) != {row["path"] for row in manifest["files"]}:
        raise ValueError("Raw manifest does not match the source specification")
    for row in manifest["files"]:
        if pinned[row["path"]] != (row["repo_id"], row["revision"]):
            raise ValueError(f"Source revision differs from manifest: {row['path']}")
        path = ROOT / row["path"]
        if not path.exists() or file_sha256(path) != row["sha256"]:
            raise ValueError(f"Missing or modified raw file: {row['path']}")
