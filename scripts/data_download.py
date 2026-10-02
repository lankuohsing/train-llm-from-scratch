"""Download Pile shards with HTTP range based resume support."""

import argparse
import os
import re
from typing import List

import requests
from tqdm import tqdm

BASE_URL = "https://hf-mirror.com/datasets/monology/pile-uncopyrighted/resolve/main"
VAL_URL = f"{BASE_URL}/val.jsonl.zst"
TRAIN_URLS = [f"{BASE_URL}/train/{i:02d}.jsonl.zst" for i in range(65)]

CHUNK_SIZE = 1024 * 1024
TIMEOUT = (10, 60)
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def remote_size(url: str) -> int:
    """Get the full file size after following the mirror redirect."""
    with requests.head(url, allow_redirects=True, timeout=TIMEOUT) as response:
        response.raise_for_status()
        length = response.headers.get("Content-Length")
    if length is None:
        raise RuntimeError(f"Server did not provide Content-Length: {url}")
    size = int(length)
    if size <= 0:
        raise RuntimeError(f"Invalid file size {size}: {url}")
    return size


def check_zstd_header(path: str) -> None:
    """Reject an HTML error page or other non-zstd content before resuming."""
    with open(path, "rb") as stream:
        if stream.read(4) != ZSTD_MAGIC:
            raise RuntimeError(
                f"{path} does not start with a zstd frame; inspect or remove it before retrying"
            )


def download_file(url: str, file_name: str) -> None:
    """Download to a .part file and resume it on the next run."""
    expected_size = remote_size(url)
    part_name = f"{file_name}.part"
    os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)

    if os.path.exists(file_name):
        current_size = os.path.getsize(file_name)
        if current_size == expected_size:
            check_zstd_header(file_name)
            print(f"Already complete: {file_name}")
            return
        if current_size > expected_size:
            raise RuntimeError(
                f"{file_name} is larger than the remote file; inspect or remove it"
            )
        if os.path.exists(part_name):
            raise RuntimeError(
                f"Both {file_name} and {part_name} exist; inspect them before retrying"
            )
        os.replace(file_name, part_name)
        print(f"Moved incomplete file to {part_name}")

    offset = os.path.getsize(part_name) if os.path.exists(part_name) else 0
    if offset > expected_size:
        raise RuntimeError(
            f"{part_name} is larger than the remote file; inspect or remove it"
        )
    if offset == expected_size:
        check_zstd_header(part_name)
        os.replace(part_name, file_name)
        print(f"Already complete: {file_name}")
        return
    if 0 < offset < len(ZSTD_MAGIC):
        print(f"{part_name} is too short to resume; restarting it")
        offset = 0
    elif offset:
        check_zstd_header(part_name)

    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT) as response:
        response.raise_for_status()
        if offset and response.status_code == 206:
            match = CONTENT_RANGE.fullmatch(
                response.headers.get("Content-Range", "").strip()
            )
            if (
                match is None
                or int(match.group(1)) != offset
                or int(match.group(2)) != expected_size - 1
                or int(match.group(3)) != expected_size
            ):
                raise RuntimeError(f"Unexpected Content-Range for {url}")
            mode = "ab"
            print(f"Resuming {file_name} at byte {offset:,} of {expected_size:,}")
        elif response.status_code == 200:
            if offset:
                print(f"Server ignored Range; restarting {file_name} from byte 0")
            offset = 0
            mode = "wb"
        else:
            raise RuntimeError(f"Unexpected HTTP {response.status_code} for {url}")

        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) != expected_size - offset:
            raise RuntimeError(f"Unexpected Content-Length for {url}")

        with open(part_name, mode) as output, tqdm(
            total=expected_size,
            initial=offset,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=os.path.basename(file_name),
        ) as progress:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    output.write(chunk)
                    progress.update(len(chunk))

    actual_size = os.path.getsize(part_name)
    if actual_size != expected_size:
        raise RuntimeError(
            f"Incomplete download: {part_name} has {actual_size:,} of "
            f"{expected_size:,} bytes; rerun to resume"
        )
    check_zstd_header(part_name)
    os.replace(part_name, file_name)
    print(f"Downloaded: {file_name}")


def download_dataset(
    val_url: str,
    train_urls: List[str],
    val_dir: str,
    train_dir: str,
    max_train_files: int,
) -> None:
    """Download the validation file and the requested training shards."""
    download_file(val_url, os.path.join(val_dir, "val.jsonl.zst"))
    for idx, url in enumerate(train_urls[:max_train_files]):
        download_file(url, os.path.join(train_dir, f"{idx:02d}.jsonl.zst"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Pile dataset shards.")
    parser.add_argument(
        "--train_max", type=int, default=1, help="Number of training shards to download."
    )
    parser.add_argument(
        "--train_dir", default="data/train", help="Directory for training data."
    )
    parser.add_argument(
        "--val_dir", default="data/val", help="Directory for validation data."
    )
    args = parser.parse_args()
    if not 0 <= args.train_max <= len(TRAIN_URLS):
        parser.error(f"--train_max must be between 0 and {len(TRAIN_URLS)}")

    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.val_dir, exist_ok=True)
    download_dataset(VAL_URL, TRAIN_URLS, args.val_dir, args.train_dir, args.train_max)
    print("Dataset downloaded successfully.")


if __name__ == "__main__":
    main()
