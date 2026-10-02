#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "用法：bash tutorials/download_pretrain_data_with_curl.sh /path/to/pile/train" >&2
  exit 2
fi

train_dir="$1"
mkdir -p "$train_dir"

curl -fL --retry 5 --retry-all-errors -C - \
  -o "$train_dir/00.jsonl.zst" \
  'https://hf-mirror.com/datasets/monology/pile-uncopyrighted/resolve/main/train/00.jsonl.zst'
