"""将下载好的 Pile .jsonl.zst 文件分词并写入 HDF5。

默认每个文件只处理前 1000 条记录，便于快速检查；传入 --all_data 处理完整文件。
"""

import argparse
import json
import os
from itertools import islice
from pathlib import Path

WRITE_CHUNK_TOKENS = 1_000_000


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def process_files(
    input_dir: str,
    output_file: str,
    tokenizer_name: str,
    max_data: int | None = None,
    write_chunk_tokens: int = WRITE_CHUNK_TOKENS,
) -> int:
    """按文件名顺序处理目录中的 .jsonl.zst 文件，返回写入的 token 数。"""
    import h5py
    import numpy as np
    import tiktoken
    import zstandard as zstd
    from tqdm import tqdm

    input_path = Path(input_dir)
    files = sorted(input_path.glob("*.jsonl.zst"))
    if not files:
        raise FileNotFoundError(f"没有找到 .jsonl.zst 文件：{input_path}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_name(output_path.name + ".part")
    encoder = tiktoken.get_encoding(tokenizer_name)

    total_tokens = 0
    total_lines = 0
    kept_docs = 0
    skipped_docs = 0
    buffer: list[int] = []

    with h5py.File(part_path, "w") as out_file:
        tokens = out_file.create_dataset(
            "tokens",
            shape=(0,),
            maxshape=(None,),
            dtype="i4",
            chunks=(write_chunk_tokens,),
        )

        def flush() -> None:
            nonlocal total_tokens
            if not buffer:
                return
            values = np.asarray(buffer, dtype=np.int32)
            new_total = total_tokens + len(values)
            tokens.resize((new_total,))
            tokens[total_tokens:new_total] = values
            total_tokens = new_total
            buffer.clear()

        for file in files:
            print(f"正在处理：{file}")
            with zstd.open(file, "rt", encoding="utf-8") as stream:
                lines = islice(stream, max_data) if max_data is not None else stream
                for line in tqdm(lines, total=max_data, desc=file.name, unit="条"):
                    total_lines += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        skipped_docs += 1
                        continue
                    if not isinstance(record, dict):
                        skipped_docs += 1
                        continue
                    text = record.get("text")
                    if not isinstance(text, str) or not text:
                        skipped_docs += 1
                        continue

                    buffer.extend(encoder.encode_ordinary(text))
                    buffer.append(encoder.eot_token)
                    kept_docs += 1
                    if len(buffer) >= write_chunk_tokens:
                        flush()
            flush()

    os.replace(part_path, output_path)
    print(
        f"完成：{output_path}；读取 {total_lines} 条，写入 {kept_docs} 条，"
        f"跳过 {skipped_docs} 条，token 数 {total_tokens:,}"
    )
    return total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 Pile 训练集和验证集从 .jsonl.zst 分词为 HDF5。"
    )
    parser.add_argument("--train_dir", default="data/train", help="训练集原始文件目录")
    parser.add_argument("--val_dir", default="data/val", help="验证集原始文件目录")
    parser.add_argument(
        "--out_train_file", default="data/train/pile_train.h5", help="训练集 HDF5 路径"
    )
    parser.add_argument(
        "--out_val_file", default="data/val/pile_dev.h5", help="验证集 HDF5 路径"
    )
    parser.add_argument(
        "--tokenizer_name", default="r50k_base", help="tiktoken 编码名称"
    )
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument(
        "--max_data",
        type=positive_int,
        default=1000,
        help="每个原始文件最多处理的记录数，默认 1000",
    )
    limit.add_argument(
        "--all_data", action="store_true", help="处理每个原始文件中的全部记录"
    )
    parser.add_argument(
        "--write_chunk_tokens",
        type=positive_int,
        default=WRITE_CHUNK_TOKENS,
        help=f"每批写入 HDF5 的 token 数，默认 {WRITE_CHUNK_TOKENS:,}",
    )
    args = parser.parse_args()

    for directory in (args.train_dir, args.val_dir):
        if not Path(directory).is_dir():
            parser.error(f"目录不存在：{directory}")
    if Path(args.out_train_file).resolve() == Path(args.out_val_file).resolve():
        parser.error("训练集和验证集输出路径不能相同")

    max_data = None if args.all_data else args.max_data
    if max_data is None:
        print("处理每个文件中的全部记录。")
    else:
        print(f"每个文件最多处理 {max_data} 条记录。")
    process_files(
        args.train_dir,
        args.out_train_file,
        args.tokenizer_name,
        max_data,
        args.write_chunk_tokens,
    )
    process_files(
        args.val_dir,
        args.out_val_file,
        args.tokenizer_name,
        max_data,
        args.write_chunk_tokens,
    )


if __name__ == "__main__":
    main()
