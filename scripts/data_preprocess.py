r"""将下载好的 Pile .jsonl 或 .jsonl.zst 文件分词并写入 HDF5。

输入 / 输出一览
----------------
输入（原始文本，尚未分词，仓库之外）：
    --train_dir    训练集原始 .jsonl / .jsonl.zst 所在目录（默认 data/train）
    --val_dir      验证集原始 .jsonl / .jsonl.zst 所在目录（默认 data/val）
    本地调试时原始数据通常指向例如 ~/projects/datasets/pretrain/pile/val_small。

输出（分词后的 token，整数 id 数组）：
    --out_train_file   训练集 token HDF5（默认 data/train/pile_train.h5）
    --out_val_file     验证集 token HDF5（默认 data/val/pile_dev.h5）
    --debug_tokens_json 仅本地调试：额外把验证集完整 token 序列写成 JSON
                        （例如 data/tokens_local/pile_val_small_tokens.json）

目录约定：
    data/tokens_local/   仅本地调试用的小样本 token 产物（含肉眼查看的 JSON）
    data/train/          正式训练的 token（pile_train.h5）
    data/val/            正式验证的 token（pile_dev.h5）

默认每个文件只处理前 1000 条记录；传入 --all_data 处理完整文件。

本地验证集样本调试示例（在仓库根目录运行）：
    python3 scripts/data_preprocess.py --val_only \
        --val_dir /path/to/val_small \
        --out_val_file data/tokens_local/pile_val_small.h5 \
        --debug_tokens_json data/tokens_local/pile_val_small_tokens.json \
        --all_data

仅本地调试用选项说明：
    --debug_tokens_json 仅供本地调试：把完整 token 序列额外写成 JSON 方便肉眼
    核对分词结果；服务器正式跑数据时不要传该参数。同样地，--write_chunk_tokens
    在生产环境保持默认值、--max_data 用 --all_data 覆盖即可。
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
    debug_tokens_json: str | None = None,
) -> int:
    """按文件名顺序处理目录中的 JSONL 文件，返回写入的 token 数。

    参数：
        input_dir: 存放 .jsonl 或 .jsonl.zst 的目录，按文件名排序处理。
        output_file: 分词后 flat token 数组的 HDF5 输出路径。
        tokenizer_name: tiktoken 编码名称（如 "r50k_base"）。
        max_data: 每个原始文件最多读取的记录数；None 表示读取全部。
        write_chunk_tokens: 每批向 HDF5 flush 的 token 数阈值。
        debug_tokens_json: 仅本地调试用。若提供，额外把完整 token 序列写成
            JSON 文件方便肉眼核对（服务器正式跑数据时保持默认的 None）。
    """
    import h5py
    import numpy as np
    import tiktoken
    import zstandard as zstd
    from tqdm import tqdm

    input_path = Path(input_dir)
    files = sorted(
        [*input_path.glob("*.jsonl"), *input_path.glob("*.jsonl.zst")]
    )
    if not files:
        raise FileNotFoundError(f"没有找到 .jsonl 或 .jsonl.zst 文件：{input_path}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 先写到一个 .part 临时文件，全部写完后再原子地 replace 成正式文件名。
    # 这样写入中途崩溃不会留下一个不完整的 HDF5，训练时也不会读到半成品。
    part_path = output_path.with_name(output_path.name + ".part")
    encoder = tiktoken.get_encoding(tokenizer_name)

    total_tokens = 0
    total_lines = 0
    kept_docs = 0
    skipped_docs = 0
    # buffer 是内存中的 token 缓冲：凑够 write_chunk_tokens 个就 flush 一次到 HDF5，
    # 避免每读一条记录就 resize 一次 dataset（HDF5 的 resize 是 O(n) 的，频繁调用会很慢）。
    buffer: list[int] = []

    # 磁盘上写入的是分词后的 HDF5 .part 文件，不是解压后的 JSONL。
    # 解压只在内存中流式进行（见下方 zstd.open），最终落盘的是整数 token id 数组。
    with h5py.File(part_path, "w") as out_file:
        # 创建可伸缩的 1-D dataset：一开始 shape 是 (0,)，maxshape 是 (None,)，
        # 之后每批 flush 时通过 resize 逐步扩容。
        tokens = out_file.create_dataset(
            "tokens",
            shape=(0,),
            maxshape=(None,),
            dtype="i4",
            chunks=(write_chunk_tokens,),
        )

        def flush() -> None:
            """把 buffer 里的 token 追加写入 HDF5 dataset 尾部，然后清空 buffer。"""
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
            # 若文件是 .jsonl.zst 就用 zstd 流式解压，否则当作普通文本直接打开。
            # zstd.open 随读取逐步解压，不生成单独的解压后 JSONL 文件。
            # 流式解压并非 zstd 独有；gzip、bz2、lzma 也提供按流读取接口。
            open_input = zstd.open if file.name.endswith(".jsonl.zst") else open
            with open_input(file, "rt", encoding="utf-8") as stream:
                # max_data 为 None 时读整个文件，否则只读前 max_data 行。
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

                    # 把一条文档编码为 token id，并在末尾附上 EOT token（50256），
                    # 作为文档之间的分隔符，让模型知道一段文本在哪里结束。
                    buffer.extend(encoder.encode_ordinary(text))
                    buffer.append(encoder.eot_token)
                    kept_docs += 1
                    if len(buffer) >= write_chunk_tokens:
                        flush()
            # 每个文件读完都 flush 一次，避免 buffer 里的 token 跨文件残留。
            flush()

    # 所有文件处理完、HDF5 关闭后，原子地把 .part 重命名为正式输出路径。
    os.replace(part_path, output_path)

    # 仅本地调试：把完整 token 序列额外写成 JSON，方便肉眼查看/核对分词结果。
    # 服务器正式跑数据时 debug_tokens_json 为 None，这一段不会执行，也不会产生额外开销。
    if debug_tokens_json is not None:
        # 重新打开刚写好的 HDF5 取出全部 token，避免在内存里额外维护第二份大数组。
        import h5py as _h5py

        with _h5py.File(output_path, "r") as hf:
            token_list = hf["tokens"][...].tolist()
        debug_path = Path(debug_tokens_json)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("w", encoding="utf-8") as fh:
            json.dump(token_list, fh)
        print(
            f"调试用 token JSON 已写出（{len(token_list):,} 个 token）：{debug_path}"
        )

    print(
        f"完成：{output_path}；读取 {total_lines} 条，写入 {kept_docs} 条，"
        f"跳过 {skipped_docs} 条，token 数 {total_tokens:,}"
    )
    return total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 Pile 的 .jsonl 或 .jsonl.zst 训练集和验证集分词为 HDF5。"
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
        "--val_only", action="store_true", help="只处理验证集，不要求训练目录"
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
    parser.add_argument(
        "--debug_tokens_json",
        default=None,
        help=(
            "仅本地调试用：把验证集的完整 token 序列额外写成 JSON 文件方便肉眼查看。"
            "服务器正式跑数据时不要传此参数。"
        ),
    )
    args = parser.parse_args()

    input_dirs = (args.val_dir,) if args.val_only else (args.train_dir, args.val_dir)
    for directory in input_dirs:
        if not Path(directory).is_dir():
            parser.error(f"目录不存在：{directory}")
    if (
        not args.val_only
        and Path(args.out_train_file).resolve() == Path(args.out_val_file).resolve()
    ):
        parser.error("训练集和验证集输出路径不能相同")

    max_data = None if args.all_data else args.max_data # all_data可以覆盖掉max_data
    if max_data is None:
        print("处理每个文件中的全部记录。")
    else:
        print(f"每个文件最多处理 {max_data} 条记录。")
    if not args.val_only:
        process_files(
            args.train_dir,
            args.out_train_file,
            args.tokenizer_name,
            max_data,
            args.write_chunk_tokens,
            # 训练集不写调试 JSON，避免服务器上产生超大文件。
            None,
        )
    process_files(
        args.val_dir,
        args.out_val_file,
        args.tokenizer_name,
        max_data,
        args.write_chunk_tokens,
        # 仅本地调试用：把验证集 token 序列额外写成 JSON；生产环境传 None（见 docstring）。
        args.debug_tokens_json,
    )


if __name__ == "__main__":
    main()
