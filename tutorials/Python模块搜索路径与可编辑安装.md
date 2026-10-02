# Python 模块搜索路径与可编辑安装

本文解释 Python 如何寻找模块、`PYTHONPATH` 的作用，以及为什么执行可编辑安装后通常不再需要手动设置 `PYTHONPATH`。

## 1. 问题背景

本项目包含以下目录：

```text
train-llm-from-scratch/
├── config/
├── data_loader/
├── scripts/
├── src/
└── ui/
```

训练脚本会导入项目根目录下的模块，例如：

```python
from config.post_training_config import SFTConfig
from data_loader.sft_dataset import get_sft_batch_iterator
from src.models.transformer import Transformer
```

执行 `import` 时，Python 必须先知道应该到哪些目录中寻找 `config`、`data_loader` 和 `src`。

## 2. Python 如何寻找模块

Python 会按照 `sys.path` 中记录的目录依次寻找模块：

```python
import sys

for path in sys.path:
    print(path)
```

`sys.path` 通常包含：

- 当前运行方式决定的脚本目录或当前目录；
- Python 标准库目录；
- 当前虚拟环境的 `site-packages`；
- `PYTHONPATH` 环境变量额外指定的目录；
- 通过 `pip` 安装并注册的项目路径。

如果这些位置都不包含目标模块，Python 就会报错：

```text
ModuleNotFoundError: No module named 'src'
```

## 3. `PYTHONPATH` 是什么

`PYTHONPATH` 是一个环境变量，用于给 Python 的模块搜索路径添加额外目录。

在项目根目录执行：

```bash
PYTHONPATH=. python scripts/train_sft.py
```

其中：

- `PYTHONPATH=.` 表示把当前目录加入 Python 的模块搜索路径；
- `.` 表示当前工作目录；
- `python scripts/train_sft.py` 是在该环境变量设置下执行的命令。

如果当前目录是项目根目录，Python 就可以在其中找到 `src/`、`config/`、`data_loader/` 和 `ui/`。

这种写法只对当前这一条命令生效，不会永久修改终端环境。

## 4. 临时设置与导出 `PYTHONPATH`

### 仅对一条命令生效

推荐在临时运行时使用：

```bash
PYTHONPATH=. python scripts/train_sft.py
```

运行结束后，这个设置就失效了。

### 对当前终端会话生效

也可以先导出环境变量：

```bash
export PYTHONPATH="$PWD"
python scripts/train_sft.py
python scripts/train_dpo.py
```

这里的 `$PWD` 是当前目录的绝对路径。该设置对当前终端以及它启动的子进程有效，关闭终端后通常会失效。

查看当前设置：

```bash
echo "$PYTHONPATH"
```

取消设置：

```bash
unset PYTHONPATH
```

如果需要加入多个目录，在 macOS 和 Linux 中使用冒号分隔：

```bash
export PYTHONPATH="/path/to/project-a:/path/to/project-b"
```

## 5. 为什么直接运行脚本有时找不到项目模块

执行下面的命令时：

```bash
python scripts/train_sft.py
```

Python 会把脚本所在的 `scripts/` 目录作为重要的模块搜索位置，但项目中的 `src/` 和 `config/` 与 `scripts/` 是同级目录，而不是它的子目录。

如果项目根目录没有通过其他方式加入 `sys.path`，下面的导入就可能失败：

```python
from src.models.transformer import Transformer
```

`PYTHONPATH=.` 的作用就是明确地把项目根目录补充到搜索路径中。

需要注意：`.` 取决于当前工作目录。如果不在项目根目录执行，`.` 指向的就不是本项目。

例如下面的用法可能仍然失败：

```bash
cd scripts
PYTHONPATH=. python train_sft.py
```

因为此时 `.` 是 `scripts/`，而不是项目根目录。

## 6. 什么是可编辑安装

在项目根目录执行：

```bash
pip install -e .
```

其中：

- `pip install` 表示安装 Python 项目；
- `-e` 是 `--editable` 的缩写，表示可编辑模式；
- `.` 表示安装当前目录中的项目；
- 项目的安装信息由 `pyproject.toml` 描述。

普通安装通常会把项目代码复制或构建到 Python 环境的 `site-packages` 中。可编辑安装则会在 Python 环境中注册一个指向当前源码目录的映射。

因此，Python 知道应该到当前仓库中寻找这些包：

```python
import config
import data_loader
import src
import ui
```

修改仓库中的 Python 源码后，下一次运行程序就能直接使用新代码，一般不需要重新安装。

## 7. 为什么可编辑安装后可以省略 `PYTHONPATH`

执行可编辑安装前，项目根目录可能不在 Python 的默认搜索路径中，因此需要：

```bash
PYTHONPATH=. python scripts/train_sft.py
```

执行以下命令后：

```bash
pip install -e .
```

项目已经注册到当前 Python 环境，通常可以直接运行：

```bash
python scripts/train_sft.py
```

两种方式的核心作用可以概括为：

```text
PYTHONPATH=.       临时告诉 Python 项目源码在哪里
pip install -e .   把项目注册到当前 Python 环境中
```

## 8. 必须安装到正确的 Python 环境

`pip install -e .` 只会影响执行该命令时所使用的 Python 环境。

推荐先创建并激活虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

使用 `python -m pip` 可以降低 `python` 和 `pip` 指向不同环境的风险。

检查它们的位置：

```bash
which python
python -m pip --version
```

如果使用 A 环境安装，却使用 B 环境运行，B 环境仍然可能报 `ModuleNotFoundError`。

## 9. 如何验证可编辑安装是否成功

查看项目是否已安装：

```bash
python -m pip show train-llm-from-scratch
```

测试模块导入：

```bash
python -c "import config, data_loader, src, ui; print('import success')"
```

查看模块实际来自哪里：

```bash
python -c "import src; print(src.__file__)"
```

如果是可编辑安装，输出路径通常指向当前仓库中的源码文件。

## 10. 修改哪些内容后需要重新安装

修改普通 Python 源码通常不需要重新安装，例如：

```text
src/models/transformer.py
src/post_training/sft.py
scripts/train_sft.py
```

但修改以下内容后，建议重新运行 `python -m pip install -e .`：

- `pyproject.toml` 中的依赖；
- 项目包列表或安装配置；
- 命令行入口；
- 需要编译的扩展模块。

## 11. 常见问题

### 在错误的目录执行 `PYTHONPATH=.`

错误示例：

```bash
cd scripts
PYTHONPATH=. python train_sft.py
```

解决办法是回到项目根目录：

```bash
cd /path/to/train-llm-from-scratch
PYTHONPATH=. python scripts/train_sft.py
```

### 使用了错误的虚拟环境

确认当前 Python：

```bash
which python
python -c "import sys; print(sys.executable)"
```

然后在同一个环境中安装：

```bash
python -m pip install -e .
```

### 源码能修改，但新增包无法导入

如果新增了包目录，要确保其中包含需要的 `__init__.py`，并检查 `pyproject.toml` 是否把它包含在安装范围内。修改安装配置后重新执行：

```bash
python -m pip install -e .
```

## 12. 本项目推荐的使用方式

初次配置环境时，在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

以后激活同一个虚拟环境，就可以直接运行：

```bash
python scripts/train_sft.py
python scripts/train_dpo.py
python scripts/chat.py --help
```

如果只想临时执行一次、不想安装项目，也可以使用：

```bash
PYTHONPATH=. python scripts/train_sft.py
```

