# 使用 JSONL 日志监视训练过程

本项目可以选择把训练指标同步到 Weights & Biases，但即使关闭 W&B，也会始终把指标写入本地 JSONL 文件。因此，在网络不方便或者不想依赖外部服务时，可以直接通过 JSONL 日志监视训练过程。

## 1. 什么是 JSONL

JSONL 是 JSON Lines 的缩写。它和普通 JSON 的主要区别是：每一行都是一个独立的 JSON 对象。

例如：

```jsonl
{"step": 0, "wall": 1780000000.1, "train_loss": 10.82, "lr": 1e-05}
{"step": 20, "wall": 1780000062.4, "train_loss": 8.31, "lr": 5e-05}
{"step": 40, "wall": 1780000124.8, "train_loss": 7.56, "lr": 9e-05}
```

这种格式很适合训练日志，因为程序每次只需要向文件末尾追加一行，不需要反复读取和重写整个文件。

其中：

- `step`：训练步数或迭代编号；
- `wall`：写入日志时的 Unix 时间戳；
- 其他字段：当前训练阶段记录的 loss、准确率、reward、KL 等指标。

## 2. 本项目如何写入 JSONL

统一的日志实现位于：

```text
src/post_training/logging_utils.py
```

核心逻辑可以简化为：

```python
record = {"step": step, "wall": time.time(), **metrics}
self._fh.write(json.dumps(record) + "\n")
self._fh.flush()
```

`flush()` 会在每次记录后尽快把内容刷新到文件，因此可以在训练仍在运行时使用 `tail -f` 查看最新数据。

即使 W&B 初始化失败，代码也会回退到仅写 JSONL，不应该因此终止训练。

## 3. 日志保存在哪里

默认配置为：

```python
log_dir = "/ephemeral/logs"
```

日志文件名由训练阶段和时间戳组成：

```text
/ephemeral/logs/pretrain_时间戳.jsonl
/ephemeral/logs/sft_时间戳.jsonl
/ephemeral/logs/reward_时间戳.jsonl
/ephemeral/logs/dpo_dpo_时间戳.jsonl
/ephemeral/logs/ppo_时间戳.jsonl
/ephemeral/logs/grpo_时间戳.jsonl
```

`/ephemeral` 是作者训练机器上的大容量磁盘路径。在本地电脑或其他服务器上，这个目录可能不存在或者没有写权限。

更通用的做法是在项目根目录下使用 `logs/`：

```bash
python scripts/train_sft.py --log_dir ./logs --use_wandb false
```

也可以修改 `configs/base.json`：

```json
{
  "log_dir": "./logs",
  "use_wandb": false
}
```

建议每台机器都明确检查 `log_dir`，避免训练很久以后才发现日志写到了意料之外的位置。

## 4. 查看已有日志文件

列出所有日志：

```bash
ls -lh ./logs
```

按照修改时间排序：

```bash
ls -lt ./logs
```

查找所有 JSONL 日志：

```bash
find ./logs -type f -name '*.jsonl' -print
```

查看某个文件的前几行：

```bash
head -n 5 ./logs/sft_时间戳.jsonl
```

查看最后十行：

```bash
tail -n 10 ./logs/sft_时间戳.jsonl
```

## 5. 实时监视训练

最简单的方法是：

```bash
tail -f ./logs/sft_时间戳.jsonl
```

训练程序每写入一行，终端就会显示一条新记录。按 `Ctrl+C` 可以停止查看，不会停止另一个终端中运行的训练进程。

如果想自动查看最新的 SFT 日志，可以先获取最新文件：

```bash
latest_log=$(ls -t ./logs/sft_*.jsonl | head -n 1)
echo "$latest_log"
tail -f "$latest_log"
```

注意：应该在训练已经创建日志文件后再执行，否则通配符可能匹配不到文件。

## 6. 使用 `jq` 格式化日志

如果系统安装了 `jq`，可以让 JSON 更容易阅读：

```bash
tail -f ./logs/sft_时间戳.jsonl | jq -c .
```

只显示 SFT 的步数、loss 和学习率：

```bash
jq -r '
  select(.train_loss != null)
  | [.step, .train_loss, .lr]
  | @tsv
' ./logs/sft_时间戳.jsonl
```

只查看 PPO 的 reward 和 KL：

```bash
jq -r '
  select(.reward != null)
  | [.step, .reward, .kl_ref]
  | @tsv
' ./logs/ppo_时间戳.jsonl
```

`jq` 只是辅助查看工具，不是训练依赖；没有安装也不影响训练。

## 7. 各训练阶段会记录什么

不同阶段的目标不同，因此不能只盯着一个统一的 loss。

### 预训练

常见字段：

- `train_loss`：当前训练 loss；
- `lr`：当前学习率；
- `tok_per_s`：每秒处理的 token 数；
- `eval_train`：训练集抽样评估 loss；
- `eval_dev`：验证集抽样评估 loss。

重点观察：

- 训练初期 loss 是否明显下降；
- `eval_dev` 是否同步下降；
- `tok_per_s` 是否大致稳定；
- 是否出现 `NaN`、`inf` 或突然增大的 loss。

不能只看 `train_loss`。如果训练 loss 持续下降但验证 loss 上升，可能已经开始过拟合。

### SFT

常见字段：

- `train_loss`：只在 assistant token 上计算的 masked loss；
- `lr`：学习率；
- `dev_loss`：验证集 masked loss。

SFT 的 loss 和预训练 loss 不是完全相同的统计口径，不适合直接比较数值大小。

### 奖励模型

常见字段：

- `train_loss`：Bradley–Terry pairwise loss；
- `train_acc`：训练 preference accuracy；
- `test_acc`：评测 preference accuracy；
- `test_margin`：chosen reward 与 rejected reward 的平均差值；
- `lr`：学习率。

重点观察 `test_acc`，因为仅有训练准确率升高可能意味着过拟合。

### DPO、ORPO 和 KTO

常见字段：

- `train_loss`：当前偏好优化目标的 loss；
- `train_acc`：隐式奖励选择 chosen response 的比例；
- `r_chosen`：chosen response 的平均隐式奖励；
- `r_rejected`：rejected response 的平均隐式奖励；
- `test_acc`：评测集隐式奖励准确率；
- `test_margin`：chosen 与 rejected 的奖励差；
- `lr`：学习率。

理想情况下，`r_chosen - r_rejected` 应逐渐变为正值并保持稳定，但不能只根据训练集 margin 判断模型生成质量。

### PPO

常见字段：

- `reward`：当前 rollout 的平均任务奖励；
- `kl_ref`：策略相对 SFT reference model 的偏移；
- `policy_loss`：PPO policy loss；
- `value_loss`：价值模型 loss；
- `clipfrac`：触发 PPO clipping 的 token 比例；
- `resp_len`：平均回答长度；
- `gsm8k_acc`：阶段性 GSM8K 准确率。

重点观察：

- reward 是否整体上升，而不是只看单次波动；
- `kl_ref` 是否保持受控；
- `clipfrac` 是否长期接近 0 或长期过高；
- 回答长度是否异常增长或快速缩短；
- reward 上升时，真实评测指标是否也上升。

reward 上升但评测效果下降，可能意味着 reward hacking。

### GRPO

常见字段：

- `reward`：group completion 的平均奖励；
- `informative_groups`：组内奖励存在差异的 group 比例；
- `loss`：GRPO loss；
- `kl`：策略相对 reference model 的偏移；
- `resp_len`：平均回答长度；
- `gsm8k_acc`：阶段性 GSM8K 准确率。

如果 `informative_groups` 长期接近 0，说明同一问题采样出的答案几乎全部同分，GRPO 缺少有效的相对优势信号。

## 8. 使用 Python 读取 JSONL

不依赖 Pandas：

```python
import json

path = "logs/sft_时间戳.jsonl"

records = []
with open(path, encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line:
            records.append(json.loads(line))

for record in records[-5:]:
    print(record)
```

只提取包含 `train_loss` 的记录：

```python
points = [
    (record["step"], record["train_loss"])
    for record in records
    if "train_loss" in record
]
```

## 9. 绘制 loss 曲线

安装绘图库：

```bash
python -m pip install matplotlib
```

示例：

```python
import json
import matplotlib.pyplot as plt

path = "logs/sft_时间戳.jsonl"
steps = []
losses = []

with open(path, encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)
        if "train_loss" in record:
            steps.append(record["step"])
            losses.append(record["train_loss"])

plt.plot(steps, losses)
plt.xlabel("step")
plt.ylabel("train loss")
plt.title("SFT training loss")
plt.grid(True)
plt.show()
```

训练 loss 通常有较大随机波动。为了更容易观察趋势，可以增加滑动平均，但不要用过度平滑掩盖异常尖峰。

## 10. JSONL 不能替代所有监控

当前项目的 JSONL 主要记录算法指标，并没有完整记录以下系统信息：

- GPU 显存使用量；
- GPU 利用率；
- GPU 温度和功耗；
- CPU、内存和磁盘使用量；
- 数据加载耗时；
- 梯度范数；
- 是否发生 CUDA OOM。

训练时建议同时打开另一个终端：

```bash
watch -n 1 nvidia-smi
```

macOS 没有 `watch` 时，可以循环执行：

```bash
while true; do
  clear
  nvidia-smi
  sleep 1
done
```

如果使用云服务器，还应定期检查磁盘空间：

```bash
df -h
du -sh ./logs
```

## 11. 重要注意事项

### JSONL 不是每一步都会写入

训练脚本通常隔若干步才调用一次 `logger.log()`。因此日志中的 `step` 跳跃是正常现象，不表示训练跳过了中间步骤。

### 多卡训练通常只有 rank 0 写日志

本项目只让主进程创建 `MetricsLogger`，避免多个进程同时写入一个文件。因此，多卡训练时看到的是聚合指标或 rank 0 负责记录的指标，而不是每张 GPU 各有一个日志文件。

### 不要横向比较不同目标的 loss

预训练 CE、SFT masked CE、reward-model pairwise loss、DPO loss、PPO loss 和 GRPO loss 含义不同。数值更小不代表一个阶段比另一个阶段更好。

### 不要只看训练指标

训练 loss 或 reward 变好，不等于真实模型能力一定提升。至少还要结合：

- 验证集指标；
- 固定 benchmark；
- 固定 prompt 的定期生成样例；
- reward、KL 和回答长度之间的关系。

### 保留标准输出

有些信息只打印到终端，不一定写入 JSONL。建议同时保存控制台输出：

```bash
python scripts/train_sft.py \
  --log_dir ./logs \
  --use_wandb false \
  2>&1 | tee ./logs/sft_console.log
```

这样既能查看 JSONL 中的结构化指标，也能保留报错、checkpoint 路径和普通进度信息。

### 异常结束时检查最后一行

本项目每次写入后都会调用 `flush()`，通常不会丢失很多日志。但如果进程被强制终止，最后一行仍可能不完整。

可以检查文件是否仍能逐行解析：

```bash
python -c '
import json, sys
for number, line in enumerate(open(sys.argv[1]), 1):
    try:
        json.loads(line)
    except json.JSONDecodeError as error:
        print(f"invalid line {number}: {error}")
' ./logs/sft_时间戳.jsonl
```

不要因为最后一行损坏就删除整个日志文件；通常只需要忽略或修复最后一行。

### 定期备份日志和 checkpoint

JSONL 文件通常不大，但 checkpoint 可能很大。训练时间较长时，应把日志、配置和关键 checkpoint 一起保存，以便复现实验。

建议为每次实验保留：

```text
experiment-name/
├── config.json
├── metrics.jsonl
├── console.log
├── checkpoint.pt
└── notes.md
```

## 12. 推荐的本地监控流程

第一个终端启动训练：

```bash
python scripts/train_sft.py \
  --log_dir ./logs \
  --use_wandb false \
  2>&1 | tee ./logs/sft_console.log
```

第二个终端监视最新 JSONL：

```bash
latest_log=$(ls -t ./logs/sft_*.jsonl | head -n 1)
tail -f "$latest_log"
```

第三个终端监视 GPU：

```bash
watch -n 1 nvidia-smi
```

训练结束后，再使用 Python、Matplotlib 或 Pandas读取 JSONL，绘制完整曲线并比较不同实验。

