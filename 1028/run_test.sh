# python scripts/train_transformer.py \
#   --preset 3b --arch classic --device cuda \
#   --train-path data/pile/pile_train_00.h5 \
#   --dev-path data/pile/pile_dev.h5 \
#   --out-path models/3b_pile_20k.pt \
#   --steps 20000 --eval-every 1000 --set t_eval_iters=20 \
#   --amp --amp-dtype bf16 \
#   --checkpoint-every 1000 --keep-last 3 \
#   --set t_context_length=128 \
#   --batch-size 4 --grad-accum 8

# cd /gpu-nas/experiment_workspace/fengpinqi/code/train-llm-from-scratch


# 默认配置，约406M，上下文1024
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# NCCL_DEBUG=WARN \
# torchrun --standalone --nproc_per_node=8 scripts/pretrain_base.py


# 约3.43B，上下文512
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_DEBUG=WARN \
torchrun --standalone --nproc_per_node=8 scripts/pretrain_base.py \
  --config configs/pretrain_3b.json