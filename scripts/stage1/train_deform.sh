CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
        --master_port 47749 \
        --nproc_per_node=1 main.py \
        --accumulate_steps 1 \
        --lr_backbone 1e-5 --lr 5e-5 --num_queries 200 \
        --dataset_config configs/vidvrd_stage1_deform.json
        