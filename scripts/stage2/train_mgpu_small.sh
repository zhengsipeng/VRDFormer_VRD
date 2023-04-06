CUDA_VISIBLE_DEVICES=3,4,5,6,7 CUBLAS_WORKSPACE_CONFIG=:16:8 python -m torch.distributed.launch \
        --master_port 47769 \
        --nproc_per_node=5 main.py \
        --num_workers 5 \
        --accumulate_steps 4 \
        --resolution small \
        --batch_size 1 --lr_backbone 1e-5 --lr 5e-5 --num_queries 200 \
        --dataset_config configs/vidvrd_stage2_mdetr.json
        