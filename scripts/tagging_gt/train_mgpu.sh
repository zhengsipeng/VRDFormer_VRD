CUDA_VISIBLE_DEVICES=3,4,5,6,7 CUBLAS_WORKSPACE_CONFIG=:16:8 python -m torch.distributed.launch \
        --master_port 47761 \
        --nproc_per_node=5 main.py \
        --num_workers 10 \
        --accumulate_steps 4 \
        --batch_size 1 --lr_backbone 1e-5 --lr 5e-5 --num_queries 200 \
        --dataset_config configs/vidvrd_tag.json
        