CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
        --master_port 47751 \
        --nproc_per_node=1 main.py \
        --batch_size 1 \
        --num_queries 200 \
        --eval \
        --dataset_config configs/vidvrd_tag_eval.json
        