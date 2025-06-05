SAVE_DIR="checkpoints/output"

torchrun --nproc_per_node=8 train.py \
    --train_dataset "5_000 @ Scannetpp(split='train', ROOT='data/scannetpp_processed', aug_crop=16, resolution=(518, 518)) + 5_000 @ Scannet(split='train', ROOT='data/scannet_processed', aug_crop=16, resolution=(518, 518))" \
    --test_dataset "100 @ Scannet(split='val', ROOT='data/scannet_processed', resolution=(518, 518), seed=777)" \
    --train_criterion "GaussianLoss()" \
    --test_criterion "GaussianLoss()" \
    --lr 0.0001 \
    --min_lr 1e-06 \
    --warmup_epochs 10 \
    --epochs 100 \
    --batch_size 2 \
    --accum_iter 2 \
    --save_freq 1 \
    --keep_freq 5 \
    --eval_freq 1 \
    --output_dir $SAVE_DIR
