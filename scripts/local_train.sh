SAVE_DIR="checkpoints/output"

python3 train.py \
    --train_dataset "1 @ Scannetpp(split='train', ROOT='data/scannetpp_processed', aug_crop=16, resolution=(256, 256))" \
    --test_dataset "8_000 @ Scannetpp(split='val', ROOT='data/scannetpp_processed', resolution=(256, 256), seed=777)" \
    --train_criterion "GaussianLoss() + ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2) " \
    --test_criterion "GaussianLoss() + ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2) " \
    --num_workers 0 \
    --lr 0.0001 \
    --min_lr 1e-06 \
    --warmup_epochs 10 \
    --epochs 1 \
    --batch_size 1 \
    --accum_iter 2 \
    --save_freq 1 \
    --keep_freq 5 \
    --eval_freq 1 \
    --output_dir $SAVE_DIR \
    --pretrained /home/users/haoyi.jiang/repos/GauS3TR/checkpoint-best.pth
