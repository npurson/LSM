SAVE_DIR="checkpoints/output"

python test.py \
    --test_dataset "TestDataset(split='test', ROOT='/home/users/haoyi.jiang/data/scannet_test', resolution=(224, 224), seed=777)" \
    --test_criterion "TestLoss(pose_align_steps=100)" \
    --pretrained checkpoint-last.pth
