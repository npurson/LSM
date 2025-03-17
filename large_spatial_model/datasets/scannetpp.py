import os
import os.path as osp
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
import numpy as np
import cv2
from dust3r.utils.image import imread_cv2

from ..datasets_preprocess.scannetpp_preprocess import calculate_iou


class Scannetpp(BaseStereoViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        # assert self.split == 'train' # just for training
        self.num_views = 3 # render third view
        self._load_data()
        
    def _load_data(self):
        # Traverse all the folders in the data_root
        # scene_names = [folder for folder in os.listdir(self.ROOT) if os.path.isdir(os.path.join(self.ROOT, folder))]

        # if self.split == 'train':
        #     split_file = 'splits/nvs_sem_train_v1.txt'
        # elif self.split == 'val':
        #     split_file = 'splits/nvs_sem_val.txt'
        split_file = 'splits/nvs_sem_train_v1.txt'
        with open(osp.join(self.ROOT, split_file), 'r') as f:
            scene_names = [line.strip() for line in f.readlines()]
        if self.split == 'train':
            scene_names = scene_names[:200]
        elif self.split == 'val':
            scene_names = scene_names[:5]

        # Filter out scenes without scene_data.npz
        valid_scenes = []
        for scene_name in scene_names:
            scene_data_path = osp.join(self.ROOT, scene_name, "scene_data.npz")
            if osp.exists(scene_data_path):
                valid_scenes.append(scene_name)
            else:
                print(f"Skipping {scene_name}: scene_data.npz not found")
        scene_names = valid_scenes
        scene_names.sort()

        # merge all pairs and images
        pairs = [] # (scene_name, image_idx1, image_idx2)
        images = {} # (scene_name, image_idx) -> image_path
        for scene_name in scene_names:
            scene_path = osp.join(self.ROOT, scene_name, "scene_data.npz")
            scene_data = np.load(scene_path)
            pairs.extend([(scene_name, *pair) for pair in scene_data['pairs']])
            images.update({(scene_name, idx): path for idx, path in enumerate(scene_data['images'])})
        self.pairs = pairs
        self.images = images
        
    def __len__(self):
        return len(self.pairs)
    
    def _get_views(self, idx, resolution, rng):
        scene_name, image_idx1, image_idx2, _ = self.pairs[idx]
        image_idx1 = int(image_idx1)
        image_idx2 = int(image_idx2)
        views = []
        interval = image_idx2 - image_idx1
        src_view_idxes = [idx_ for idx_ in range(image_idx1 - interval * 2, image_idx2 + interval * 2, max(interval // 2, 1)) if idx_ not in [image_idx1, image_idx2]]
        src_view_idxes = [image_idx1, image_idx2] + src_view_idxes
        for i, view_idx in enumerate(src_view_idxes):
            if view_idx < 0 or view_idx >= len(self.images):
                continue
            if (scene_name, view_idx) not in self.images:
                print(f"Skipping {scene_name} {view_idx}: image not found")
                continue

            basename = self.images[(scene_name, view_idx)]
            # Load RGB image
            rgb_path = osp.join(self.ROOT, scene_name, 'images', f'{basename}.jpg')
            rgb_image = imread_cv2(rgb_path)
            # Load depthmap
            depthmap_path = osp.join(self.ROOT, scene_name, 'depths', f'{basename}.png')
            depthmap = imread_cv2(depthmap_path, cv2.IMREAD_UNCHANGED)
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            # Load camera parameters
            # meta_path = osp.join(self.ROOT, scene_name, 'images', f'{basename}.npz')
            # meta = np.load(meta_path)
            # intrinsics = meta['camera_intrinsics']
            # camera_pose = meta['camera_pose']
            meta_path = osp.join(self.ROOT, scene_name, 'scene_metadata.npz')
            meta = np.load(meta_path)
            image_idx = np.argwhere(meta['images'] == basename).item()
            camera_pose = meta['trajectories'][image_idx]
            intrinsics = meta['intrinsics'][image_idx]
            # crop if necessary
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx)

            if len(views) >= 2:
                if len(views) == 2:
                    v1_args = (views[0]['depthmap'], views[0]['camera_pose'], views[0]['camera_intrinsics'])
                    v2_args = (views[1]['depthmap'], views[1]['camera_pose'], views[1]['camera_intrinsics'])
                iou_1 = calculate_iou(depthmap, camera_pose, intrinsics, *v1_args)
                iou_2 = calculate_iou(*v1_args, depthmap, camera_pose, intrinsics)
                iou_3 = calculate_iou(depthmap, camera_pose, intrinsics, *v2_args)
                iou_4 = calculate_iou(*v2_args, depthmap, camera_pose, intrinsics)
                iou = (iou_1 + iou_2 + iou_3 + iou_4) / 4

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='ScanNet++',
                label=scene_name + '_' + basename,
                instance=f'{str(idx)}_{str(view_idx)}',
                iou=iou if i >= 2 else 1
            ))
        return views

if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = Scannetpp(split='train', ROOT="data/scannetpp_processed", resolution=224, aug_crop=16)

    print(len(dataset))

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 3
        print(view_name(views[0]), view_name(views[1]), view_name(views[2]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1, 2]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1, 2]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx*255, (1 - idx)*255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()