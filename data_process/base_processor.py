from typing import List, Dict
import torch
from abc import abstractmethod
import numpy as np
import os
import json
from torchvision.utils import save_image
from concurrent.futures import ProcessPoolExecutor
import tqdm
import multiprocessing as mp
import random
import cv2

class BaseSceneProcessorConfig:
    def __init__(self, root_dir: str, save_dir: str, name: str, device: torch.device, num_workers: int = 16):
        self.root_dir = root_dir
        self.save_dir = save_dir
        self.name = name
        self.device = device
        self.load_frame_parallel = False
        self.target_height = 480
        self.target_width = 640
        self.num_workers = num_workers

        # create dirs
        os.makedirs(self.save_dir, exist_ok=True)

class BaseSceneProcessor:
    def __init__(self, config: BaseSceneProcessorConfig):
        # Set multiprocessing start method to 'spawn'
        mp.set_start_method('spawn', force=True)
        # set random seed
        random.seed(0)
        np.random.seed(0)
        self.config = config
        self.scene_paths = self.get_all_scene_paths()
    
    def get_scene_image_save_path(self, scene_name: str) -> str:
        """
        Get scene image save path
        """
        return os.path.join(self.config.image_save_dir, scene_name)
    
    def get_image_save_path(self, scene_name: str, frame_idx: int) -> str:
        """
        Get image save path
        """
        return os.path.join(self.get_scene_image_save_path(scene_name), f'{frame_idx:06d}.png')
    
    def get_all_scene_paths(self) -> List[str]:
        """
        Get a list of scene names from the root directory
        """
        pass
    
    def load_and_process_scene(self, scene_path: str) -> dict:
        """
        Load and process scene data from scene_path
        Args:
            scene_path: str, path to the scene
        Returns:
            scene_data: dict, including processed frames, instance_id_to_class_map
        """
        # 1. Get frame paths and instance mapping
        frame_paths = self.get_all_frame_paths(scene_path)
        intrinsics = self.get_intrinsics(scene_path)
        
        # 2. Load all frames
        frames_data = {}
        for frame_idx, frame_path in frame_paths.items():
            frames_data[frame_idx] = self.load_single_frame(frame_path)

        # 3. Stack and process frames
        depth_data = torch.stack([frame_data['depth_data'] for frame_data in frames_data.values()], axis=0)
        color_data = torch.stack([frame_data['color_data'] for frame_data in frames_data.values()], axis=0)
        pose_data = torch.stack([frame_data['pose_data'] for frame_data in frames_data.values()], axis=0)
        
        # 4. Process image size and resize
        batch_size, original_h, original_w = depth_data.shape
        h_ratio = self.config.target_height / original_h
        w_ratio = self.config.target_width / original_w
        ratio = max(h_ratio, w_ratio)
        new_h, new_w = int(original_h * ratio), int(original_w * ratio)
        
        # Calculate adjusted intrinsics first
        new_intrinsic = torch.clone(intrinsics)
        new_intrinsic[0, 0] *= ratio
        new_intrinsic[1, 1] *= ratio
        start_x = (new_w - self.config.target_width) // 2
        start_y = (new_h - self.config.target_height) // 2
        new_intrinsic[0, 2] = (intrinsics[0, 2] * ratio) - start_x
        new_intrinsic[1, 2] = (intrinsics[1, 2] * ratio) - start_y
        
        # Resize images
        depth_data = torch.nn.functional.interpolate(depth_data.unsqueeze(1), size=(new_h, new_w), mode='nearest')
        color_data = torch.nn.functional.interpolate(color_data.permute(0, 3, 1, 2), size=(new_h, new_w), mode='bilinear')
        
        # Crop images
        depth_data = depth_data[:, :, start_y:start_y + self.config.target_height, start_x:start_x + self.config.target_width]
        color_data = color_data[:, :, start_y:start_y + self.config.target_height, start_x:start_x + self.config.target_width]
        
        torch.cuda.empty_cache()
        
        return {
            "frames": {
                'color_data': color_data,
                'depth_data': depth_data,
                'pose_data': pose_data,
            },
            "intrinsics": new_intrinsic,
            "frame_num": batch_size,
            "scene_name": scene_path.split('/')[-1]
        }
    
    @abstractmethod
    def get_intrinsics(self, scene_path: str) -> dict:
        """
        Get intrinsics from scene_path
        """
        pass
    
    @abstractmethod
    def get_all_frame_paths(self, scene_path: str) -> Dict[str, List[str]]:
        """
        Get all frame paths from scene_path
        input:
            scene_path: str, path to the scene
        return:
            Dict: {
                "frame_idx": [depth_path, color_path, pose_path]
            }
        """
        pass
    
    @abstractmethod
    def load_single_frame(self, frame_path: str) -> dict:
        """
        Load frame data from frame_path
        Args:
            frame_path: str, path to the frame
        Returns:
            dict: {
                "frame_idx": [depth_data, color_data, pose_data]
            }
        """
        pass

    def process_single_scene(self, scene_path: str) -> dict:
        # Load and process scene data
        scene_data = self.load_and_process_scene(scene_path)
        # save scene data
        scene_name = scene_data['scene_name']
        scene_save_path = os.path.join(self.config.save_dir, scene_name)
        scene_color_save_path = os.path.join(scene_save_path, 'color')
        scene_depth_save_path = os.path.join(scene_save_path, 'depth')
        scene_pose_save_path = os.path.join(scene_save_path, 'pose')
        os.makedirs(scene_color_save_path, exist_ok=True)
        os.makedirs(scene_depth_save_path, exist_ok=True)
        os.makedirs(scene_pose_save_path, exist_ok=True)
        # save color data
        for frame_idx, color_data in enumerate(scene_data['frames']['color_data']):
            color_save_path = os.path.join(scene_color_save_path, f'{frame_idx:06d}.png')
            # Convert to uint8 range [0, 255] if needed
            color_data = color_data.cpu().numpy()
            color_data = color_data.transpose(1, 2, 0).astype(np.uint8)
            # Save as JPG for color images
            cv2.imwrite(color_save_path, color_data)

        # save depth data
        for frame_idx, depth_data in enumerate(scene_data['frames']['depth_data']):
            depth_save_path = os.path.join(scene_depth_save_path, f'{frame_idx:06d}.png')
            # ScanNet depth is in millimeters, stored as 16-bit PNG
            # Ensure depth is in uint16 format with proper scaling
            depth_data = depth_data.cpu().numpy()
            depth_data = depth_data.transpose(1, 2, 0).astype(np.uint16)
            # Save as 16-bit PNG for depth images
            cv2.imwrite(depth_save_path, depth_data)
        # save meta data
        for frame_idx, pose_data in enumerate(scene_data['frames']['pose_data']):
            pose_save_path = os.path.join(scene_pose_save_path, f'{frame_idx:06d}.npz')
            meta_data = {
                'camera_intrinsics': scene_data['intrinsics'].cpu().numpy(),
                'camera_pose': pose_data.cpu().numpy()
            }
            np.savez(pose_save_path, **meta_data)

        return True
    
    def _process_scene_with_gpu(self, scene_path: str, gpu_id: int) -> dict:
        """
        Process a single scene with specified GPU
        Args:
            scene_path: str, path to the scene
            gpu_id: int, GPU device ID
        Returns:
            dict: processed conversations
        """
        # Set CUDA device for this process
        torch.cuda.set_device(gpu_id)
        try:
            result = self.process_single_scene(scene_path)
            # Clean up GPU memory after processing
            torch.cuda.empty_cache()
            return result
        except Exception as e:
            print(f"Error processing scene {scene_path} on GPU {gpu_id}: {e}")
            # Clean up GPU memory in case of error
            torch.cuda.empty_cache()
            return None

    def process_all_scenes_serial(self):
        """
        Process all scenes in serial
        """
        for scene_path in tqdm.tqdm(self.scene_paths):
            self.process_single_scene(scene_path)
        
    def process_all_scenes_parallel(self) -> list:
        """
        Process all scenes in parallel with tasks distributed across available GPUs
        """
        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPU available for processing")
        
        # Calculate workers per GPU
        workers_per_gpu = max(1, self.config.num_workers // num_gpus)
        total_workers = workers_per_gpu * num_gpus
        
        print(f"Processing {len(self.scene_paths)} scenes with {total_workers} workers across {num_gpus} GPUs "
              f"({workers_per_gpu} workers per GPU)...")
        
        # Distribute scenes across GPUs
        with ProcessPoolExecutor(max_workers=total_workers) as executor:
            # Create tasks with GPU assignments
            futures = []
            for i, scene_path in enumerate(self.scene_paths):
                gpu_id = i % num_gpus  # Distribute scenes across GPUs in round-robin fashion
                futures.append(
                    executor.submit(self._process_scene_with_gpu, scene_path, gpu_id)
                )
            
            # Process results with progress bar
            for future in tqdm.tqdm(futures, total=len(self.scene_paths)):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in processing scene: {e}")
                    continue
